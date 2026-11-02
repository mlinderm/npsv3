import atexit
import contextlib
import copy
import json
import logging
import os
import pathlib
import random
import re
import subprocess
import tempfile
from shlex import quote

import portalocker
import pysam

from npsv3.types import PathType
from npsv3.util.range import Range
from npsv3.util.sample import Sample

LOCK_CHECK_INTERVAL = 1

# Aligners that can stage their index in shared memory. bwa and bwa-mem3 share the same shared-memory registry (and
# `shm -d` with either drops every index in it), but their index formats are incompatible, so each aligner's index needs
# a distinct basename.
SHM_ALIGNERS = ("bwa-mem3", "bwa")


def _check_shm_aligner(aligner: str):
    if aligner not in SHM_ALIGNERS:
        msg = f"Aligner {aligner} does not support shared memory indices"
        raise ValueError(msg)


def _is_bwa_index_loaded(shared_name, aligner="bwa-mem3"):
    indices = subprocess.check_output(
        f"{aligner} shm -l", shell=True, universal_newlines=True, stderr=subprocess.DEVNULL
    )
    # Each line is "<name>\t<size>"
    return any(index.split("\t", 1)[0] == shared_name for index in indices.split("\n"))


def _write_lock_file(file, counts: dict):
    file.seek(0)
    file.truncate(0)
    json.dump(counts, file)
    # Docs suggest flushing and syncing the file: https://portalocker.readthedocs.io/en/latest/#tips
    file.flush()
    os.fsync(file.fileno())


def _bwa_index_unload(shared_name: str, lock_file: str, aligner="bwa-mem3"):
    while True:
        try:
            with portalocker.Lock(lock_file, mode="r+", check_interval=LOCK_CHECK_INTERVAL) as lock:
                current_counts = json.load(lock)
                assert current_counts[shared_name] > 0, "BWA index is already unloaded"
                current_counts[shared_name] -= 1

                for count in current_counts.values():
                    if count > 0:
                        # There is still an index file in use. Don't delete any indices, just write an updated lock file.
                        _write_lock_file(lock, current_counts)
                        return

                # No shared references in use, we can delete the references (bwa and bwa-mem3 delete all shared
                # references, including those staged by the other aligner)
                logging.info("Unloading all BWA indices in shared memory")
                subprocess.run(f"{aligner} shm -d", shell=True, check=True)
                os.unlink(lock_file)
                return
        except portalocker.LockException:
            continue

# Use the same lock file as npsv2 in case the jobs are running on the same node. Since `shm -d` (bwa or bwa-mem3)
# drops every index in shared memory, the shared reference counts are what prevent npsv2 (bwa) and npsv3 (bwa-mem3) from
# unloading each other's indices while they are in use.
def _bwa_index_load(reference, lock_file="/var/tmp/npsv2/bwa.lock", aligner="bwa-mem3") -> str | None:
    # Create lock directory if it doesn't exist
    os.makedirs(os.path.dirname(lock_file), mode=0o777, exist_ok=True)

    shared_name = os.path.basename(reference)
    while True:
        try:
            with portalocker.Lock(lock_file, mode="a+", check_interval=LOCK_CHECK_INTERVAL, timeout=120) as lock:
                logging.info("Holding lock on %s", lock_file)
                lock.seek(0)
                current_counts = json.loads(lock.read() or "{}")
                # Since we can't unload indices until none are needed, an index with reference count of 0 could still be loaded
                # if other references are in use
                if current_counts.get(shared_name, 0) > 0 or _is_bwa_index_loaded(shared_name, aligner):
                    logging.info("Incrementing reference count for %s", shared_name)
                    assert _is_bwa_index_loaded(shared_name, aligner)
                    current_counts[shared_name] += 1
                else:
                    logging.info("Loading %s index for %s into shared memory", aligner, reference)
                    subprocess.run(f"{aligner} shm {quote(reference)}", shell=True, check=True)
                    current_counts[shared_name] = 1

                # Write reference counts to lock file
                _write_lock_file(lock, current_counts)

                os.chmod(lock_file, 0o666)  # Ensure others can write to the lock file

                # Register handler to unload reference when no longer needed
                atexit.register(_bwa_index_unload, shared_name, lock_file, aligner)
            logging.info("Released lock on %s", lock_file)
            return shared_name
        except portalocker.LockException:
            logging.info("Unable to obtain lock on %s", lock_file)
            continue


def bwa_index_loaded(reference: str, load=False, aligner="bwa-mem3") -> str | None:
    """Check if bwa (or bwa-mem3) index is loaded in shared memory

    If BWA index is loaded into shared memory, this function implements reference counting with a lock file to enable
    multiple instances of npsv2/npsv3 to run on the same node. However, since indices loaded into shared memory are global,
    this management might conflict with other users who have loaded bwa indices into shared memory.

    The shared memory registry only records index names, not the aligner that staged them, so a bwa-mem3 index must
    have a different basename than any bwa index of the same reference (and not start with it, since npsv2 matches names
    by prefix).

    Args:
        reference (str): Path to reference file (or index prefix)
        load (bool, optional): Load index into shared memory if not present. Defaults to False.
        aligner (str, optional): Aligner whose index to check, "bwa-mem3" or "bwa". Defaults to "bwa-mem3".

    Returns:
        str: Shared reference name if index is loaded into shared memory, None otherwise
    """
    _check_shm_aligner(aligner)
    if load:
        return _bwa_index_load(reference, aligner=aligner)
    shared_name = os.path.basename(reference)
    return shared_name if _is_bwa_index_loaded(shared_name, aligner) else None


def _art_read_length(read_length, profile):
    """Make sure read length is compatible ART"""
    if profile in ("HS10", "HS20"):
        return min(read_length, 100)
    if profile in ("HS25", "HSXn", "HSXt"):
        return min(read_length, 150)
    return read_length


def simulate_variant_sequencing(
    fasta_path: str,
    hap_covg: float,
    sample: Sample,
    reference: str,
    shared_reference=None,
    dir=None,
    stats_path: str | None = None,
    region: Range | None = None,
    phase_vcf_path: str | None = None,
    aligner: str = "bwa-mem3",
    aligner_index: str | None = None,
    aligner_preset: str = "bcbio",
):
    shared_ref_arg = f"-S {quote(shared_reference)}" if shared_reference else ""
    aligner_index_arg = f"-x {quote(aligner_index)}" if aligner_index else ""
    stats_path_arg = f"-j {quote(stats_path)}" if stats_path else ""
    phase_vcf_arg = (
        f"-r {quote(str(region))} -v {quote(phase_vcf_path)} -N {sample.name}" if region and phase_vcf_path else ""
    )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".bam", dir=dir) as replicate_bam_file:
        replicate_bam = replicate_bam_file.name

    synth_commandline = f"synthBAM \
        -t {quote(dir)} \
        -R {quote(reference)} \
        {aligner_index_arg} \
        {shared_ref_arg} \
        {stats_path_arg} \
        -c {hap_covg:0.1f} \
        -m {sample.mean_insert_size} \
        -s {sample.std_insert_size} \
        -l {_art_read_length(sample.read_length, sample.sequencer)} \
        -p {sample.sequencer} \
        {phase_vcf_arg} \
        -i 1 \
        -a {quote(aligner)} \
        -b {quote(aligner_preset)} \
        {quote(fasta_path)} \
        {quote(replicate_bam)}"

    synth_result = subprocess.run(synth_commandline, shell=True, stderr=subprocess.PIPE, check=False)
    if synth_result.returncode != 0 or not os.path.exists(replicate_bam):
        msg = "Synthesis script failed to generate BAM"
        raise RuntimeError(msg)

    return replicate_bam


def augment_sample(original_sample: Sample, n, keep_original=True):
    new_samples = [original_sample] if keep_original else []

    # TODO: Apply augmentation approach like snorkel, with multiple augmentor functions. Potential augmentors
    # include coverage, insert-size distribution, GC distribution
    for _ in range(n - len(new_samples)):
        new_sample = copy.copy(original_sample)

        new_sample.mean_coverage = random.uniform(  # noqa: S311
            max(original_sample.mean_coverage * 0.5, 0), original_sample.mean_coverage + 0
        )
        # new_sample.mean_insert_size = random.uniform(original_sample.mean_insert_size - 75, original_sample.mean_insert_size + 75)
        # new_sample.std_insert_size = random.uniform(original_sample.std_insert_size - 30, original_sample.std_insert_size + 30)

        new_samples.append(new_sample)

    return new_samples

def split_bam_by_tag(bam_path: PathType, output_dir: PathType, tag: str) -> dict[str, PathType]:
    """Split BAM file into distinct files by tag

    Args:
        bam_path (PathType): Merged BAM path
        output_dir (PathType): Directory to write split BAM files
        tag (str): BAM tag (must be string or integer)

    Returns:
        dict[str, PathType]: Tag->Path mapping
    """
    bam_path = pathlib.Path(bam_path)
    output_dir = pathlib.Path(output_dir)
    with contextlib.chdir(output_dir):
        # These split files are small, immediately-consumed-then-discarded intermediates (one region's worth
        # of simulated reads per haplotype), so a low compression level trades a larger on-disk size we never
        # keep for less time spent deflating. `--write-index` has samtools build each output's index (a CSI,
        # for BAM) inline while it writes it, in the same process, rather than a second pysam.index() pass
        # that reopens and re-reads every split file from scratch -- profiling showed splitting itself (not
        # indexing) dominates this step, and both changes together roughly halve its cost.
        pysam.split(
            "-d", tag, "--output-fmt-option", "level=1", "--write-index", str(bam_path.absolute()), catch_stdout=False
        )

    # Extract tag from samtools split output files, e.g., <basename>_<tag>.bam, to { tag: path } mapping
    tag_pattern = fr"{bam_path.stem}_([^.]+)\.bam"

    tag_paths = {}
    for path in output_dir.iterdir():
        if match := path.is_file() and re.fullmatch(tag_pattern, path.name):
            tag_paths[match.group(1)] = path
    return tag_paths

