import atexit
import copy
import json
import logging
import os
import random
import subprocess
import tempfile
import typing
from shlex import quote

import portalocker

from npsv3.util.range import Range
from npsv3.util.sample import Sample

LOCK_CHECK_INTERVAL = 1


def _is_bwa_index_loaded(shared_name):
    indices = subprocess.check_output("bwa shm -l", shell=True, universal_newlines=True, stderr=subprocess.DEVNULL)
    for index in indices.split("\n"):
        if index.startswith(shared_name):
            return True
    return False


def _write_lock_file(file, counts: dict):
    file.seek(0)
    file.truncate(0)
    json.dump(counts, file)
    # Docs suggest flushing and syncing the file: https://portalocker.readthedocs.io/en/latest/#tips
    file.flush()
    os.fsync(file.fileno())


def _bwa_index_unload(shared_name: str, lock_file: str):
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

                # No shared references in use, we can delete the references (bwa deletes all shared references)
                logging.info("Unloading all BWA indices in shared memory")
                subprocess.run("bwa shm -d", shell=True, check=True)
                os.unlink(lock_file)
                return
        except portalocker.LockException:
            continue


def _bwa_index_load(reference, lock_file="/var/tmp/npsv3/bwa.lock") -> typing.Optional[str]:
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
                if current_counts.get(shared_name, 0) > 0 or _is_bwa_index_loaded(shared_name):
                    logging.info("Incrementing reference count for %s", shared_name)
                    # logging.info("Count: %d", current_counts.get(shared_name, 0))
                    # logging.info("Loaded: %d", _is_bwa_index_loaded(shared_name))
                    assert _is_bwa_index_loaded(shared_name)
                    current_counts[shared_name] += 1
                else:
                    logging.info("Loading BWA index for %s into shared memory", reference)
                    subprocess.run(f"bwa shm {reference}", shell=True, check=True)
                    current_counts[shared_name] = 1

                # Write reference counts to lock file
                _write_lock_file(lock, current_counts)

                os.chmod(lock_file, 0o666)  # Ensure others can write to the lock file

                # Register handler to unload reference when no longer needed
                atexit.register(_bwa_index_unload, shared_name, lock_file)
            logging.info("Released lock on %s", lock_file)
            return shared_name
        except portalocker.LockException:
            logging.info("Unable to obtain lock on %s", lock_file)
            continue


def bwa_index_loaded(reference: str, load=False) -> str:
    """Check if bwa index is loaded in shared memory

    If BWA index is loaded into shared memory, this function implements reference counting with a lock file to enable
    multiple instances of npsv2 to run on the same node. However, since indices loaded into shared memory are global, this
    management might conflict with other users who have loaded bwa indices into shared memory.

    Args:
        reference (str): Path to reference file
        load (bool, optional): Load index into shared memory if not present. Defaults to False.

    Returns:
        str: Shared reference name if index is loaded into shared memory, None otherwise
    """
    if load:
        return _bwa_index_load(reference)
    else:
        shared_name = os.path.basename(reference)
        return shared_name if _is_bwa_index_loaded(shared_name) else None


def _art_read_length(read_length, profile):
    """Make sure read length is compatible ART"""
    if profile in ("HS10", "HS20"):
        return min(read_length, 100)
    elif profile in ("HS25", "HSXn", "HSXt"):
        return min(read_length, 150)
    else:
        return read_length


def simulate_variant_sequencing(
    fasta_path: str,
    hap_covg: float,
    sample: Sample,
    reference: str,
    shared_reference=None,
    dir=tempfile.gettempdir(),
    stats_path: typing.Optional[str] = None,
    region: Range = None,
    phase_vcf_path: typing.Optional[str] = None,
    aligner: str = "bwa",
):
    shared_ref_arg = f"-S {quote(shared_reference)}" if shared_reference else ""
    stats_path_arg = f"-j {quote(stats_path)}" if stats_path else ""
    phase_vcf_arg = (
        f"-r {quote(str(region))} -v {quote(phase_vcf_path)} -N {sample.name}" if region and phase_vcf_path else ""
    )

    replicate_bam = tempfile.NamedTemporaryFile(delete=False, suffix=".bam", dir=dir)
    replicate_bam.close()

    synth_commandline = f"synthBAM \
        -t {quote(dir)} \
        -R {quote(reference)} \
        {shared_ref_arg} \
        {stats_path_arg} \
        -c {hap_covg:0.1f} \
        -m {sample.mean_insert_size} \
        -s {sample.std_insert_size} \
        -l {_art_read_length(sample.read_length, sample.sequencer)} \
        -p {sample.sequencer} \
        {phase_vcf_arg} \
        -i 1 \
        -a {aligner} \
        {quote(fasta_path)} \
        {quote(replicate_bam.name)}"

    synth_result = subprocess.run(synth_commandline, shell=True, stderr=subprocess.PIPE)
    if synth_result.returncode != 0 or not os.path.exists(replicate_bam.name):
        print(synth_result.stderr)
        msg = "Synthesis script failed to generate BAM"
        raise RuntimeError(msg)

    return replicate_bam.name


def augment_sample(original_sample: Sample, n, keep_original=True):
    new_samples = [original_sample] if keep_original else []

    # TODO: Apply augmentation approach like snorkel, with multiple augmentor functions. Potential augmentors
    # include coverage, insert-size distribution, GC distribution
    for _ in range(n - len(new_samples)):
        new_sample = copy.copy(original_sample)

        new_sample.mean_coverage = random.uniform(
            max(original_sample.mean_coverage * 0.5, 0), original_sample.mean_coverage + 0
        )
        # new_sample.mean_insert_size = random.uniform(original_sample.mean_insert_size - 75, original_sample.mean_insert_size + 75)
        # new_sample.std_insert_size = random.uniform(original_sample.std_insert_size - 30, original_sample.std_insert_size + 30)

        new_samples.append(new_sample)

    return new_samples
