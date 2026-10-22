import gzip
import json

import pytest

from npsv3.constants import COMPARISONS_DIR
from npsv3.graphs.sharding import GRAPH_MEMBER, HAPLOTYPE_PRIOR_MEMBER, UNIQUE_KMERS_MEMBER
from npsv3.images.haplotype_tree import (
    HaplotypeTree,
    HaplotypeTreeWriter,
    haplotype_options,
    prepare_haplotype_tree,
    source_digest,
)


def _options(cfg, sample, **kwargs):
    return haplotype_options(cfg, sample, **{"ploidy": 2, "exact_reference": False, "min_variant_size": 50, **kwargs})


def _record(true):
    return {"paths": [[1, 2], [1, 3]], "alleles": [[], [["v", 1]]], "true": true, "sequences": ["AC", "AG"]}


def test_source_digest():
    members = {GRAPH_MEMBER: b"graph", UNIQUE_KMERS_MEMBER: b"kmers"}
    digest = source_digest(members)
    assert digest == source_digest(dict(members))
    assert digest != source_digest({**members, UNIQUE_KMERS_MEMBER: b"other"}), "Unique k-mers change the sampler"
    assert digest != source_digest({**members, HAPLOTYPE_PRIOR_MEMBER: b"prior"}), "Prior changes the sampler"
    assert digest != source_digest({GRAPH_MEMBER: b"graphk", UNIQUE_KMERS_MEMBER: b"mers"}), "Members are delimited"


def test_haplotype_options(cfg, hg00733_sample):
    options = _options(cfg, hg00733_sample)
    assert options["sample"] == hg00733_sample.name
    assert options["haplotype"]["max_haplotypes"] == cfg.haplotype.max_haplotypes
    assert options["haplotype"]["strategy"]["_target_"] == cfg.haplotype.strategy._target_
    assert options == _options(cfg, hg00733_sample)

    cfg.haplotype.max_haplotypes += 1
    assert options != _options(cfg, hg00733_sample)


def test_tree_round_trip(cfg, hg00733_sample, tmp_path):
    options = _options(cfg, hg00733_sample)
    (tmp_path / "filtered_kmers.kmc_pre").touch()

    writer = HaplotypeTreeWriter(0, tmp_path)
    writer.write("HG00733_labeled", _record([0, 1]), digest="a")
    writer.write_skipped("HG00733_skipped", "too few", digest="b")
    writer.close()
    prepare_haplotype_tree(tmp_path, options, filtered_kmer_path=tmp_path / "filtered_kmers")

    tree = HaplotypeTree(tmp_path)
    tree.check(options)
    assert tree.filtered_kmer_path == str(tmp_path / "filtered_kmers"), "Filtered k-mers are recorded relative to the tree"

    assert tree.lookup("HG00733_labeled", "a") == _record([0, 1])
    assert tree.lookup("HG00733_labeled", "other") is None, "Records from a different graph aren't used"
    assert tree.lookup("HG00733_missing", "a") is None
    assert tree.lookup("HG00733_skipped", "b") == {"skipped": "too few"}
    assert tree.comparison("HG00733_labeled") is None, "There are no comparisons"
    assert tree.comparison_options() is None


def test_tree_check_reports_differences(cfg, hg00733_sample, tmp_path):
    writer = HaplotypeTreeWriter(0, tmp_path)
    writer.write("HG00733_labeled", _record([0, 1]), digest="a")
    writer.close()
    prepare_haplotype_tree(tmp_path, _options(cfg, hg00733_sample))

    tree = HaplotypeTree(tmp_path)
    assert tree.filtered_kmer_path is None
    with pytest.raises(ValueError, match="ploidy"):
        tree.check(_options(cfg, hg00733_sample, ploidy=1))
    cfg.haplotype.max_haplotypes += 1
    with pytest.raises(ValueError, match="haplotype"):
        tree.check(_options(cfg, hg00733_sample))


def test_tree_requires_provenance(tmp_path):
    with pytest.raises(ValueError, match="not a prepared haplotypes directory"):
        HaplotypeTree(tmp_path)


def test_tree_comparisons(cfg, hg00733_sample, tmp_path):
    """Comparisons made with the tree (by `add_comparisons`) are available by key, along with their options"""
    pytest.importorskip("edlib")
    pytest.importorskip("pyabpoa")
    from npsv3.images.sequence_compare import ComparisonOptions, add_comparisons, comparison_record

    labeled = {
        **_record([0, 1]),
        "reference": "ACGTACGTAC",
        "sequences": ["ACGTACGTAC", "ACGTAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACGTAC"],
        "sample_sequences": None,
    }
    writer = HaplotypeTreeWriter(0, tmp_path)
    writer.write("HG00733_labeled", labeled, digest="a")
    writer.write("HG00733_unlabeled", {**labeled, "true": [None, None]}, digest="b")
    writer.write_skipped("HG00733_skipped", "too few", digest="c")
    writer.close()
    prepare_haplotype_tree(tmp_path, _options(cfg, hg00733_sample))

    assert add_comparisons(cfg, tmp_path)
    assert (tmp_path / COMPARISONS_DIR).is_dir()
    assert not (tmp_path / "dataset.yaml").exists()

    tree = HaplotypeTree(tmp_path)
    options = ComparisonOptions.from_config(cfg.compare)
    assert tree.comparison_options() == options
    payload = tree.comparison("HG00733_labeled")
    assert payload is not None and payload[:2] == b"\x1f\x8b", "Comparisons are gzipped members, ready to copy"
    assert json.loads(gzip.decompress(payload)) == comparison_record(labeled, options)
    assert tree.comparison("HG00733_unlabeled") is None
    assert tree.comparison("HG00733_skipped") is None
