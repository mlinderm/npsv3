import numpy as np
import pytest

from npsv3.images.example import EnergonExampleWriter, energon_key
from npsv3.images.haplotype_tree import HaplotypeTreeWriter, prepare_haplotype_tree
from npsv3.images.subsets import (
    GenotypeSubset,
    HomRefDownsample,
    RegionCounts,
    expand_sources,
    genotype_subset_from_config,
    genotypes,
    is_hom_ref_label,
    num_haplotypes_from_genotypes,
    read_region_stats,
    region_key,
    region_stats,
    write_region_stats,
)
from npsv3.models.energon import prepare_energon_dataset
from npsv3.util.range import Range

# For 4 haplotypes: (0,0) (0,1) (0,2) (0,3) (1,1) (1,2) (1,3) (2,2) (2,3) (3,3)


@pytest.mark.parametrize(
    ("true", "include_homozygous", "expected"),
    [
        ((1, 2), False, [0, 1, 2, 4, 5, 6, 7, 8]),
        ((1, 2), True, [0, 1, 2, 4, 5, 6, 7, 8, 9]),
        ((1, 1), False, [0, 1, 4, 5, 6]),  # Hom. alt. labels share a haplotype with few genotypes...
        ((1, 1), True, [0, 1, 4, 5, 6, 7, 9]),  # ...so add the other homozygous genotypes
        ((0, 0), False, [0, 1, 2, 3]),
    ],
)
def test_genotype_subset(true, include_homozygous, expected):
    assert GenotypeSubset(include_homozygous=include_homozygous).genotype_idxs(4, true) == expected


def test_genotype_subset_select():
    subset = GenotypeSubset(include_homozygous=False)
    # An example with every genotype, labeled (1, 1)
    assert subset.select({"g": 10, "label": 4}) == [0, 1, 4, 5, 6]
    # An example with the "homozygous" subset, labeled (1, 1) (at position 2)
    stored = GenotypeSubset().genotype_idxs(4, (1, 1))
    assert subset.select({"g": len(stored), "label": 2, "genotype_idxs": stored, "num_haplotypes": 4}) == [0, 1, 2, 3, 4]
    # Unlabeled examples are unchanged
    assert subset.select({"g": 10, "label": None}) is None


def test_genotype_subset_from_config():
    assert genotype_subset_from_config(None) is None
    assert genotype_subset_from_config({"enabled": False, "include_homozygous": True}) is None
    assert genotype_subset_from_config({"enabled": True, "include_homozygous": False}) == GenotypeSubset(False)
    assert genotype_subset_from_config({"include_homozygous": False}) == GenotypeSubset(False)


def test_genotype_counts():
    for n in range(1, 9):
        assert num_haplotypes_from_genotypes(len(genotypes(n))) == n
    with pytest.raises(AssertionError):
        num_haplotypes_from_genotypes(7)


def test_is_hom_ref_label():
    assert is_hom_ref_label({"label": 0})
    assert not is_hom_ref_label({"label": 1})
    assert not is_hom_ref_label({"label": None})
    assert is_hom_ref_label({"label": 0, "genotype_idxs": [0, 2, 5]})
    assert not is_hom_ref_label({"label": 1, "genotype_idxs": [0, 2, 5]})


def test_hom_ref_downsample():
    samples = [f"S{i}" for i in range(1000)]
    counts = RegionCounts(carriers=10, hom_ref=len(samples), labeled=len(samples) + 10)

    def kept(downsample: HomRefDownsample) -> set[str]:
        return {sample for sample in samples if downsample.keep("chr1_1_100", sample, counts)}

    loose, tight = kept(HomRefDownsample(ratio=5, min_keep=5)), kept(HomRefDownsample(ratio=2, min_keep=5))
    assert 8 <= len(tight) <= 35  # noqa: PLR2004, about 2 x 10 carriers
    assert 15 <= len(loose) - len(tight) <= 45  # noqa: PLR2004, about 3 x 10 more
    assert tight <= loose, "Smaller ratios keep a subset of the examples"
    assert kept(HomRefDownsample(ratio=2, min_keep=5)) == tight, "Decisions are deterministic"
    assert kept(HomRefDownsample(ratio=2, min_keep=5, seed=1)) != tight

    # Keep all of the hom. ref. examples when there aren't more than would be kept
    assert kept(HomRefDownsample(ratio=100, min_keep=5)) == set(samples)
    # Keep at least `min_keep` hom. ref. examples (on average) in regions without carriers
    no_carriers = RegionCounts(carriers=0, hom_ref=len(samples), labeled=len(samples))
    assert 0 < sum(HomRefDownsample(min_keep=50).keep("chr1_1_100", sample, no_carriers) for sample in samples) < 100  # noqa: PLR2004
    assert not any(HomRefDownsample(min_keep=0).keep("chr1_1_100", sample, no_carriers) for sample in samples)


REGIONS = [Range(f"chr1:{i * 1000 + 1}-{i * 1000 + 500}") for i in range(3)]


def _write_haplotypes(directory, sample: str, trues: list) -> None:
    writer = HaplotypeTreeWriter(0, directory)
    for region, true in zip(REGIONS, trues, strict=True):
        key = energon_key(sample, region)
        if true is None:
            writer.write_skipped(key, "too few haplotypes", digest="d")
        else:
            writer.write(key, {"paths": [[1], [2], [3]], "true": true}, digest="d")
    writer.close()
    prepare_haplotype_tree(directory, {"sample": sample})


def _write_dataset(directory, sample: str, labels: list, genotype_idxs: list | None = None) -> str:
    writer = EnergonExampleWriter(0, str(directory))
    for i, (region, label) in enumerate(zip(REGIONS, labels, strict=True)):
        idxs = list(range(6)) if genotype_idxs is None or genotype_idxs[i] is None else genotype_idxs[i]
        writer.write(
            {
                "region": str(region),
                "image": np.zeros((2, 3, 1), dtype=np.uint8),
                "label": label,
                "label.rank": None,
                "num_genotypes": len(idxs),
                "genotype_idxs": idxs,
                "num_haplotypes": 3,
            },
            region=region,
            sample_name=sample,
        )
    writer.close()
    return prepare_energon_dataset(str(directory))


def test_region_stats(tmp_path):
    # S0 and S1 from their haplotypes, S2 (and S1 again) from image datasets
    _write_haplotypes(tmp_path / "S0", "S0", [[0, 0], [0, 1], [None, 0]])
    _write_haplotypes(tmp_path / "S1", "S1", [[0, 0], [0, 0], None])
    _write_dataset(tmp_path / "S1_images", "S1", [0, 0, None])
    # S2 has a subset of genotypes in the second region, where its label (at position 1) is (0, 1)
    _write_dataset(tmp_path / "S2_images", "S2", [1, 1, 0], genotype_idxs=[None, [0, 1, 3], None])

    stats = region_stats([str(tmp_path / "S0"), str(tmp_path / "S1"), str(tmp_path / "S1_images"), str(tmp_path / "S2_images")])
    keys = [energon_key("S", region)[2:] for region in REGIONS]
    assert stats == {
        keys[0]: RegionCounts(carriers=1, hom_ref=2, labeled=3),
        keys[1]: RegionCounts(carriers=2, hom_ref=1, labeled=3),
        keys[2]: RegionCounts(carriers=0, hom_ref=1, labeled=1),
    }

    path = tmp_path / "region_stats.tsv"
    write_region_stats(stats, path)
    assert read_region_stats(path) == stats

    with pytest.raises(ValueError, match="neither"):
        region_stats([str(tmp_path / "missing")])


def test_expand_sources(tmp_path, caplog):
    _write_haplotypes(tmp_path / "S0", "S0", [[0, 0], [0, 1], [0, 0]])
    _write_dataset(tmp_path / "S1", "S1", [0, 0, 0])
    # Hydra's output beside the per-sample directories, e.g., if it was the sweep directory
    (tmp_path / "main.log").write_text("[2026-10-27 10:08:21,494][HYDRA] ...\n")
    (tmp_path / ".hydra").mkdir()
    (tmp_path / "empty").mkdir()

    with caplog.at_level("WARNING"):
        assert expand_sources(str(tmp_path / "*")) == [str(tmp_path / "S0"), str(tmp_path / "S1")]
    assert "Skipping 2 paths" in caplog.text  # main.log and empty (glob skips .hydra)
    assert expand_sources(str(tmp_path / "*" / "dataset.yaml")) == [str(tmp_path / "S1" / "dataset.yaml")]
    # Paths without wildcards are used as given (and fail later if they aren't sources)
    assert expand_sources([str(tmp_path / "main.log")]) == [str(tmp_path / "main.log")]
    with pytest.raises(ValueError, match="No inputs match"):
        expand_sources(str(tmp_path / "missing*"))


def test_region_key():
    region = Range("GL000207.1:1-100")
    assert region_key(energon_key("HG00096", region), "HG00096") == energon_key("HG00733", region)[len("HG00733_") :]
