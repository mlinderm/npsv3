"""Configurable strategies for generating the candidate haplotypes for an example (selected via `cfg.graph.haplotype_strategy`)"""
import abc
import bisect
import gzip
import json
from collections.abc import Sequence

from npsv3.graphs.graph import Graph
from npsv3.graphs.haplotype import (
    HaplotypeSamplerOverlay,
    KmerClassify,
    _resolved_alleles,
    add_population_haplotypes,
    haplotype_alleles,
    prepare_genotyping_haplotypes,
)
from npsv3.types import PathType
from npsv3.util.range import Range
from npsv3.util.sample import Sample
from npsv3.util.variant import Variant


HaplotypePath = Sequence[int]
"""A haplotype as a (strictly increasing) sequence of graph node IDs"""

AlleleSet = set[tuple[str, int]]
"""The resolved analysis-variant alleles carried by a haplotype, as (variant_id, allele index) pairs"""

TrueHaplotypeIdxs = list[int | None]
"""The sample's genotype as indices into the haplotype list (empty if the "true" haplotypes are not in the graph)"""

Haplotypes = tuple[list[HaplotypePath], list[AlleleSet], TrueHaplotypeIdxs]
"""Tuple of (haplotypes, alleles per haplotype, true haplotype indices)"""


FilteredKmers = KmerClassify | PathType
"""Filtered k-mer counts as either an opened KmerClassify or the path to the filtered KMC database"""


def as_kmer_classify(filtered_kmers: FilteredKmers, sample: Sample) -> KmerClassify:
    """Return `filtered_kmers` if already a KmerClassify, otherwise open the database at that path with the sample's
    k-mer coverage. Reuse a KmerClassify across regions to avoid re-loading the database (and its prefix LUT)."""
    if isinstance(filtered_kmers, KmerClassify):
        return filtered_kmers
    return KmerClassify(str(filtered_kmers), sample.kmer_coverage)


class InsufficientHaplotypesError(RuntimeError):
    """Raised when we cannot assemble >=2 distinct haplotypes for a region, even after
    attempting to recover a population-panel individual's embedded haplotype for any still-missing
    analysis-variant ALT allele."""


def sample_path(graph: Graph, sample_name: str, haplotype: int, contig: str) -> HaplotypePath | None:
    """The sample's own haplotype path in the graph, or None if absent or not fully resolved (an inconsistent haplotype
    is broken into multiple segments during graph construction)"""
    path_name = f"{sample_name}#{haplotype}#{contig}#0"
    if not graph.has_path(path_name) or graph.has_path(f"{sample_name}#{haplotype}#{contig}#1"):
        return None
    return graph.path_nodes(path_name)


def haplotype_record(
    graph: Graph,
    region: Range,
    sample_name: str,
    haplotypes: Sequence[HaplotypePath],
    alleles: Sequence[AlleleSet],
    true_haplotype_idxs: TrueHaplotypeIdxs,
    *,
    ploidy: int = 2,
) -> dict:
    """JSON-serializable record of an example's candidate haplotypes, linking its genotype indices back to the graph.

    Genotype i is the i-th pair of `combinations_with_replacement(range(len(paths)), ploidy)`. Node IDs (`paths`,
    `sample_paths`) are only meaningful in the exact graph identified by the example's `graph_digest`. Sequences are
    included so analyses (e.g. Truvari equivalence to the sample's assembled paths) do not need to load the graph.

    Keys:
        paths: Each candidate haplotype as graph node IDs (index 0 has no analysis-variant ALT allele)
        alleles: Each candidate's resolved analysis-variant alleles, as sorted [variant_id, allele] pairs
        true: The sample's genotype as candidate indices (None entries if unknown), as used for the label
        sequences: Each candidate's sequence over the graph's region
        reference: The reference path's sequence over the graph's region
        sample_paths: The sample's own haplotype paths in the graph (None if absent or not fully resolved)
        sample_sequences: Their sequences
        graph_region: The graph's region
    """
    own_paths = [sample_path(graph, sample_name, i, region.contig) for i in range(ploidy)]
    return {
        "paths": [[int(node) for node in path] for path in haplotypes],
        "alleles": [sorted([variant_id, int(allele)] for variant_id, allele in allele_set) for allele_set in alleles],
        "true": [None if idx is None else int(idx) for idx in true_haplotype_idxs],
        "sequences": [graph.path_sequence(path) for path in haplotypes],
        "reference": graph.path_sequence(graph.path_nodes(region.contig)),
        "sample_paths": [None if path is None else [int(node) for node in path] for path in own_paths],
        "sample_sequences": [None if path is None else graph.path_sequence(path) for path in own_paths],
        "graph_region": str(graph.region),
    }


def decode_haplotype_record(payload: bytes | bytearray | str | dict) -> dict:
    """Decode a `{key}.{HAPLOTYPES_SUFFIX}` member, whether raw (gzipped JSON) or already decompressed or decoded by the
    reader (energon's default decoder gunzips `.gz` members and may parse `.json`)"""
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, (bytes, bytearray)) and bytes(payload[:2]) == b"\x1f\x8b":
        payload = gzip.decompress(payload)
    return json.loads(payload)


class HaplotypeStrategy(abc.ABC):
    """Generate the haplotypes to be genotyped for an example."""

    def __init__(self, cfg):
        self.cfg = cfg

    @abc.abstractmethod
    def __call__(
        self,
        graph: Graph,
        sampler: HaplotypeSamplerOverlay,
        region: Range,
        sample: Sample,
        filtered_kmers: FilteredKmers,
        analysis_variants: Sequence[Variant],
        *,
        ploidy: int = 2,
        exact_reference: bool = False,
    ) -> Haplotypes:
        pass


class SampledHaplotypeStrategy(HaplotypeStrategy):
    """Sample haplotypes from the graph using sample-specific k-mer counts, adding the reference and any "true"
    haplotypes present in the graph, and optionally "topping-up" with population haplotypes."""

    def __call__(
        self,
        graph: Graph,
        sampler: HaplotypeSamplerOverlay,
        region: Range,
        sample: Sample,
        filtered_kmers: FilteredKmers,
        analysis_variants: Sequence[Variant],
        *,
        ploidy: int = 2,
        exact_reference: bool = False,
    ) -> Haplotypes:
        cfg = self.cfg

        # Sample haplotypes using sample-specific k-mer counts and filtered k-mers (i.e., only those k-mers that are
        # unique to the graph and present in the sample)
        counts = as_kmer_classify(filtered_kmers, sample)
        sampler.initialize_scores(counts)
        haplotypes = sampler.sample_haplotypes(n=cfg.graph.max_haplotypes)
        assert len(haplotypes) > 0, f"No haplotypes sampled for region {region} in {sample.name}"

        # Prepare haplotypes for genotyping by ensuring the reference haplotype, and any "true" haplotypes, if present
        # in the graph, are included with the reference haplotype guaranteed to be at index 0.
        haplotypes, alleles, true_haplotype_idxs = prepare_genotyping_haplotypes(
            graph, sampler, haplotypes, analysis_variants, region.contig, sample.name, ploidy=ploidy,
            exact_reference=exact_reference,
        )
        if len(haplotypes) < 2 and cfg.graph.population_haplotype_topup:  # noqa: PLR2004
            # Only attempt "top-up" from the population if the sampler did not produce any haplotype diversity (and if
            # explicitly requested).
            excluded_samples = { sample.name, *cfg.graph.population_excluded_samples }
            haplotypes, alleles = add_population_haplotypes(
                graph,
                sampler,
                haplotypes,
                alleles,
                analysis_variants,
                region.contig,
                excluded_samples,
                cfg.graph.max_haplotypes,
                ploidy=ploidy,
            )
        return haplotypes, alleles, true_haplotype_idxs


class TrueHaplotypeStrategy(HaplotypeStrategy):
    """Generate haplotypes from the sample's "true" haplotypes (e.g., for training with high-quality assemblies).

    The haplotypes always include the sample's true haplotypes (and thus the true genotype) and a haplotype with no
    analysis-variant (SV) ALT alleles (the "reference", at index 0). The remaining haplotypes are the highest
    k-mer-scoring combinations of analysis-variant alleles applied to the true haplotypes (i.e., the true haplotypes with
    any SVs added, removed or swapped), up to `cfg.graph.max_haplotypes`. Thus all haplotypes differ from the true
    haplotypes only by the SVs of interest.

    Regions where any of the sample's true haplotypes are not fully resolved raise InsufficientHaplotypesError.
    """

    def __call__(
        self,
        graph: Graph,
        sampler: HaplotypeSamplerOverlay,
        region: Range,
        sample: Sample,
        filtered_kmers: FilteredKmers,
        analysis_variants: Sequence[Variant],
        *,
        ploidy: int = 2,
        exact_reference: bool = False,
    ) -> Haplotypes:
        # Initialize sampler scores with the sample-specific k-mer counts to rank the candidate haplotypes
        counts = as_kmer_classify(filtered_kmers, sample)
        sampler.initialize_scores(counts)

        return true_haplotype_neighbors(
            graph,
            sampler,
            analysis_variants,
            region.contig,
            sample.name,
            max_haplotypes=self.cfg.graph.max_haplotypes,
            ploidy=ploidy,
            exact_reference=exact_reference,
        )


def _variant_allele_paths(graph: Graph, variant: Variant) -> dict[int, list[int]]:
    """Return map of allele index -> allele path nodes for the alleles of `variant` with paths in `graph` ('*' alleles
    don't have paths). Empty if the variant was excluded from graph construction."""
    variant_id = variant.variant_id
    allele_paths = {}
    for allele in range(variant.num_alleles):
        path_name = f"_alt_{variant_id}_{allele}"
        if graph.has_path(path_name):
            allele_paths[allele] = graph.path_nodes(path_name)
    return allele_paths if 0 in allele_paths else {}


def set_variant_allele(
    graph: Graph, haplotype: Sequence[int], allele_paths: dict[int, list[int]], allele: int
) -> list[int] | None:
    """Return a copy of `haplotype` with the variant's current allele replaced by `allele`, or None if not possible.

    Each variant's allele paths form a "bubble" with the same predecessor and successor reference nodes, and node IDs
    are in topological order (so haplotypes are strictly increasing). We replace the portion of the haplotype within
    the bubble with the path for `allele`, joining at the last haplotype node with an edge into the allele path and the
    first subsequent haplotype node with an edge out of the allele path, so the result is always a valid walk through
    the graph. Any other (non-analysis) variants contained within the variant's reference span are removed, while
    variants outside that span are unchanged. Replacement fails, e.g., when the haplotype includes an overlapping
    variant that is not compatible with `allele`.

    Args:
        graph (Graph): Graph for the region
        haplotype (Sequence[int]): Haplotype as a sequence of (strictly increasing) node IDs
        allele_paths (dict[int, list[int]]): Map of allele index -> allele path nodes for the variant
        allele (int): Allele index to set

    Returns:
        list[int] | None: The updated haplotype, or None if `allele` can't be placed on `haplotype`
    """
    nodes = allele_paths[allele]
    prefix_last = next(
        (i for i in range(bisect.bisect_left(haplotype, nodes[0]) - 1, -1, -1) if graph.has_edge(haplotype[i], nodes[0])),
        None,
    )
    if prefix_last is None:
        return None
    suffix_first = next(
        (
            j for j in range(max(prefix_last + 1, bisect.bisect_right(haplotype, nodes[-1])), len(haplotype))
            if graph.has_edge(nodes[-1], haplotype[j])
        ),
        None,
    )
    if suffix_first is None:
        return None
    return [*haplotype[:prefix_last + 1], *nodes, *haplotype[suffix_first:]]


def _apply_edits(
    graph: Graph,
    haplotype: Sequence[int],
    edits: Sequence[tuple[dict[int, list[int]], int]],
) -> list[int] | None:
    """Apply (allele_paths, allele) edits in order with `set_variant_allele`, returning None if any edit fails"""
    result = list(haplotype)
    for allele_paths, allele in edits:
        result = set_variant_allele(graph, result, allele_paths, allele)
        if result is None:
            return None
    return result


def true_haplotype_neighbors(
    graph: Graph,
    sampler: HaplotypeSamplerOverlay,
    analysis_variants: Sequence[Variant],
    contig: str,
    sample_name: str,
    *,
    max_haplotypes: int,
    ploidy: int = 2,
    exact_reference: bool = False,
) -> Haplotypes:
    """Generate haplotypes from the sample's "true" haplotypes and their analysis-variant "neighbors".

    Candidates are found with the sampler's dynamic programming search restricted to each
    true haplotype ("backbone") plus the alleles of the analysis variants, so the highest-scoring combinations of SV
    changes are identified without enumerating every combination.

    Args:
        graph (Graph): Graph for the region, containing the sample's "true" haplotype paths
        sampler (HaplotypeSamplerOverlay): Initialized haplotype sampler, used for scoring and decoding haplotypes
        analysis_variants (Sequence[Variant]): Variants of interest (e.g., SVs)
        contig (str): The contig for the reference path
        sample_name (str): Sample name used to look up "true" haplotype paths in the graph
        max_haplotypes (int): Maximum number of haplotypes, including the reference and true haplotypes (which are
            always included).
        ploidy (int, optional): Number of "true" haplotype paths. Defaults to 2.
        exact_reference (bool, optional): If True, the haplotype at index 0 is exactly the graph's `contig` path.
            Otherwise, the reference haplotype is derived from the true haplotypes by removing all analysis-variant
            ALT alleles (so it differs from the true haplotypes only by the SVs), falling back to the `contig` path
            if that is not possible. Defaults to False.

    Raises:
        InsufficientHaplotypesError: If any of the sample's true haplotypes are not fully resolved in the graph.

    Returns:
        Haplotypes: Tuple of haplotypes, alleles and true haplotype indices
    """
    assert len(analysis_variants) > 0, "No analysis variants provided to true_haplotype_neighbors"
    
    # Obtain the true haplotypes, requiring that each is a single fully-resolved path (an inconsistent haplotype is
    # broken into multiple segments during graph construction).
    true_paths = []
    for i in range(ploidy):
        if (path := sample_path(graph, sample_name, i, contig)) is None:
            msg = f"True haplotype {sample_name}#{i}#{contig}#0 is not fully resolved in the graph"
            raise InsufficientHaplotypesError(msg)
        true_paths.append(path)
    true_alleles = [_resolved_alleles(a) for a in haplotype_alleles(sampler, true_paths, analysis_variants)]

    variant_alleles = [
        (variant.variant_id, allele_paths)
        for variant in analysis_variants
        if (allele_paths := _variant_allele_paths(graph, variant))
    ]

    # Construct the reference haplotype with no analysis-variant ALT alleles, preferring a true haplotype or, if not
    # exact_reference, a true haplotype with all SVs removed (to match the sample's "background").
    ref_alleles = {(variant.variant_id, 0) for variant in analysis_variants}
    ref_path = None
    if not exact_reference:
        ref_path = next((path for path, alleles in zip(true_paths, true_alleles, strict=True) if alleles == ref_alleles), None)
        for path, alleles in zip(true_paths, true_alleles, strict=True):
            if ref_path is not None:
                break
            edits = [(allele_paths, 0) for variant_id, allele_paths in variant_alleles if (variant_id, 0) not in alleles]
            candidate = _apply_edits(graph, path, edits)
            if candidate is not None and _resolved_alleles(haplotype_alleles(sampler, [candidate], analysis_variants)[0]) == ref_alleles:
                ref_path = candidate
    if ref_path is None:
        ref_path = graph.path_nodes(contig)

    haplotypes: list[HaplotypePath] = [ref_path]
    alleles: list[AlleleSet] = [ref_alleles]

    # Add the true haplotypes (identical haplotypes, e.g., for homozygous genotypes, are only added once)
    true_haplotype_idxs: TrueHaplotypeIdxs = []
    for path, path_alleles in zip(true_paths, true_alleles, strict=True):
        idx = next((i for i, existing in enumerate(alleles) if existing == path_alleles), None)
        if idx is None:
            haplotypes.append(path)
            alleles.append(path_alleles)
            idx = len(haplotypes) - 1
        true_haplotype_idxs.append(idx)

    # Find the highest-scoring haplotypes that differ from each distinct true haplotype only at the analysis variants.
    # Request enough paths that max_haplotypes candidates remain even if all the existing haplotypes are re-discovered.
    candidates: list[tuple[float, HaplotypePath, AlleleSet]] = []
    for path in {tuple(p): p for p in true_paths}.values():
        for candidate in sampler.find_best_backbone_paths(path, n=max_haplotypes + len(haplotypes)):
            candidate_alleles = _resolved_alleles(haplotype_alleles(sampler, [candidate], analysis_variants)[0])
            if any(candidate_alleles == existing for existing in alleles):
                continue
            candidates.append((sampler.score(candidate), candidate, candidate_alleles))

    # Add candidates in descending score order (stable, so ties are resolved in generation order)
    for _score, candidate, candidate_alleles in sorted(candidates, key=lambda c: c[0], reverse=True):
        if len(haplotypes) >= max_haplotypes:
            break
        if any(candidate_alleles == existing for existing in alleles):
            continue  # A higher-scoring candidate derived from a different true haplotype has the same alleles
        haplotypes.append(candidate)
        alleles.append(candidate_alleles)

    return haplotypes, alleles, true_haplotype_idxs
