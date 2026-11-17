"""Configurable strategies for generating the candidate haplotypes for an example (selected via `cfg.haplotype.strategy`)"""
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
    resolve_genotype_alleles,
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
"""The sample's genotype as indices into the haplotype list (empty if the "true" haplotypes are not in the graph, e.g., the
sample isn't in the VCF, which `generate_haplotypes` normalizes to all None entries)"""

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
    analysis-variant ALT allele.

    Attributes:
        haplotypes: The `haplotype_record` of the single haplotype that was generated (i.e., one with only reference
            alleles for the analysis variants), if any, e.g., so genotyping can report the region as homozygous for it
    """

    def __init__(self, *args, haplotypes: dict | None = None):
        super().__init__(*args)
        self.haplotypes = haplotypes


def sample_path(graph: Graph, sample_name: str, haplotype: int, contig: str) -> HaplotypePath | None:
    """The sample's own haplotype path in the graph, or None if absent or not fully resolved (an inconsistent haplotype
    is broken into multiple segments during graph construction)"""
    path_name = f"{sample_name}#{haplotype}#{contig}#0"
    if not graph.has_path(path_name) or graph.has_path(f"{sample_name}#{haplotype}#{contig}#1"):
        return None
    return graph.path_nodes(path_name)


def _variant_allele_nodes(graph: Graph, variant: Variant) -> dict[int, frozenset[int]] | None:
    """Map of allele index -> the nodes that distinguish that allele's path from the variant's other alleles (only the
    alleles with paths, i.e., not '*'), or None if the variant was excluded from graph construction.

    This mirrors the graph's own bookkeeping of the variant paths each node identifies (`Graph::node_variant_paths_`):
    an ALT allele is identified by its ALT node, i.e., its path's nodes not in the REF path (deletions have a zero-length
    ALT node), and the REF allele by the reference nodes any of the ALT alleles replace (for an insertion, a zero-length
    reference node).
    """
    allele_paths = _variant_allele_paths(graph, variant)
    if not allele_paths:
        return None
    ref_nodes = frozenset(allele_paths[0])
    allele_nodes = {0: frozenset()}
    for allele, path in allele_paths.items():
        if allele == 0:
            continue
        alt_nodes = frozenset(path)
        allele_nodes[allele] = alt_nodes - ref_nodes
        allele_nodes[0] |= ref_nodes - alt_nodes
    return allele_nodes


def decode_variant_alleles(
    graph: Graph, haplotypes: Sequence[HaplotypePath], variants: Sequence[Variant]
) -> list[list[int | None]]:
    """The allele of each of `variants` (of any size, not only the analysis variants) carried by each haplotype.

    A haplotype carries an ALT allele if it passes through that allele's ALT node, and otherwise the REF allele if it
    passes through the reference nodes the variant's ALT alleles replace. A haplotype that does neither, e.g., one with an
    overlapping deletion spanning the variant, carries the variant's first '*' allele, if it has one, or otherwise the
    REF allele (as for the sampler's own decoding, `Graph::ApplyReferenceFallback`).

    Args:
        graph: Graph for the region
        haplotypes: Haplotypes as sequences of graph node IDs
        variants: The variants, e.g., every VCF record overlapping the graph's region

    Returns:
        list[list[int | None]]: For each haplotype, the allele index for each variant (in the order of `variants`), or
        None for the variants that were excluded from graph construction (e.g., those not strictly inside the region)
    """
    decoders = []
    for variant in variants:
        allele_nodes = _variant_allele_nodes(graph, variant)
        star = next((a for a in range(1, variant.num_alleles) if variant.allele_length_change(a) is None), None)
        decoders.append((allele_nodes, 0 if star is None else star))

    result = []
    for haplotype in haplotypes:
        nodes = frozenset(int(node) for node in haplotype)
        alleles: list[int | None] = []
        for allele_nodes, fallback in decoders:
            if allele_nodes is None:
                alleles.append(None)
                continue
            allele = next((a for a, distinct in allele_nodes.items() if a != 0 and not distinct.isdisjoint(nodes)), None)
            if allele is None:
                allele = 0 if not allele_nodes[0].isdisjoint(nodes) else fallback
            alleles.append(allele)
        result.append(alleles)
    return result


def haplotype_record(
    graph: Graph,
    region: Range,
    sample_name: str,
    haplotypes: Sequence[HaplotypePath],
    alleles: Sequence[AlleleSet],
    true_haplotype_idxs: TrueHaplotypeIdxs,
    *,
    ploidy: int = 2,
    variants: Sequence[Variant] | None = None,
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
        variant_ids: The IDs of `variants` (only if `variants` is provided)
        variant_alleles: Each candidate's allele for each of `variants`, None if the variant isn't in the graph (only if
            `variants` is provided, see `decode_variant_alleles`)
    """
    own_paths = [sample_path(graph, sample_name, i, region.contig) for i in range(ploidy)]
    record = {
        "paths": [[int(node) for node in path] for path in haplotypes],
        "alleles": [sorted([variant_id, int(allele)] for variant_id, allele in allele_set) for allele_set in alleles],
        "true": [None if idx is None else int(idx) for idx in true_haplotype_idxs],
        "sequences": [graph.path_sequence(path) for path in haplotypes],
        "reference": graph.path_sequence(graph.path_nodes(region.contig)),
        "sample_paths": [None if path is None else [int(node) for node in path] for path in own_paths],
        "sample_sequences": [None if path is None else graph.path_sequence(path) for path in own_paths],
        "graph_region": str(graph.region),
    }
    if variants is not None:
        record["variant_ids"] = [variant.variant_id for variant in variants]
        record["variant_alleles"] = decode_variant_alleles(graph, haplotypes, variants)
    return record


def generate_haplotypes(
    cfg,
    graph: Graph,
    sampler: HaplotypeSamplerOverlay,
    region: Range,
    sample: Sample,
    sample_idx: int | None,
    filtered_kmers: FilteredKmers,
    analysis_variants: Sequence[Variant],
    *,
    strategy: "HaplotypeStrategy",
    ploidy: int = 2,
    exact_reference: bool = False,
    variants: Sequence[Variant] | None = None,
) -> dict:
    """Generate an example's candidate haplotypes with `strategy` and return them as a `haplotype_record`.

    The record's `true` entries are None if any analysis variant's genotype for the sample is missing or filtered (see
    below), or if the sample isn't in the VCF (`sample_idx` is None, e.g., when genotyping), in which case the example
    is unlabeled. This is the only haplotype generation used by image generation, whether run inline or ahead of time
    (`command=haplotypes`), so the two can't disagree.

    Args:
        variants: If provided, record each candidate's alleles for all of these variants (e.g., every VCF record in the
            region, not just the analysis variants) in the record's `variant_alleles` (see `haplotype_record`)

    Raises:
        InsufficientHaplotypesError: If fewer than 2 haplotypes are generated (with the record for the single haplotype,
            if one was generated)
    """
    haplotypes, alleles, true_haplotype_idxs = strategy(
        graph, sampler, region, sample, filtered_kmers, analysis_variants, ploidy=ploidy, exact_reference=exact_reference,
    )
    if not true_haplotype_idxs:
        # None of the sample's own haplotype paths are in the graph (e.g., the sample isn't in the VCF), so it's unlabeled
        true_haplotype_idxs = [None] * ploidy
    if len(haplotypes) < 2:  # noqa: PLR2004
        msg = f"Fewer than 2 haplotypes for region {region} in {sample.name} (have {len(haplotypes)})"
        record = None
        if haplotypes:
            record = haplotype_record(
                graph, region, sample.name, haplotypes, alleles, [None] * ploidy, ploidy=ploidy, variants=variants,
            )
        raise InsufficientHaplotypesError(msg, haplotypes=record)
    assert len(true_haplotype_idxs) == ploidy, f"Expected {ploidy} true haplotypes, got {len(true_haplotype_idxs)}"

    # Graph construction embeds a sample's "true" haplotype path as reference nodes wherever a variant's genotype
    # for that sample was missing, making a missing genotype indistinguishable from an explicit homozygous-REF call.
    # Update the true_haplotype_idxs to None for any filtered or missing genotypes, unless cfg.haplotype.missing_are_ref
    # is specified to translate fully missing genotypes to homozygous REF (relevant for merged high-quality assemblies
    # where missing variants are assumed to be identical to the reference rather than "not genotyped"). Filtered
    # genotypes are treated as fully missing if cfg.graph.filtered_as_missing (matching graph construction).
    if analysis_variants and sample_idx is not None and None not in true_haplotype_idxs and any(
        resolve_genotype_alleles(
            variant.genotype(sample_idx),
            ploidy=ploidy,
            missing_are_ref=cfg.haplotype.missing_are_ref,
            filtered_as_missing=cfg.graph.filtered_as_missing,
        ) is None
        for variant in analysis_variants
    ):
        true_haplotype_idxs = [None] * ploidy

    return haplotype_record(
        graph, region, sample.name, haplotypes, alleles, true_haplotype_idxs, ploidy=ploidy, variants=variants,
    )


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
    haplotypes present in the graph (unless `haplotype.add_true_haplotypes` is false), and optionally "topping-up" with
    population haplotypes."""

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
        haplotypes = sampler.sample_haplotypes(n=cfg.haplotype.max_haplotypes)
        assert len(haplotypes) > 0, f"No haplotypes sampled for region {region} in {sample.name}"

        # Prepare haplotypes for genotyping by ensuring the reference haplotype, and any "true" haplotypes, if present
        # in the graph, are included with the reference haplotype guaranteed to be at index 0.
        haplotypes, alleles, true_haplotype_idxs = prepare_genotyping_haplotypes(
            graph, sampler, haplotypes, analysis_variants, region.contig, sample.name, ploidy=ploidy,
            exact_reference=exact_reference, add_true_haplotypes=cfg.haplotype.get("add_true_haplotypes", True),
        )
        if len(haplotypes) < 2 and cfg.haplotype.population_haplotype_topup:  # noqa: PLR2004
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
                cfg.haplotype.max_haplotypes,
                ploidy=ploidy,
            )
        return haplotypes, alleles, true_haplotype_idxs


class TrueHaplotypeStrategy(HaplotypeStrategy):
    """Generate haplotypes from the sample's "true" haplotypes (e.g., for training with high-quality assemblies).

    The haplotypes always include the sample's true haplotypes (and thus the true genotype) and a haplotype with no
    analysis-variant (SV) ALT alleles (the "reference", at index 0). The remaining haplotypes are the highest
    k-mer-scoring combinations of analysis-variant alleles applied to the true haplotypes (i.e., the true haplotypes with
    any SVs added, removed or swapped), up to `cfg.haplotype.max_haplotypes`. Thus all haplotypes differ from the true
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
            max_haplotypes=self.cfg.haplotype.max_haplotypes,
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
