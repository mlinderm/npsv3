#include "graph.hpp"

#include <algorithms/topological_sort.hpp>
#include <algorithms/kmer.hpp>
#include <fmt/format.h>
#include <boost/core/span.hpp>
#include <boost/dynamic_bitset.hpp>

#include "fasta.hpp"
#include "variant.hpp"

/*
Construct a pangenome graph from a VCF in a specific region, with paths for each haplotype.

The resulting graph contains dedicated nodes with '*' sequence for zero-length alleles, e.g.,
the alternate alleles for a DEL variant and the reference allele for an INS variant. These nodes facilitate
all paths that traverse a set of variant alleles.
*/


namespace npsv3 {

namespace {
template <typename T>
void SortAndUniquify(std::vector<T>& vec) {
  std::sort(vec.begin(), vec.end());
  auto last = std::unique(vec.begin(), vec.end());
  vec.erase(last, vec.end());
}

/**
 * @brief Return true if the two path sets entirely belong to a single variant
 * 
 * @param paths1, paths2 Paths to compare
 * @param variant_path_starts [start, end] path indices for each variant
 * @return true if the two path sets entirely belong to a single variant
 */
bool PathsInSingleVariant(const Graph::PathIdSet& paths1, const Graph::PathIdSet& paths2,
                          const std::vector<size_t>& variant_path_starts) {
  auto combined_paths = paths1 | paths2;
  auto path_id = combined_paths.find_first();

  auto last = std::upper_bound(variant_path_starts.begin(), variant_path_starts.end(), path_id);
  assert(last != variant_path_starts.begin());
  auto first = std::prev(last);

  for (auto next_path_id = combined_paths.find_next(path_id); next_path_id != Graph::PathIdSet::npos;
       next_path_id = combined_paths.find_next(next_path_id)) {
    if (next_path_id >= *last) {
      return false;  // Found a path outside the current variant bucket
    }
  }

  return true;
}

/**
 * @brief Helper class for construct graph from VCF by identifying reference regions
 * 
 */
class ReferenceNodes {
 public:
  typedef std::vector<handlegraph::handle_t> HandleSeq;
  typedef std::pair<HandleSeq::const_iterator, HandleSeq::const_iterator> HandleRange;

  ReferenceNodes(odgi::graph_t& graph, const std::string& reference_fasta_path, const Range& region)
      : graph_(graph), region_(region) {
    FastaReader fasta_reader(reference_fasta_path);
    ref_seq_ = fasta_reader.FetchSequence(region_);
  }

  void CreateRefNode(Pos start, Pos end) {
    if (!regions_.empty() && start < regions_.back().end()) {
      throw std::runtime_error("Reference node regions must be in sorted order");
    }
    regions_.emplace_back(region_.contig(), start, end);
    auto next_handle =
        graph_.create_handle(start == end ? "*" : ref_seq_.substr(start - region_.start(), end - start));
    if (!ref_handles_.empty()) {
      graph_.create_edge(ref_handles_.back(), next_handle);
    }
    ref_handles_.push_back(next_handle);
  }

  handlegraph::path_handle_t AddRefPath() {
    odgi::path_handle_t path_handle = graph_.create_path_handle(region_.contig());
    for (const auto& handle : ref_handles_) {
      graph_.append_step(path_handle, handle);
    }
    return path_handle;
  }

  handlegraph::path_handle_t AddVariantPath(const Variant::VariantId& variant_id, int allele,
                                            const HandleRange& ref_handles, const std::string& path_prefix = "alt") {
    handlegraph::path_handle_t path_handle = graph_.create_path_handle(fmt::format("_{}_{}_{}", path_prefix, to_string(variant_id), allele));
    for (auto it = ref_handles.first; it != ref_handles.second; ++it) {
      graph_.append_step(path_handle, *it);
    }
    return path_handle;
  }

  handlegraph::path_handle_t AddVariantPath(const Variant::VariantId& variant_id, int allele,
                                            const HandleRange& prefix_handles,
                                            const handlegraph::handle_t& alt_handle,
                                            const HandleRange& suffix_handles,
                                            const std::string& path_prefix = "alt") {
    auto path_handle = AddVariantPath(variant_id, allele, prefix_handles, path_prefix);
    graph_.append_step(path_handle, alt_handle);
    for (auto it = suffix_handles.first; it != suffix_handles.second; ++it) {
      graph_.append_step(path_handle, *it);
    }
    return path_handle;
  }

  HandleRange FindContainedHandles(const Range& region) const {
    if (region.length() == 0) {
      // There should be one exactly matching zero-length region
      auto it = std::lower_bound(std::begin(regions_), std::end(regions_), region.start());
      if (it == std::end(regions_) || it->start() != region.start() || it->end() != region.end()) {
        throw std::runtime_error("No matching zero-length reference region found");
      }
      auto idx = std::distance(std::begin(regions_), it);
      return std::make_pair(std::begin(ref_handles_) + idx, std::begin(ref_handles_) + idx + 1);
    } else {
      auto it = std::lower_bound(std::begin(regions_), std::end(regions_), region.start());
      if (it == std::end(regions_) || it->start() != region.start()) {
        throw std::runtime_error("No matching reference region start found");
      }
      // Since region has non-zero length, it should come "after" any leading zero length regions
      for (; it->length() == 0; ++it);
      // We assume most regions will be small and so perform linear search for the end
      for (auto end_it = it; end_it != std::end(regions_); ++end_it) {
        if (end_it->end() == region.end()) {  // There must be a region with a matching end
          auto start_idx = std::distance(std::begin(regions_), it);
          auto end_idx = std::distance(std::begin(regions_), end_it) + 1;
          return std::make_pair(std::begin(ref_handles_) + start_idx, std::begin(ref_handles_) + end_idx);
        }
      }
      // No matching end found
      throw std::runtime_error("No matching sequence of reference regions found");
    }
  }

  handlegraph::handle_t GetOrCreateAltHandle(const std::string_view& alt_sequence, const HandleRange& ref_handles) {
    auto node_sequence = alt_sequence.size() == 0 ? std::string_view("*") : alt_sequence;

    // Have we already created this ALT allele node before?
    auto [begin, end] = alt_handles_.equal_range(ref_handles);
    for (auto it = begin; it != end; ++it) {
      auto & handle = it->second;
      if (graph_.get_sequence(handle) == node_sequence) {
        return handle;
      }
    }
    
    auto handle = graph_.create_handle(std::string(node_sequence));  // TODO: Do we have to create a string here?
    // Link to the reference nodes before and after
    if (ref_handles.first != std::begin(ref_handles_)) {
      graph_.create_edge(*std::prev(ref_handles.first), handle);
    }
    if (ref_handles.second != std::end(ref_handles_)) {
      graph_.create_edge(handle, *(ref_handles.second));
    }

    alt_handles_.emplace(ref_handles, handle);
    alt_handle_links_.emplace_back(handle, ref_handles.second);
    return handle;
  }

  void LinkAltHandles(const std::vector<Graph::PathIdSet>& node_variant_paths,
                      const std::vector<size_t>& variant_path_starts, bool enforce_multiallelic = true) {
    for (const auto& [alt_handle, next_ref_handle_it] : alt_handle_links_) {
      auto it = alt_handles_.lower_bound(std::make_pair(next_ref_handle_it, next_ref_handle_it));
      for (; it != alt_handles_.end(); ++it) {
        const auto& [ref_handles, target_alt_handle] = *it;
        if (ref_handles.first != next_ref_handle_it) {
          break;
        }
        // Link ALT handle to the next ALT handle. If specified, don't create links between alternate alleles of the
        // same multi-allelic variant since that is not consistent with the VCF, i.e., the alternate alleles must be
        // associated with another variant.

        // Should not create self-cycles since "next_ref_handle" is after the last reference handle of the current ALT
        // allele
        assert(target_alt_handle != alt_handle);

        if (enforce_multiallelic &&
            PathsInSingleVariant(node_variant_paths[graph_.get_id(alt_handle)],
                                 node_variant_paths[graph_.get_id(target_alt_handle)], variant_path_starts)) {
          continue;  // Same variant, skip linking
        }

        graph_.create_edge(alt_handle, target_alt_handle);
      }
    }
  }

  odgi::graph_t& graph_;
  const Range region_;
  FastaSequence ref_seq_;

  // Regions and corresponding graph handles for reference alleles in sorted order of position
  std::vector<Range> regions_;
  HandleSeq ref_handles_;

  std::multimap<HandleRange, handlegraph::handle_t> alt_handles_;  // ALT allele handles keyed by the corresponding reference handles
  std::vector<std::pair<handlegraph::handle_t, HandleRange::first_type>> alt_handle_links_; // Outgoing link from ALT alleles to reference handles 
};

}  // namespace

Graph::Graph(const std::string& reference_fasta_path, const std::string& vcf_path, const Range& region, bool enforce_multiallelic) {
  auto vcf_file = VariantFileReader::Open(vcf_path);

  // For phases 1 & 2 don't load genotypes, just the variant alleles, to reduce memory usage when storing all variants in the region.
  {  // Scope for vector of variants
    ReferenceNodes ref_nodes(graph_, reference_fasta_path, region);
    size_t num_variant_paths = 0;  // Track number of variant allele paths that will be defined
    std::vector<VariantFileReader::VariantPtr> variants;
    {  // Scope of collecting reference breakpoints/
      // 1. Collect unique variant breakpoints to construct reference nodes
      std::vector<Pos> ref_breakpoints = {region.start(), region.end()}, zero_width_breakpoints;
      vcf_file->SetRegion(region);
      while (auto variant = vcf_file->NextVariant()) {
        auto ref_region = variant->ReferenceRegion();
        if (ref_region.start() <= region.start() || ref_region.end() >= region.end()) {
          continue; // Skip variants that only partially overlap the graph region
        }
        if (variant->has_flag(Variant::kHasStarAllele) && variant->num_alts() == 1) {
          continue; // Skip variants with only '*' ALTS
        }

        num_variant_paths++;
        for (int i = 1; i < variant->num_alleles(); i++) {
          auto region = variant->AlleleReferenceRegion(i);
          if (region) {
            ref_breakpoints.push_back(region->start());
            ref_breakpoints.push_back(region->end());
            if (region->length() == 0) {
              zero_width_breakpoints.push_back(region->start());
            }
            num_variant_paths++;
          }
        }
        variants.emplace_back(std::move(variant));
      }

      SortAndUniquify(ref_breakpoints);
      SortAndUniquify(zero_width_breakpoints);

      // Handle any zero-width regions before the first reference breakpoint
      auto next_zw = zero_width_breakpoints.begin();
      if (next_zw != zero_width_breakpoints.end() && *next_zw <= ref_breakpoints[0]) {
        ref_nodes.CreateRefNode(*next_zw, *next_zw);
        ++next_zw;
      }
      for (int i = 0; i < ref_breakpoints.size() - 1; i++) {
        ref_nodes.CreateRefNode(ref_breakpoints[i], ref_breakpoints[i + 1]);
        // Insert any zero-width regions (for insertions). There can only be one for each breakpoint interval.
        if (next_zw != zero_width_breakpoints.end() && *next_zw <= ref_breakpoints[i + 1]) {
          ref_nodes.CreateRefNode(*next_zw, *next_zw);
          ++next_zw;
        }
      }

      // Add path for the reference contig
      auto ref_path = ref_nodes.AddRefPath();
      assert(graph_.get_path_count() == 1 && handlegraph::as_integer(ref_path) == 1);  // So far only the reference path (index 1), should exist
    }

    // 2. Add variant alleles as paths (recording which variant paths are associated with each node)
    std::vector<PathIdSet> node_variant_paths(graph_.max_node_id() + 1, PathIdSet(num_variant_paths + 2));
    std::vector<size_t> variant_path_starts;
    for (const auto& variant : variants) {
      auto variant_id = variant->variant_id();

      std::vector<std::optional<Range>> allele_regions;
      for (int i = 0; i < variant->num_alleles(); i++) {
        allele_regions.emplace_back(variant->AlleleReferenceRegion(i));
        // Defined ALT region must be fully contained in the REF region
        assert(i == 0 || !allele_regions[i] || (*allele_regions[i] <= *allele_regions[0]));
      }

      // Identify the nodes that cover the reference region. If one of the alleles is a simple insertion (i.e., its reference region has
      // zero length), make sure the reference path includes the zero-length node.
      auto ref_handles = ref_nodes.FindContainedHandles(*allele_regions[0]);
      for (int i = 1; i < variant->num_alleles(); i++) {
        auto alt_region = allele_regions[i];
        if (!alt_region || alt_region->length() > 0) {
          continue;  // Spanning deletion or non-zero length region, nothing to do here
        }
        auto alt_handles = ref_nodes.FindContainedHandles(*alt_region);
        // Ensure the reference handles include the zero-length node for this insertion allele
        ref_handles = std::make_pair(std::min(ref_handles.first, alt_handles.first),
                                     std::max(ref_handles.second, alt_handles.second));
      }
      // Add the path for the REF allele
      auto ref_path =ref_nodes.AddVariantPath(variant_id, 0, ref_handles);
      variant_path_starts.push_back(handlegraph::as_integer(ref_path));

      // Add the path for each ALT allele
      for (int i = 1; i < variant->num_alleles(); i++) {
        auto alt_region = allele_regions[i];
        if (!alt_region) {
          continue;  // Spanning deletion, no path to add
        }
        auto alt_sequence = variant->AlleleSequence(i);
        assert(alt_sequence);  // Should always have a valid sequence since spanning deletions were handled above

        auto alt_ref_handles = ref_nodes.FindContainedHandles(*alt_region);

        // Create or return a handle for the ALT allele sequence with appropriate edges
        auto alt_handle = ref_nodes.GetOrCreateAltHandle(alt_sequence.value(), alt_ref_handles);      

        // Create path with prefix that links (potentially) smaller ALT allele to full extent of the variant's
        // reference path, then the alternate allele node, then the suffix to link the ALT allele to the end of the
        // reference path. This ensures the variant's REF and ALT form a bubble (have the same predecessor and
        // successor nodes).
        auto alt_path =ref_nodes.AddVariantPath(variant_id, i, std::make_pair(ref_handles.first, alt_ref_handles.first), alt_handle,
                                 std::make_pair(alt_ref_handles.second, ref_handles.second));

        // Record paths that distinguish the ALT allele paths for corresponding REF and ALT allele nodes
        auto alt_node_id = graph_.get_id(alt_handle);
        if (alt_node_id >= node_variant_paths.size()) {
          node_variant_paths.resize(alt_node_id + 1, PathIdSet(num_variant_paths + 2));
        }
        for (auto it = alt_ref_handles.first; it != alt_ref_handles.second; ++it) {
          auto node_id = graph_.get_id(*it);
          node_variant_paths[node_id].set(handlegraph::as_integer(ref_path));
        }
        node_variant_paths[alt_node_id].set(handlegraph::as_integer(alt_path));
      }
    }
    variant_path_starts.push_back(graph_.get_path_count() + 1);  // End sentinel (since path IDs are 1-based)
    assert(num_variant_paths + 1 == graph_.get_path_count());  // +1 for reference path

    // 3. Connect ALT allele nodes to successor ALT allele nodes
    ref_nodes.LinkAltHandles(node_variant_paths, variant_path_starts, enforce_multiallelic);

    // TODO: Create new nodes for possible reorderings of co-located INS variants. Link these nodes to relevant allele paths, even though
    // they are not actually included in the path. That permits subsequent traversal of all possible allele combinations while still
    // knowing which original variants the newly created alleles correspond to.
  
    // 4. Perform topological sort, remapping and compaction of the graph to ensure integer node ids on all paths are in
    // ascending sorted order. Since the VCF is by definition acyclic we can use odgi's `lazier_topological_order` algorithm.
    // Remap the variant path sets to the new node IDs at the same time.
  
    auto order = odgi::algorithms::lazier_topological_order(&graph_);
    graph_.apply_ordering(order, true /* compact IDs */);
    node_variant_paths_.resize(node_variant_paths.size());
    for (int i = 0; i < order.size(); i++) {
      // When compacting IDs, the new nodes will start at index 1
      node_variant_paths_[i+1] = std::move(node_variant_paths[graph_.get_id(order[i])]);
    }
    variant_path_starts_ = std::move(variant_path_starts);
  }

  // 5. Add genotype paths
  auto ref_nodes = PathNodes(region.contig()); // All samples share the same reference path
  assert(std::is_sorted(std::begin(ref_nodes), std::end(ref_nodes)));
  
  std::vector<detail::Polytype> polytypes;
  for (const auto& sample : vcf_file->Samples()) {
    polytypes.emplace_back(2 /* ploidy */, *this, ref_nodes, sample, region.contig());
  }

  std::optional<Range> prev_range;  // Region of previous variant(s)
  vcf_file->SetRegion(region);
  while (auto variant = vcf_file->NextVariant(BCF_UN_ALL)) {
    auto ref_region = variant->ReferenceRegion();
    if (ref_region.start() <= region.start() || ref_region.end() >= region.end()) {
      continue; // Skip variants that only partially overlap the graph region
    }

    if (variant->has_flag(Variant::kHasStarAllele) && variant->num_alts() == 1) {
      continue; // Skip variants with only '*' ALTS
    }
    
    auto genotypes = variant->Genotypes();
    assert(genotypes.size() == polytypes.size());

    int star_allele_index = -1;
    if (variant->has_flag(Variant::kHasStarAllele)) {
      // Skip variants with all genotypes just '*'
      star_allele_index = variant->AlleleIndex("*");
      assert(star_allele_index > 0);
      if (std::all_of(std::begin(genotypes), std::end(genotypes), [star_allele_index](const Variant::Genotype& genotype) {
            return genotype.AllAlleles(star_allele_index);
          })) {
        continue;
      }
      
      if (!std::any_of(std::begin(genotypes), std::end(genotypes), [star_allele_index](const Variant::Genotype& genotype) {
        return genotype.AnyAllele(star_allele_index);
      })) {
        // Only note '*' if present in one of the genotypes
        star_allele_index = -1;
      }
    }

    auto variant_range = variant->ReferenceRegion();
    if (prev_range && (*prev_range == variant_range || prev_range->Overlaps(variant_range))) {
      // Coordinate overlap with previous variant(s)
      prev_range->UnionWith(variant_range);
      variant->add_flag(Variant::kIsOverlapping);
    } else if (!prev_range && star_allele_index > 0) {
      // Treat variants with explicit '*' alleles as overlapping, even if there is not actual coordinate overlap
      prev_range = variant_range;
      variant->add_flag(Variant::kIsOverlapping);
    } else {
      // TODO: Look for overlaps in entire record region (i.e., including padding bases)?
      prev_range = variant_range;
    }

    // Extract reference and alternate allele paths
    PathHandleSeq allele_paths;
    {
      auto variant_id = variant->variant_id();
      for (int i=0; i < variant->num_alleles(); i++) {
        auto allele_path_name = fmt::format("_alt_{}_{}", to_string(variant_id), i);
        if (has_path(allele_path_name)) {
          allele_paths.push_back(graph_.get_path_handle(allele_path_name));
        } else {
          allele_paths.push_back(handlegraph::path_handle_t()); // Likely a '*' allele (with no path)
        }
      }
    }
    
    // Extract the index range of reference nodes for the REF allele, since that is constant for all samples
    NodeIdRange ref_allele_indices;
    {
      auto ref_allele_nodes = PathNodes(allele_paths[0]);
      assert(!ref_allele_nodes.empty());

      auto it = std::lower_bound(std::begin(ref_nodes), std::end(ref_nodes), ref_allele_nodes[0]);
      assert(std::distance(it, std::end(ref_nodes)) >= ref_allele_nodes.size() && std::equal(std::begin(ref_allele_nodes), std::end(ref_allele_nodes), it));
      
      auto start_idx = std::distance(std::begin(ref_nodes), it);
      ref_allele_indices = std::make_pair(start_idx, start_idx + ref_allele_nodes.size());
    }

    for (int i=0; i < polytypes.size(); i++) {
      polytypes[i].AddGenotype(*variant, allele_paths, ref_allele_indices, genotypes[i], star_allele_index);
    }
  }

  // Finalize any incomplete paths
  for (auto& polytype : polytypes) {
    polytype.FinalizePaths();
  }
}

// HandleGraph interface

size_t Graph::get_length(const handlegraph::handle_t& handle) const {
  auto handle_seq = graph_.get_sequence(handle);
  return (handle_seq != "*") ? handle_seq.size() : 0;
}

std::string Graph::get_sequence(const handlegraph::handle_t& handle) const {
  auto handle_seq = graph_.get_sequence(handle);
  return (handle_seq != "*") ? handle_seq : "";
}

std::vector<handlegraph::handle_t> Graph::PathHandles(const handlegraph::path_handle_t& path_handle) const {
  std::vector<handlegraph::handle_t> handles;
  graph_.for_each_step_in_path(path_handle, [&](const handlegraph::step_handle_t& step) {
    handles.emplace_back(graph_.get_handle_of_step(step));
  });
  return handles;
}

std::vector<handlegraph::handle_t> Graph::PathHandles(const std::string& path_name) const {
  return PathHandles(graph_.get_path_handle(path_name));
}

std::vector<odgi::nid_t> Graph::PathNodes(const handlegraph::path_handle_t& path_handle) const {
  std::vector<odgi::nid_t> nodes;
  graph_.for_each_step_in_path(path_handle, [&](const handlegraph::step_handle_t& step) {
    nodes.emplace_back(graph_.get_id(graph_.get_handle_of_step(step)));
  });
  return nodes;
}

std::vector<odgi::nid_t> Graph::PathNodes(const std::string& path_name) const {
  return PathNodes(graph_.get_path_handle(path_name));
}

std::string Graph::PathSequence(const handlegraph::path_handle_t& path_handle) const {
  std::string seq;
  graph_.for_each_step_in_path(path_handle, [&](const handlegraph::step_handle_t& step) {
    auto handle_seq = graph_.get_sequence(graph_.get_handle_of_step(step));
    if (handle_seq != "*")
      seq.append(handle_seq);
  });
  return seq;
}

std::string Graph::PathSequence(const std::string& path_name) const {
  return PathSequence(graph_.get_path_handle(path_name));
}

std::vector<std::string> Graph::SamplesIncluding(const NodeIdSeq& nodes) const {
  std::set<std::string> samples;
  for (auto & node : nodes) {
    auto handle = graph_.get_handle(node);
    graph_.for_each_step_on_handle(handle, [this, &samples](const handlegraph::step_handle_t& step) {
      auto path_name = graph_.get_path_name(graph_.get_path_handle_of_step(step));
      auto end_of_sample = path_name.find('#'); // Sample haplotypes are named Sample#HaplotypeIndex#Contig#SegmentIndex
      if (end_of_sample != std::string::npos) {
        samples.emplace(path_name.substr(0, end_of_sample));
      }
    });
  }
  return std::vector<std::string>(samples.begin(), samples.end());
}

namespace {
  /**
   * @brief State for path enumeration via dynamic programming
   */
struct AllPathState {
  AllPathState(size_t inference_paths_size = 0) : paths(inference_paths_size) {}

  size_t path_cost = std::numeric_limits<size_t>::max();
  size_t path_count = 0;
  std::vector<odgi::nid_t> incoming_nodes; // Retain these edges for path enumeration
  Graph::PathIdSet paths; // All paths that traverse this node
};
}  // namespace

AllPathGraphOverlay Graph::AllPaths(const std::string& inference_vcf, const std::string& backbone_prefix, const Range& region, int min_size) const {
  // Identify inference alleles, marking the corresponding identifying nodes and recording the associated inference paths
  NodeIdSet inference_nodes(graph_.max_node_id() + 1);
  PathIdSet inference_paths(variant_path_starts_.back());
  
  auto inference_vcf_file = VariantFileReader::Open(inference_vcf);
  inference_vcf_file->SetRegion(region);
  while (auto variant = inference_vcf_file->NextVariant()) {
    auto variant_id = variant->variant_id();
    
    auto ref_path = graph_.get_path_handle(AltPathName(variant_id, 0));
    auto ref_path_idx = as_integer(ref_path);
    auto ref_nodes = PathNodes(ref_path);
    for (int i=1; i < variant->num_alleles(); i++) {
      auto change = variant->AlleleLengthChange(i);
      if (!change || std::abs(*change) < min_size) {
        continue; // Only consider SV alleles
      }

      auto alt_path = graph_.get_path_handle(AltPathName(variant_id, i));
      auto alt_path_idx = as_integer(alt_path);
      
      // Mark the differing nodes between alt and ref paths as zero cost and set the corresponding inference path bit
      auto alt_nodes = PathNodes(alt_path);
      auto [ref_prefix, ref_suffix, alt_prefix, alt_suffix] = detail::TrimSequence(ref_nodes, alt_nodes);
      for (auto it = ref_prefix; it != ref_suffix; ++it) {
        inference_nodes.set(*it);
        inference_paths.set(ref_path_idx);
      }
      for (auto it = alt_prefix; it != alt_suffix; ++it) {
        inference_nodes.set(*it);
        inference_paths.set(alt_path_idx);
      }
    }
  }

  // State for DP path enumeration algorithm
  NodeIdSet zero_cost_nodes(inference_nodes);
  std::vector<AllPathState> node_state(graph_.max_node_id() + 1, AllPathState(variant_path_starts_.back()));

  // Transfer the inference path sets into each node states. The state above only depends on the inference VCF, not the backbone paths
  // and so could be pre-computed once for multiple backbone prefixes.
  for (odgi::nid_t i = graph_.min_node_id(); i <= graph_.max_node_id(); i++) {
    if (graph_.has_node(i) && inference_nodes.test(i)) {
      node_state[i].paths = node_variant_paths_[i] & inference_paths;
    }
  }

  // Set potential backbone paths as "zero cost"
  graph_.for_each_path_handle([&](const handlegraph::path_handle_t& path_handle) {
    auto path_name = graph_.get_path_name(path_handle);
    if (backbone_prefix.size() > path_name.size() || !std::equal(std::begin(backbone_prefix), std::end(backbone_prefix), std::begin(path_name))) {
      return; // Not a potential backbone path
    }
    // Mark all nodes on this backbone path as zero cost
    graph_.for_each_step_in_path(path_handle, [&](const handlegraph::step_handle_t& step) {
      auto handle = graph_.get_handle_of_step(step);
      zero_cost_nodes.set(graph_.get_id(handle));
    });
  });

  // Return an overlay encoding all possible paths for this graph, inference VCF, and backbone prefix
  AllPathGraphOverlay overlay(*this, inference_nodes, inference_paths);

  node_state[graph_.min_node_id()].path_cost = 0;
  node_state[graph_.min_node_id()].path_count = 1;
  for (auto i=graph_.min_node_id(); i <= graph_.max_node_id(); i++) {
    if (!graph_.has_node(i)) {
      continue;
    }
    auto handle = graph_.get_handle(i);
    auto & state = node_state[i];
    auto & overlay_state = overlay.node_states_[i];

    std::vector<std::pair<odgi::nid_t, size_t>> predecessors;
    graph_.follow_edges(handle, true /* previous nodes */, [&](const handlegraph::handle_t& from_handle) {
      predecessors.emplace_back(graph_.get_id(from_handle), predecessors.size());
    });
    std::sort(std::begin(predecessors), std::end(predecessors), [&](const auto& a, const auto& b) {
      auto & a_state = node_state[a.first];
      auto & b_state = node_state[b.first];
      if (a_state.paths == b_state.paths) {
        return a_state.path_cost < b_state.path_cost;
      }
      return a_state.paths < b_state.paths;
    });
    overlay_state.InitEdges(predecessors.size());

    // Iterate through the groups formed by inference paths, retaining the predecessor with the minimum path cost in each group. This represents
    // the optimal way to reach node `i` for each distinct set of inference paths.
    for (size_t g=0; g < predecessors.size(); ) {
      auto pred_id = predecessors[g].first;
      node_state[i].incoming_nodes.push_back(pred_id);
      overlay_state.SetEdge(predecessors[g].second);
      
      auto & pred_state = node_state[pred_id];
      state.paths |= pred_state.paths;
      state.path_cost = std::min(state.path_cost, pred_state.path_cost);
      state.path_count += pred_state.path_count;

      // Skip to the start of the next group
      g = std::distance(std::begin(predecessors), std::find_if(std::begin(predecessors) + g + 1, std::end(predecessors), [&](const auto& pred) {
        return node_state[pred.first].paths != node_state[pred_id].paths;
      }));
    }
    // Update path cost with cost of traversing this node
    if (!zero_cost_nodes[i]) {
      const auto & sequence = graph_.get_sequence(handle);
      state.path_cost += sequence == "*" ? 0 : sequence.size();
    }
  }
  // We assume that the graph is a bubble, e.g., initial node has no predecessors and final node has no successors.
  assert(node_state[graph_.min_node_id()].incoming_nodes.empty());
  overlay.total_paths_ = node_state[graph_.max_node_id()].path_count;

  return overlay;
};

std::string Graph::AltPathName(const Variant::VariantId& variant_id, int allele,
                               const std::string& path_prefix) const {
  return fmt::format("_{}_{}_{}", path_prefix, to_string(variant_id), allele);
}

void Graph::ToGFA(std::ostream& ostream) {
  graph_.to_gfa(ostream);
}

namespace {

struct KmerDFSState {
  std::vector<handlegraph::handle_t> handles_;
  uint64_t starting_handle_offset_ = 0;
  std::string buffer_;
};

void KmersDFS(
    const handlegraph::HandleGraph& graph, size_t k, size_t edge_max,
    const std::function<void(const std::string&, const std::vector<handlegraph::handle_t>&, uint64_t)>& callback,
    KmerDFSState& state, size_t buffer_offset = 0) {
  auto init_buffer_size = state.buffer_.size();

  // We should only call this function using a buffer that is not yet long enough to generate a full k-mer, i.e.
  // that contains the last k-1 bases of the current path. Any full k-mers should already have been generated.
  assert(buffer_offset < k && init_buffer_size < (buffer_offset + k));

  // Stop if we've reached the maximum edge traversal depth (`n` handles is `n-1` edges)
  if (state.handles_.size() > edge_max) {
    return;
  }

  graph.follow_edges(state.handles_.back(), false /* traverse forward */, [&](const handlegraph::handle_t& next_handle) {
    state.buffer_.append(graph.get_sequence(next_handle), 0 /* start pos */, k-1);
    state.handles_.push_back(next_handle);

    // Generate all the k-mers that can be formed by extending the current buffer with the sequence of this node.
    size_t offset = buffer_offset;
    for (; (offset  +  k) <= state.buffer_.size(); ++offset) {
      callback(state.buffer_.substr(offset, k), state.handles_, state.starting_handle_offset_ + offset);
    }

    // If this node was not long enough to exhaut all possible prefixes, recurse into the next node to try to generate any remaining
    // k-mers that started in the original node.
    if ((state.buffer_.size() - init_buffer_size) < (k - 1)) {
      KmersDFS(graph, k, edge_max, callback, state, offset);
    }

    // Reset handles and buffer after completing this node and its children
    state.handles_.pop_back();
    state.buffer_.resize(init_buffer_size);
  });
}

} // namespace

void Graph::Kmers(size_t k, size_t edge_max, const std::function<void(const std::string&, const std::vector<handlegraph::handle_t>&, uint64_t)>& callback) const {
  KmerDFSState state;

  graph_.for_each_handle([&](const handlegraph::handle_t &handle) {
    auto seq = get_sequence(handle);
    if (seq.empty()) {
      return true; // Skip null nodes (i.e., deletions), continuing iteration
    }
    size_t handle_offset = 0;
    for (; (handle_offset + k) <= seq.size(); ++handle_offset) {
      callback(seq.substr(handle_offset, k), std::vector<handlegraph::handle_t>{handle}, handle_offset);
    }

    // Recursively extend k-mers from this handle along outgoing edges, up to the specified edge limit.
    state.handles_ = {handle};
    state.starting_handle_offset_ = handle_offset;
    state.buffer_ = seq.substr(handle_offset, k - 1);
    KmersDFS(*this, k, edge_max, callback, state);

    return true; // continue for_each_handle
  }, false /* parallel */);
}

void AllPathGraphOverlay::ForEachPath(const function<void(const CallbackIter&, const CallbackIter&, const Graph::PathIdSet&)>& callback) const { 
    CallbackSeq current_path({graph_.get_handle(graph_.max_node_id())});
    return ForEachPath(current_path, callback);
  }

void AllPathGraphOverlay::ForEachPath(CallbackSeq& current_path, const function<void(const CallbackIter&, const CallbackIter&, const Graph::PathIdSet&)>& callback) const {
  // DFS enumeration of all paths from sink back to the source
  auto & current_handle = current_path.back();
  auto current_node = graph_.get_id(current_handle);
  if (current_node == graph_.min_node_id()) {
    // Compute inference paths for this complete path
    Graph::PathIdSet inference_paths(graph_.variant_path_starts_.back()); //inference_path_handles_.size());
    for (auto it = current_path.rbegin(); it != current_path.rend(); ++it) {
      auto node_id = graph_.get_id(*it);
      if (node_mask_.test(node_id)) {
        inference_paths |= (graph_.node_variant_paths_[node_id] & path_mask_);
      }
    }
    
    // Set "reference" paths when no path is selected for a given inference variant, even if the path does
    // not explicitly traverse the reference allele nodes, i.e., if no one in corresponding region for the variant
    // set the reference path bit.
    for (size_t i=0; i < graph_.variant_path_starts_.size() - 1; i++) {
      Graph::PathIdSet variant_paths(inference_paths.size());
      variant_paths.set(graph_.variant_path_starts_[i], graph_.variant_path_starts_[i + 1]-graph_.variant_path_starts_[i], true);
      if ((inference_paths & variant_paths).none()) {
        inference_paths.set(graph_.variant_path_starts_[i]); // Set reference path if no path selected for this variant
      }
    }
    
    callback(current_path.rbegin(), current_path.rend(), inference_paths);
  } else {
    size_t prev_edge = 0;
    graph_.graph_.follow_edges(current_handle, true /* previous nodes */, [&](const handlegraph::handle_t& from_handle) {
      if (node_states_[current_node].pred_edges.test(prev_edge++)) {
        current_path.push_back(from_handle);
        ForEachPath(current_path, callback);
        current_path.pop_back();
      }
    });
  }
}

namespace detail {

void Polytype::AddGenotype(const Variant& variant, const Graph::PathHandleSeq& allele_paths,
                           const Graph::NodeIdRange& ref_allele_indices, const Variant::Genotype& genotype,
                           int star_allele_index) {
  if (genotype.num_alleles() == 0 || genotype.AllAlleles(Variant::Genotype::kMissingAllele)) {
    return; // No alleles to add
  }
  if (genotype.num_alleles() != haplotypes_.size()) {
    throw std::runtime_error("Different ploidy for genotypes not currently supported");
  }

  auto variant_phase = genotype.phase();
  bool permute_alleles = false;
  if (variant_phase == Phase::kUnphased && star_allele_index > 0 && (genotype.AlleleCount(star_allele_index) == genotype.num_alleles() - 1)) {
    // Implicitly phase variants with '*' alleles, allowing permutation of originally unphased variants
    // if needed to try to find a consistent phasing.
    variant_phase = Phase(Phase::kImplicit); 
    permute_alleles = true;
  }

  auto [next_phase, break_kind] = NextPhase(variant_phase);
  
  Variant::Genotype::AlleleIndices indices(genotype.allele_indices());
  bool added_genotype = false;
  do {
    int h = 0;
    try {
      for (; h < genotype.num_alleles(); h++) {
        if (indices[h] == Variant::Genotype::kMissingAllele || (star_allele_index > 0 && indices[h] == star_allele_index)) {
          continue; // Skip missing alleles or '*' alleles
        }
        haplotypes_[h].AddGenotypeAllele(variant, ref_allele_indices, indices[h], allele_paths[indices[h]], break_kind);
      }
      added_genotype = true;
      break; // Successfully added alleles, finalize haplotypes and terminate permutation loop
    } catch (const NonRefAlleleOverlappingError&) {
      // Inconsistent haplotypes, undo any actions taken so far
      for (int i=0; i <= h; i++) {
        haplotypes_[i].UndoActions();
      }
    } 
  } while (permute_alleles && std::next_permutation(std::begin(indices), std::begin(indices) + genotype.num_alleles()));
  
  if (!added_genotype) {
    for (int h = 0; h < genotype.num_alleles(); h++) {
      if (indices[h] == Variant::Genotype::kMissingAllele || (star_allele_index > 0 && indices[h] == star_allele_index)) {
        continue; // Skip missing alleles or '*' alleles
      }
      haplotypes_[h].AddGenotypeAllele(variant, ref_allele_indices, indices[h], allele_paths[indices[h]], Haplotype::kBreakInconsistent);
    }
  }
  
  for (auto & haplotype : haplotypes_) {
    haplotype.CommitActions();
  }
  current_phase_ = next_phase;
}

std::tuple<Phase,Haplotype::BreakKind> Polytype::NextPhase(const Phase& var_phase) const {
  #define NEXT_CASE(curr, var, next, break_kind) if (current_phase_ == Phase::curr && var_phase == Phase::var) { return std::make_tuple(Phase(Phase::next), break_kind); }
  #define NEXT_CASE_VAR(curr, var, break_kind) if (current_phase_ == Phase::curr && var_phase == Phase::var) { return std::make_tuple(var_phase, break_kind); }
  #define NEXT_CASE_CURR(curr, var, break_kind) if (current_phase_ == Phase::curr && var_phase == Phase::var) { return std::make_tuple(current_phase_, break_kind); }

  NEXT_CASE(kUnphased, kUnphased, kUnphased, Haplotype::kBreakBefore)
  NEXT_CASE(kUnphased, kGlobal, kGlobal, Haplotype::kBreakBefore)
  NEXT_CASE_VAR(kUnphased, kLocal, Haplotype::kBreakBefore)
  NEXT_CASE(kUnphased, kImplicit, kUnphased, Haplotype::kBreakNone)
  
  NEXT_CASE(kGlobal, kUnphased, kUnphased, Haplotype::kBreakBefore)
  NEXT_CASE(kGlobal, kGlobal, kGlobal, Haplotype::kBreakNone)
  NEXT_CASE_VAR(kGlobal, kLocal, Haplotype::kBreakBefore)
  NEXT_CASE(kGlobal, kImplicit, kGlobal, Haplotype::kBreakNone)

  NEXT_CASE(kLocal, kUnphased, kUnphased, Haplotype::kBreakBefore)
  NEXT_CASE(kLocal, kGlobal, kGlobal, Haplotype::kBreakBefore)
  if (current_phase_ == Phase::kLocal && var_phase == Phase::kLocal) {
    return std::make_tuple(var_phase, current_phase_ != var_phase ? Haplotype::kBreakBefore : Haplotype::kBreakNone);
  }
  NEXT_CASE_CURR(kLocal, kImplicit, Haplotype::kBreakNone)

  NEXT_CASE(kImplicit, kUnphased, kUnphased, Haplotype::kBreakNone)
  NEXT_CASE(kImplicit, kGlobal, kGlobal, Haplotype::kBreakNone)
  NEXT_CASE_VAR(kImplicit, kLocal, Haplotype::kBreakNone)
  NEXT_CASE(kImplicit, kImplicit, kImplicit, Haplotype::kBreakNone)
  else {
    throw std::runtime_error("Phasing transition not yet implemented");
  }

  #undef NEXT_CASE_CURR
  #undef NEXT_CASE_VAR
  #undef NEXT_CASE
}

void Polytype::FinalizePaths() {
  for (auto& haplotype : haplotypes_) {
    haplotype.FinalizePaths();
  }
}

Haplotype::Haplotype(int index, Graph& graph, const Graph::NodeIdSeq& ref_nodes, const std::string& sample, const ContigName& contig) : index_(index), graph_(graph), ref_nodes_(ref_nodes), sample_(sample), contig_(contig) {
  current_segment_handle_ = graph_.graph_.create_path_handle(PathName());
}

void Haplotype::AddGenotypeAllele(const Variant& variant, const Graph::NodeIdRange& ref_allele_indices, int allele_index, const Graph::PathHandleSeq::value_type& allele_path, BreakKind break_kind) {
  if (ref_allele_indices.first < next_ref_index_) {
    if (variant.has_flag(Variant::kIsOverlapping) && allele_index == 0) {
      // We have found an overlapping reference allele (without an explicit * allele). Skip it.
      return;
    } else if (break_kind == kBreakInconsistent) {
      // Inconsistent haplotype, introduce break and reset next_ref_index_ to expected nodes
      AddSegment();
      next_ref_index_ = ref_allele_indices.first;
    } else {
      // Inconsistent haplotype, potentially try to recover by permuting the genotype
      throw NonRefAlleleOverlappingError();
    }
  }
  if (break_kind == kBreakBefore) {
    // Fill in any pending reference nodes before introducing a break in the haplotype
    AddReferenceNodes(ref_allele_indices.first);
    AddSegment();
  }
  
  if (allele_index > 0) { // Insert alternate allele
    // Fill in any pending reference nodes (also saving current state of haplotype to enable undo)
    AddReferenceNodes(ref_allele_indices.first);
    
    // Insert the ALT allele handles, not including any suffix shared with the REF allele
    auto alt_allele_nodes = graph_.PathNodes(allele_path);
    auto ref_allele_nodes = boost::span<const odgi::nid_t>(ref_nodes_).subspan(ref_allele_indices.first, ref_allele_indices.second - ref_allele_indices.first);
    auto [alt_suffix_it, _ref_suffix_it] = std::mismatch(
      alt_allele_nodes.rbegin(), 
      alt_allele_nodes.rend(),
      ref_allele_nodes.rbegin(),
      ref_allele_nodes.rend()
    );
    auto alt_right_padding = std::distance(alt_allele_nodes.rbegin(), alt_suffix_it);
    assert(alt_right_padding == std::distance(ref_allele_nodes.rbegin(), _ref_suffix_it));
    for (int i=0; i < alt_allele_nodes.size() - alt_right_padding; i++) {
      graph_t().append_step(current_segment_handle_, graph_.get_handle(alt_allele_nodes[i]));
    }
    
    // Update next reference index
    next_ref_index_ = ref_allele_indices.second - alt_right_padding;
  }
}

void Haplotype::UndoActions() {
  std::for_each(actions_.rbegin(), actions_.rend(), [this](const std::unique_ptr<HaplotypeAction>& action) { action->Undo(*this); });
  actions_.clear();
}

void Haplotype::CommitActions() {
  actions_.clear();
}

void Haplotype::FinalizePaths() {
  // Populate final segment with any pending reference nodes
  AddReferenceNodes(ref_nodes_.size());
}

std::string Haplotype::PathName() const {
 return fmt::format("{}#{}#{}#{}", sample_, index_, contig_, curr_segment_);
}

void Haplotype::AddReferenceNodes(size_t end_index) {
  actions_.push_back(std::move(std::make_unique<HaplotypeAddSteps>(*this)));
  for (; next_ref_index_ < end_index; ++next_ref_index_) {
    graph_t().append_step(current_segment_handle_, graph_t().get_handle(ref_nodes_[next_ref_index_]));
  }
}

void Haplotype::AddSegment() {
  actions_.push_back(std::move(std::make_unique<HaplotypeAddSegment>(*this)));
  curr_segment_++;
  current_segment_handle_ = graph_t().create_path_handle(PathName());
}

HaplotypeAddSteps::HaplotypeAddSteps(const Haplotype& haplotype)
    : curr_next_ref_index_(haplotype.next_ref_index_),
      curr_step_(haplotype.graph_.path_back(haplotype.current_segment_handle_)) {}

void HaplotypeAddSteps::Undo(Haplotype& haplotype) const {
  // Remove any steps that were added after the save point and reset the ref index
  auto end = haplotype.graph_t().path_front_end(haplotype.current_segment_handle_);
  auto step = haplotype.graph_.path_back(haplotype.current_segment_handle_);
  while (step != end) {
    if (step == curr_step_) break;
    auto prev = haplotype.graph_t().get_previous_step(step);
    haplotype.graph_t().destroy_step(step);
    step = prev;
  }
  haplotype.next_ref_index_ = curr_next_ref_index_;
}

HaplotypeAddSegment::HaplotypeAddSegment(const Haplotype& haplotype)
    : current_segment_handle_(haplotype.current_segment_handle_) {}

void HaplotypeAddSegment::Undo(Haplotype& haplotype) const {
  haplotype.graph_.destroy_path(haplotype.current_segment_handle_);
  haplotype.current_segment_handle_ = current_segment_handle_;
}

}  // namespace detail
}  // namespace npsv3
