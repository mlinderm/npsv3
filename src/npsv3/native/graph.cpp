#include "graph.hpp"

#include <algorithms/topological_sort.hpp>

#include "fasta.hpp"
#include "variant.hpp"

namespace npsv3 {

namespace {
void SortAndUniquify(std::vector<Pos>& vec) {
  std::sort(vec.begin(), vec.end());
  auto last = std::unique(vec.begin(), vec.end());
  vec.erase(last, vec.end());
}

class ReferenceNodes {
 public:
  typedef std::vector<handlegraph::handle_t> HandleSeq;
  typedef std::pair<HandleSeq::const_iterator, HandleSeq::const_iterator> HandleRange;

  ReferenceNodes(odgi::graph_t& graph, const std::string& reference_fasta_path, const Range& region)
      : graph_(graph), region_(region) {
    FastaReader fasta_reader(reference_fasta_path);
    // TODO: Return a unique_ptr char* from FetchSequence to avoid the extra copy into a string?
    ref_seq_ = fasta_reader.FetchSequence(region_);
  }

  void CreateRefNode(Pos start, Pos end) {
    if (!regions_.empty() && start < regions_.back().End()) {
      throw std::runtime_error("Reference node regions must be in sorted order");
    }
    regions_.emplace_back(region_.Contig(), start, end);
    auto next_handle =
        graph_.create_handle(start == end ? "*" : ref_seq_.substr(start - region_.Start(), end - start));
    if (!handles_.empty()) {
      graph_.create_edge(handles_.back(), next_handle);
    }
    handles_.push_back(next_handle);
  }

  void AddRefPath() {
    odgi::path_handle_t path_handle = graph_.create_path_handle(region_.Contig());
    for (const auto& handle : handles_) {
      graph_.append_step(path_handle, handle);
    }
  }

  handlegraph::path_handle_t AddVariantPath(const Variant::VariantIdType& variant_id, int allele,
                                            const HandleRange& handles, const std::string& path_prefix = "_alt_") {
    std::ostringstream path_name_ss(path_prefix, std::ios_base::ate);
    path_name_ss << variant_id << '_' << allele;
    handlegraph::path_handle_t path_handle = graph_.create_path_handle(path_name_ss.str());
    for (auto it = handles.first; it != handles.second; ++it) {
      graph_.append_step(path_handle, *it);
    }
    return path_handle;
  }

  HandleRange FindContainedHandles(const Range& region) const {
    if (region.Length() == 0) {
      // There should be one exactly matching zero-length region
      auto it = std::lower_bound(std::begin(regions_), std::end(regions_), region.Start());
      if (it == std::end(regions_) || it->Start() != region.Start() || it->End() != region.End()) {
        throw std::runtime_error("No matching zero-length reference region found");
      }
      auto idx = std::distance(std::begin(regions_), it);
      return std::make_pair(std::begin(handles_) + idx, std::begin(handles_) + idx + 1);
    } else {
      auto it = std::lower_bound(std::begin(regions_), std::end(regions_), region.Start());
      if (it == std::end(regions_) || it->Start() != region.Start()) {
        throw std::runtime_error("No matching reference region start found");
      }
      // We assume most regions will be small and so perform linear search for the end
      for (auto end_it = it; end_it != std::end(regions_); ++end_it) {
        if (end_it->End() == region.End()) {  // There must be a region with a matching end
          auto start_idx = std::distance(std::begin(regions_), it);
          auto end_idx = std::distance(std::begin(regions_), end_it) + 1;
          return std::make_pair(std::begin(handles_) + start_idx, std::begin(handles_) + end_idx);
        }
      }
      // No matching end found
      throw std::runtime_error("No matching sequence of reference regions found");
    }
  }

  handlegraph::handle_t GetOrCreateAltHandle(const std::string_view& alt_sequence, const HandleRange& ref_handles) {
    // Have we already created this ALT allele node before?
    auto [begin, end] = alt_handles_.equal_range(ref_handles);
    for (auto it = begin; it != end; ++it) {
      auto handle = it->second;
      if (graph_.get_sequence(handle) == alt_sequence) {
        return handle;
      }
    }
    auto handle = graph_.create_handle(std::string(alt_sequence));  // TODO: Do we have to create a string here?
    // Link to the reference nodes before and after
    if (ref_handles.first != std::begin(handles_)) {
      graph_.create_edge(*std::prev(ref_handles.first), handle);
    }
    if (ref_handles.second != std::end(handles_)) {
      graph_.create_edge(handle, *(ref_handles.second));
    }

    alt_handles_.emplace(ref_handles, handle);
    return handle;
  }

  odgi::graph_t& graph_;
  const Range region_;
  std::string ref_seq_;

  std::vector<Range> regions_;
  HandleSeq handles_;
  std::multimap<HandleRange, handlegraph::handle_t> alt_handles_;
};

}  // namespace

Graph::Graph(const std::string& reference_fasta_path, const std::string& vcf_path, const Range& region) {
  ReferenceNodes ref_nodes(graph_, reference_fasta_path, region);
  auto vcf_file = VariantFileReader::Open(vcf_path);

  // For phases 1 & 2 don't load genotypes, just the variant alleles, to reduce memory usage when storing
  // all variants in the region.
  {  // Scope for vector of variants
    std::vector<VariantFileReader::VariantPtr> variants;
    {  // Scope of collecting reference breakpoints
      // 1. Collect unique variant breakpoints to construct reference nodes
      std::vector<Pos> ref_breakpoints = {region.Start(), region.End()}, zero_width_breakpoints;
      vcf_file->SetRegion(region);
      while (auto variant = vcf_file->NextVariant()) {
        for (int i = 1; i < variant->NumAllele(); i++) {
          auto region = variant->AlleleReferenceRegion(i);
          if (region) {
            ref_breakpoints.push_back(region->Start());
            ref_breakpoints.push_back(region->End());
            if (region->Length() == 0) {
              zero_width_breakpoints.push_back(region->Start());
            }
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
      ref_nodes.AddRefPath();
    }

    // 2. Add variant alleles as paths
    for (const auto& variant : variants) {
      auto variant_id = variant->VariantId();

      std::vector<std::optional<Range> > allele_regions;
      for (int i = 0; i < variant->NumAllele(); i++) {
        allele_regions.emplace_back(variant->AlleleReferenceRegion(i));
        // Defined ALT region must be fully contained in the REF region
        assert(i == 0 || !allele_regions[i] || (*allele_regions[i] <= *allele_regions[0]));
      }

      // Identify the nodes that cover the reference region. If one of the alleles is a simple insertion (i.e., its reference region has
      // zero length), make sure the reference path includes the zero-length node.
      auto ref_handles = ref_nodes.FindContainedHandles(*allele_regions[0]);
      for (int i = 1; i < variant->NumAllele(); i++) {
        auto alt_region = allele_regions[i];
        if (!alt_region || alt_region->Length() > 0) {
          continue;  // Spanning deletion or non-zero length region, nothing to do here
        }
        auto alt_handles = ref_nodes.FindContainedHandles(*alt_region);
        // Ensure the reference handles include the zero-length node for this insertion allele
        ref_handles = std::make_pair(std::min(ref_handles.first, alt_handles.first),
                                     std::max(ref_handles.second, alt_handles.second));
      }
      // Add the path for the REF allele
      ref_nodes.AddVariantPath(variant_id, 0, ref_handles);

      // Add the path for each ALT allele
      for (int i = 1; i < variant->NumAllele(); i++) {
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
        auto alt_path_handle =
            ref_nodes.AddVariantPath(variant_id, i, std::make_pair(ref_handles.first, alt_ref_handles.first));
        graph_.append_step(alt_path_handle, alt_handle);
        for (auto it = alt_ref_handles.second; it != ref_handles.second; ++it) {
          graph_.append_step(alt_path_handle, *it);
        }
      }
    }
  }
  
  // 3. Add genotype paths
  vcf_file->SetRegion(region);
  while (auto variant = vcf_file->NextVariant(BCF_UN_ALL)) {
    std::cerr << *variant;
  }

  // 4. Perform topological sort and compaction of the graph to prepare for subsequent analyses. Since the VCF
  // is by definition a DAG we can use odgi's `lazier_topological_order` algorithm.
  graph_.apply_ordering(odgi::algorithms::lazier_topological_order(&graph_), true /* compact IDs */);

  graph_.to_gfa(std::cerr);
}

std::vector<handlegraph::handle_t> Graph::PathHandles(const std::string& path_name) const {
  std::vector<handlegraph::handle_t> handles;
  auto path_handle = graph_.get_path_handle(path_name);
  graph_.for_each_step_in_path(path_handle, [&](const handlegraph::step_handle_t& step) {
    handles.emplace_back(graph_.get_handle_of_step(step));
  });
  return handles;
}

namespace test {
void TestCreateGraph(const std::string& reference_fasta_path, const std::string& vcf_path) {
  Graph graph(reference_fasta_path, vcf_path, Range("chr1", 3693757, 3693777));
}
}  // namespace test

}  // namespace npsv3
