#include "graph.hpp"

#include <algorithms/topological_sort.hpp>
#include <fmt/format.h>
#include <boost/core/span.hpp>

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
    if (!regions_.empty() && start < regions_.back().end()) {
      throw std::runtime_error("Reference node regions must be in sorted order");
    }
    regions_.emplace_back(region_.contig(), start, end);
    auto next_handle =
        graph_.create_handle(start == end ? "*" : ref_seq_.substr(start - region_.start(), end - start));
    if (!handles_.empty()) {
      graph_.create_edge(handles_.back(), next_handle);
    }
    handles_.push_back(next_handle);
  }

  void AddRefPath() {
    odgi::path_handle_t path_handle = graph_.create_path_handle(region_.contig());
    for (const auto& handle : handles_) {
      graph_.append_step(path_handle, handle);
    }
  }

  handlegraph::path_handle_t AddVariantPath(const Variant::VariantId& variant_id, int allele,
                                            const HandleRange& handles, const std::string& path_prefix = "alt") {
    handlegraph::path_handle_t path_handle = graph_.create_path_handle(fmt::format("_{}_{}_{}", path_prefix, to_string(variant_id), allele));
    for (auto it = handles.first; it != handles.second; ++it) {
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
      return std::make_pair(std::begin(handles_) + idx, std::begin(handles_) + idx + 1);
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
    auto handle = graph_.create_handle(alt_sequence.size() == 0 ? "*" : std::string(alt_sequence));  // TODO: Do we have to create a string here?
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
  auto vcf_file = VariantFileReader::Open(vcf_path);

  // For phases 1 & 2 don't load genotypes, just the variant alleles, to reduce memory usage when storing
  // all variants in the region.
  {  // Scope for vector of variants
    ReferenceNodes ref_nodes(graph_, reference_fasta_path, region);
    std::vector<VariantFileReader::VariantPtr> variants;
    {  // Scope of collecting reference breakpoints/
      // 1. Collect unique variant breakpoints to construct reference nodes
      std::vector<Pos> ref_breakpoints = {region.start(), region.end()}, zero_width_breakpoints;
      vcf_file->SetRegion(region);
      while (auto variant = vcf_file->NextVariant()) {
        for (int i = 1; i < variant->num_alleles(); i++) {
          auto region = variant->AlleleReferenceRegion(i);
          if (region) {
            ref_breakpoints.push_back(region->start());
            ref_breakpoints.push_back(region->end());
            if (region->length() == 0) {
              zero_width_breakpoints.push_back(region->start());
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
      auto variant_id = variant->variant_id();

      std::vector<std::optional<Range> > allele_regions;
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
      ref_nodes.AddVariantPath(variant_id, 0, ref_handles);

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
        auto alt_path_handle =
            ref_nodes.AddVariantPath(variant_id, i, std::make_pair(ref_handles.first, alt_ref_handles.first));
        graph_.append_step(alt_path_handle, alt_handle);
        for (auto it = alt_ref_handles.second; it != ref_handles.second; ++it) {
          graph_.append_step(alt_path_handle, *it);
        }
      }
    }
  }
  
  // 3. Perform topological sort and compaction of the graph to prepare for subsequent steps. Since the VCF
  // is by definition a DAG we can use odgi's `lazier_topological_order` algorithm.
  graph_.apply_ordering(odgi::algorithms::lazier_topological_order(&graph_), true /* compact IDs */);

  // 4. Add genotype paths
  auto ref_nodes = PathNodes(region.contig()); // All samples share the same reference path
  assert(std::is_sorted(std::begin(ref_nodes), std::end(ref_nodes)));
  
  std::vector<detail::Polytype> polytypes;
  for (const auto& sample : vcf_file->Samples()) {
    polytypes.emplace_back(2 /* ploidy */, *this, ref_nodes, sample, region.contig());
  }

  std::optional<Range> prev_range;  // Region of previous variant(s)
  
  vcf_file->SetRegion(region);
  while (auto variant = vcf_file->NextVariant(BCF_UN_ALL)) {
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

std::vector<handlegraph::handle_t> Graph::PathHandles(const std::string& path_name) const {
  std::vector<handlegraph::handle_t> handles;
  auto path_handle = graph_.get_path_handle(path_name);
  graph_.for_each_step_in_path(path_handle, [&](const handlegraph::step_handle_t& step) {
    handles.emplace_back(graph_.get_handle_of_step(step));
  });
  return handles;
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
      seq.append(graph_.get_sequence(graph_.get_handle_of_step(step)));
  });
  return seq;
}

std::string Graph::PathSequence(const std::string& path_name) const {
  return PathSequence(graph_.get_path_handle(path_name));
}

std::vector<std::string> Graph::SamplesIncluding(const NodeIdSeq& nodes) const {
  std::vector<std::string> samples;
  for (auto & node : nodes) {
    auto handle = graph_.get_handle(node);
    graph_.for_each_step_on_handle(handle, [this, &samples](const handlegraph::step_handle_t& step) {
      auto path_name = graph_.get_path_name(graph_.get_path_handle_of_step(step));
      auto end_of_sample = path_name.find('#'); // Sample haplotypes are named Sample#HaplotypeIndex#Contig#SegmentIndex
      if (end_of_sample != std::string::npos) {
        samples.emplace_back(path_name.substr(0, end_of_sample));
      }
    });
  }
  return samples;
}

void Graph::ToGFA(std::ostream& ostream) {
  graph_.to_gfa(ostream);
}

namespace detail {

void Polytype::AddGenotype(const Variant& variant, const Graph::PathHandleSeq& allele_paths,
                           const Graph::NodeIdRange& ref_allele_indices, const Variant::Genotype& genotype,
                           int star_allele_index) {
  assert(genotype.num_alleles() > 0);
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

  auto [next_phase, break_before] = NextPhase(variant_phase);
  
  Variant::Genotype::AlleleIndices indices(genotype.allele_indices());
  bool added_genotype = false;
  do {
    int h = 0;
    try {
      for (; h < genotype.num_alleles(); h++) {
        if (indices[h] == Variant::Genotype::kMissingAllele || (star_allele_index > 0 && indices[h] == star_allele_index)) {
          continue; // Skip missing alleles or '*' alleles
        }
        haplotypes_[h].AddGenotypeAllele(variant, ref_allele_indices, indices[h], allele_paths[indices[h]],
                                         break_before ? Haplotype::kBreakBefore : Haplotype::kBreakNone);
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

std::tuple<Phase,bool> Polytype::NextPhase(const Phase& var_phase) const {
  #define NEXT_CASE(curr, var, next, break_before) if (current_phase_ == Phase::curr && var_phase == Phase::var) { return std::make_tuple(Phase(Phase::next), break_before); }
  #define NEXT_CASE_VAR(curr, var, break_before) if (current_phase_ == Phase::curr && var_phase == Phase::var) { return std::make_tuple(var_phase, break_before); }
  #define NEXT_CASE_CURR(curr, var, break_before) if (current_phase_ == Phase::curr && var_phase == Phase::var) { return std::make_tuple(current_phase_, break_before); }

  NEXT_CASE(kUnphased, kUnphased, kUnphased, true)
  NEXT_CASE(kUnphased, kGlobal, kGlobal, true)
  NEXT_CASE_VAR(kUnphased, kLocal, true)
  NEXT_CASE(kUnphased, kImplicit, kUnphased, false)
  
  NEXT_CASE(kGlobal, kUnphased, kUnphased, true)
  NEXT_CASE(kGlobal, kGlobal, kGlobal, false)
  NEXT_CASE_VAR(kGlobal, kLocal, true)
  NEXT_CASE(kGlobal, kImplicit, kGlobal, false)

  NEXT_CASE(kLocal, kUnphased, kUnphased, true)
  NEXT_CASE(kLocal, kGlobal, kGlobal, true)
  if (current_phase_ == Phase::kLocal && var_phase == Phase::kLocal) {
    return std::make_tuple(var_phase, current_phase_ != var_phase);
  }
  NEXT_CASE_CURR(kLocal, kImplicit, false)

  NEXT_CASE(kImplicit, kUnphased, kUnphased, false)
  NEXT_CASE(kImplicit, kGlobal, kGlobal, false)
  NEXT_CASE_VAR(kImplicit, kLocal, false)
  NEXT_CASE(kImplicit, kImplicit, kImplicit, false)
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
  for (auto step = haplotype.graph_.path_back(haplotype.current_segment_handle_); step != end;
       step = haplotype.graph_t().get_previous_step(step)) {
    if (step == curr_step_) break;
    haplotype.graph_t().destroy_step(step);
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
