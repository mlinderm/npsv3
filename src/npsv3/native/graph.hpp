#pragma once

#include <odgi.hpp>
#include <handlegraph/handle_graph.hpp>
#include <algorithms/kmer.hpp>
#include <stdexcept>
#include <vector>
#include <iosfwd>
#include <functional>
#include <unordered_map>
#include <boost/dynamic_bitset.hpp>

#include "range.hpp"
#include "variant.hpp"
#include "kmer.hpp"

namespace npsv3 {
namespace detail {
class Polytype;
class Haplotype;
}  // namespace detail

class AllPathGraphOverlay;
class HaplotypeSamplerOverlay;

class Graph : public handlegraph::HandleGraph {
 public:
  typedef std::vector<handlegraph::handle_t> HandleSeq;
  typedef std::vector<odgi::nid_t> NodeIdSeq;
  typedef std::pair<size_t, size_t> NodeIdRange;
  typedef std::vector<handlegraph::path_handle_t> PathHandleSeq;
  typedef boost::dynamic_bitset<> NodeIdSet;
  typedef boost::dynamic_bitset<> PathIdSet;

  /**
   * @brief Construct a new pangenome Graph from a VCF in a specific region
   * 
   * Graph nodes are guaranteed to be in topological order.
   * 
   * @param reference_fasta_path Path to the reference FASTA file
   * @param vcf_path Path to the VCF file
   * @param region Construct graph in this region
   * @param enforce_multiallelic If true, do not create edges between alternate alleles of the same variant
   */
  Graph(const std::string& reference_fasta_path, const std::string& vcf_path, const Range& region, bool enforce_multiallelic = true);

  /** \name handlegraph interface */
  // @{
 public:
  handlegraph::handle_t get_handle(const odgi::nid_t& node_id, bool is_reverse = false) const { return graph_.get_handle(node_id, is_reverse); }
  odgi::nid_t get_id(const handlegraph::handle_t& handle) const { return graph_.get_id(handle); }
  size_t get_length(const handlegraph::handle_t& handle) const;
  std::string get_sequence(const handlegraph::handle_t& handle) const;
  bool has_node(handlegraph::nid_t node_id) const { return graph_.has_node(node_id); }
  bool get_is_reverse(const handlegraph::handle_t& handle) const { return graph_.get_is_reverse(handle); }
  handlegraph::handle_t flip(const handlegraph::handle_t& handle) const { return graph_.flip(handle); }

  size_t get_node_count() const { return graph_.get_node_count(); }
  odgi::nid_t min_node_id() const { return graph_.min_node_id(); }
  odgi::nid_t max_node_id() const { return graph_.max_node_id(); }

  bool has_edge(const handlegraph::handle_t& left, const handlegraph::handle_t& right) const { return graph_.has_edge(left, right); }

  bool has_path(const std::string& path_name) const { return graph_.has_path(path_name); }
  handlegraph::path_handle_t get_path_handle(const std::string& path_name) const;
  std::string get_path_name(const handlegraph::path_handle_t& path_handle) const { return graph_.get_path_name(path_handle); }
  handlegraph::step_handle_t path_back(const handlegraph::path_handle_t& path) const { return graph_.path_back(path); }
  handlegraph::step_handle_t path_front_end(const handlegraph::path_handle_t& path_handle) const { return graph_.path_front_end(path_handle); }
  bool is_empty(const handlegraph::path_handle_t& path_handle) const { return graph_.is_empty(path_handle); }
  void destroy_path(const handlegraph::path_handle_t& path) { return graph_.destroy_path(path); }

  handlegraph::step_handle_t get_previous_step(const handlegraph::step_handle_t& step_handle) const { return graph_.get_previous_step(step_handle); }

  odgi::graph_t::path_metadata_t& get_path_metadata(const handlegraph::path_handle_t& path) { return graph_.get_path_metadata(path); }

  void destroy_step(const handlegraph::step_handle_t& step_handle) { return graph_.destroy_step(step_handle); }

 protected:
  bool follow_edges_impl(const handlegraph::handle_t& handle, bool go_left, const std::function<bool(const handlegraph::handle_t&)>& iteratee) const override {
    return graph_.follow_edges(handle, go_left, iteratee);
  }

  bool for_each_handle_impl(const std::function<bool(const handlegraph::handle_t&)>& iteratee, bool parallel = false) const override {
    return graph_.for_each_handle(iteratee, parallel);
  }
  // @}

 public:

  /// Return standardized name for an alternate allele path
  std::string AltPathName(const Variant::VariantId& variant_id, int allele, const std::string& path_prefix = "alt") const; 

  HandleSeq PathHandles(const handlegraph::path_handle_t& path_handle) const;
  HandleSeq PathHandles(const std::string& path_name) const;
  NodeIdSeq PathNodes(const handlegraph::path_handle_t& path_handle) const;
  NodeIdSeq PathNodes(const std::string& path_name) const;

  /// Enumerate all path names with the given prefix and return a concatenated NodeIdSeq.
  /// Used to gather all segments of one sample haplotype (e.g., prefix = "SAMPLE#0#chr1").
  NodeIdSeq HaplotypePaths(const std::string& prefix) const;
  
  /** Return the sequence of a path from path name, handle or node ID iterator */
  // @{
  std::string PathSequence(const handlegraph::path_handle_t& path_handle) const;
  std::string PathSequence(const std::string& path_name) const;
  template<typename Iterator>
  std::string PathSequence(Iterator begin, Iterator end) const;
  // @}

  std::vector<std::string> SamplesIncluding(const NodeIdSeq& nodes) const;

  /**
   * @brief Construct an overlay for enumerating all paths traversing variants in the inference VCF
   * 
   * @param inference_vcf Path to the VCF file defining the variants for path enumeration
   * @param backbone_prefix Link inference variants with paths with this prefix, e.g., "chr1" or "sample1#0"
   * @param region Restrict inference variants to this region
   * @param min_size Minimum size of allele length change to consider for path enumeration
   * @return AllPathGraphOverlay 
   */
  AllPathGraphOverlay AllPaths(const std::string& inference_vcf, const std::string& backbone_prefix, const Range& region, int min_size=50) const;

  /**
   * @brief Generate kmers from the graph, invoking callback for each kmer with its path and offset information
   *
   * @param k Length of kmers to generate
   * @param max_edge Maximum number of edges to traverse when generating kmers
   * @param callback Callback function for each kmer
   */
  void Kmers(size_t k, size_t max_edge, const std::function<void(const std::string&, const std::vector<handlegraph::handle_t>&, uint64_t)>& callback) const;

  /**
   * @brief Generate kmers that appear at exactly one position in the graph (graph-unique kmers),
   *        invoking callback for each with its handles and offset information.
   *
   *        A kmer is graph-unique if its sequence appears at only one (handle, offset) position,
   *        even if that position is traversed by multiple haplotype paths.
   *
   * @param k Length of kmers to generate
   * @param max_edge Maximum number of edges to traverse when generating kmers
   * @param callback Callback function for each unique kmer
   * @param exclude_universal If true, suppress kmers that appear in every haplotype path
   */
  void UniqueKmers(size_t k, size_t max_edge, const std::function<void(const std::string&, const std::vector<handlegraph::handle_t>&, uint64_t)>& callback, bool exclude_universal=false) const;

  /** 
   * @brief Populate masks for nodes and paths based on an inference VCF
   * 
   * @param inference_vcf Path to the VCF file defining the variants for path enumeration
   * @param region Restrict inference variants to this region
   * @param min_size Minimum size of allele length change to consider for path enumeration
   * @param node_mask Mask for nodes
   * @param path_mask Mask for paths
   */
  void PopulateNodeAndPathMasks(const std::string& inference_vcf, const Range& region, size_t min_size, NodeIdSet& node_mask, PathIdSet& path_mask) const;

  /** @brief Write the graph in GFA format to an output stream
   * 
   * @param out Output stream to write the GFA format to
   */
  void ToGFA(std::ostream&);

  friend AllPathGraphOverlay;
  friend HaplotypeSamplerOverlay;
  friend detail::Polytype;
  friend detail::Haplotype;

 private:
  odgi::graph_t graph_;
  std::vector<PathIdSet> node_variant_paths_;
  std::vector<size_t> variant_path_starts_;
};

template<typename Iterator>
std::string Graph::PathSequence(Iterator begin, Iterator end) const {
  std::string seq;
  for (auto it = begin; it != end; ++it) {
    auto handle_seq = graph_.get_sequence(*it);
    if (handle_seq != "*")
      seq.append(handle_seq);
  }
  return seq;
}

namespace detail {
  struct AllPathState{
    typedef boost::dynamic_bitset<> PredecessorEdges;

    PredecessorEdges pred_edges;

    void InitEdges(size_t num_edges) { pred_edges.resize(num_edges); }
    void SetEdge(size_t edge_idx) { pred_edges.set(edge_idx); }
  };
} // namespace detail

/**
 * @brief Overlay for enumerating all paths traversing specific nodes in the graph
 * 
 */
class AllPathGraphOverlay {
 public:
  typedef std::vector<handlegraph::handle_t> CallbackSeq;
  typedef CallbackSeq::const_reverse_iterator CallbackIter;
  typedef function<void(const CallbackIter&, const CallbackIter&, const Graph::PathIdSet&)> CallbackFunction;

  /**
   * @brief Construct a new AllPathGraphOverlay object for a graph
   * 
   * @param graph 
   * @param node_mask, path_mask Only report paths according to path mask and in nodes specified in node mask
   */
  AllPathGraphOverlay(const Graph& graph, const Graph::NodeIdSet& node_mask, const Graph::PathIdSet& path_mask)
      : graph_(graph), node_mask_(node_mask), path_mask_(path_mask), node_states_(graph.max_node_id() + 1) {}

  size_t total_paths() const { return total_paths_; }

  /// Invoke callback for each path in the overlay
  void ForEachPath(const CallbackFunction& callback) const;

  friend Graph;

 private:
  const Graph& graph_;
  Graph::NodeIdSet node_mask_;
  Graph::PathIdSet path_mask_;
  std::vector<detail::AllPathState> node_states_;
  size_t total_paths_ = 0;
  
  void ForEachPath(CallbackSeq& current_path, const CallbackFunction& callback) const;
};


namespace detail {
template <typename Sequence1, typename Sequence2>
auto TrimSequence(const Sequence1& seq1, const Sequence2& seq2) {
  auto [prefix1, prefix2] = std::mismatch(std::begin(seq1), std::end(seq1), std::begin(seq2), std::end(seq2));
  auto [suffix1, suffix2] = std::mismatch(std::rbegin(seq1), std::rend(seq1), std::rbegin(seq2), std::rend(seq2));
  return std::make_tuple(prefix1, suffix1.base(), prefix2, suffix2.base());
}

class NonRefAlleleOverlappingError : public std::runtime_error {
 public:
  NonRefAlleleOverlappingError()
      : std::runtime_error("Non-reference allele overlaps reference allele without explicit '*' allele") {}
};

// Record actions taken on the haplotype to enable undo
class HaplotypeAction {
 public:
  virtual ~HaplotypeAction() = default;
  virtual void Undo(Haplotype& haplotype) const = 0;
};
class HaplotypeAddSteps;
class HaplotypeAddSegment;

class Haplotype {
 public:
  enum BreakKind : unsigned int { kBreakNone = 0, kBreakBefore, kBreakInconsistent };

  Haplotype(int index, Graph& graph, const Graph::NodeIdSeq& ref_nodes, const std::string& sample,
            const ContigName& contig);

  void AddGenotypeAllele(const Variant& variant, const Graph::NodeIdRange& ref_allele_indices, int allele_index,
                         const Graph::PathHandleSeq::value_type& allele_path, BreakKind break_kind);
  void UndoActions();
  void CommitActions();
  void FinalizePaths();

  friend HaplotypeAddSteps;
  friend HaplotypeAddSegment;

 private:
  const int index_;

  Graph& graph_;
  const Graph::NodeIdSeq& ref_nodes_;
  std::string sample_;
  const ContigName& contig_;

  int curr_segment_ = 0;
  handlegraph::path_handle_t current_segment_handle_;
  size_t next_ref_index_ = 0;

  std::vector<std::unique_ptr<HaplotypeAction>> actions_;

  odgi::graph_t& graph_t() const { return graph_.graph_; }
  std::string PathName() const;
  void AddReferenceNodes(size_t end_index);
  void AddSegment();
};

class Polytype {
 public:
  template <typename... Args>
  Polytype(int ploidy, Args&&... haplotype_args) : current_phase_(Phase::kImplicit) {
    for (int i = 0; i < ploidy; ++i) {
      haplotypes_.emplace_back(i, std::forward<Args>(haplotype_args)...);
    }
  }

  void AddGenotype(const Variant& variant, const Graph::PathHandleSeq& allele_paths,
                   const Graph::NodeIdRange& ref_allele_indices, const Variant::Genotype& genotype,
                   int star_allele_index);
  std::tuple<Phase, Haplotype::BreakKind> NextPhase(const Phase& current_phase) const;
  void FinalizePaths();

  friend Haplotype;

 private:
  std::vector<Haplotype> haplotypes_;
  Phase current_phase_;
};

class HaplotypeAddSteps : public HaplotypeAction {
 public:
  HaplotypeAddSteps(const Haplotype&);

  void Undo(Haplotype&) const override;

 private:
  size_t curr_next_ref_index_;
  handlegraph::step_handle_t curr_step_;
};

class HaplotypeAddSegment : public HaplotypeAction {
 public:
  HaplotypeAddSegment(const Haplotype&);

  void Undo(Haplotype&) const override;

 private:
  int current_segment_;
  handlegraph::path_handle_t current_segment_handle_;
};

}  // namespace detail

}  // namespace npsv3
