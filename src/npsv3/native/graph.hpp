#pragma once

#include <odgi.hpp>
#include <vector>
#include <iosfwd>

#include "range.hpp"
#include "variant.hpp"

namespace npsv3 {
namespace detail {
class Polytype;
class Haplotype;
}  // namespace detail

class Graph {
 public:
  typedef std::vector<handlegraph::handle_t> HandleSeq;
  typedef std::vector<odgi::nid_t> NodeIdSeq;
  typedef std::pair<size_t, size_t> NodeIdRange;
  typedef std::vector<handlegraph::path_handle_t> PathHandleSeq;

  Graph(const std::string& reference_fasta_path, const std::string& vcf_path, const Range& region);

  // handlegraph interface
  handlegraph::handle_t get_handle(odgi::nid_t node_id) const { return graph_.get_handle(node_id); }
  odgi::nid_t get_id(handlegraph::handle_t handle) const { return graph_.get_id(handle); }
  size_t get_node_count() const { return graph_.get_node_count(); }

  bool has_path(const std::string& path_name) const { return graph_.has_path(path_name); }
  handlegraph::step_handle_t path_back(const handlegraph::path_handle_t& path) const { return graph_.path_back(path); }
  void destroy_path(const handlegraph::path_handle_t& path) { return graph_.destroy_path(path); }
  
  HandleSeq PathHandles(const std::string& path_name) const;
  NodeIdSeq PathNodes(const handlegraph::path_handle_t& path_handle) const;
  NodeIdSeq PathNodes(const std::string& path_name) const;
  std::string PathSequence(const handlegraph::path_handle_t& path_handle) const;
  std::string PathSequence(const std::string& path_name) const;

  std::vector<std::string> SamplesIncluding(const NodeIdSeq& nodes) const;

  void FromSource(odgi::nid_t start_id, odgi::nid_t end_id) const;


  void ToGFA(std::ostream&);

  friend detail::Polytype;
  friend detail::Haplotype;

 private:
  odgi::graph_t graph_;
};

namespace test {
void TestCreateGraph(const std::string& reference_fasta_path, const std::string& vcf_path);
}

namespace detail {
class NonRefAlleleOverlappingError : public std::runtime_error {
 public:
  NonRefAlleleOverlappingError()
      : std::runtime_error("Non-reference allele overlaps reference allele without explicit '*' allele") {}
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
  std::tuple<Phase, bool> NextPhase(const Phase& current_phase) const;
  void FinalizePaths();

  friend Haplotype;

 private:
  std::vector<Haplotype> haplotypes_;
  Phase current_phase_;
};

// Record actions taken on the haplotype to enable undo
class HaplotypeAction {
 public:
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

class HaplotypeAddSteps : public HaplotypeAction {
 public:
  HaplotypeAddSteps(const Haplotype&);

  void Undo(Haplotype&) const override;

 private:
  size_t curr_next_ref_index_;
  const handlegraph::step_handle_t& curr_step_;
};

class HaplotypeAddSegment : public HaplotypeAction {
 public:
  HaplotypeAddSegment(const Haplotype&);

  void Undo(Haplotype&) const override;

 private:
  handlegraph::path_handle_t current_segment_handle_;
};

}  // namespace detail

}  // namespace npsv3
