#pragma once

#include <odgi.hpp>
#include <vector>

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

  size_t NodeCount() const { return graph_.get_node_count(); }
  
  handlegraph::handle_t Handle(odgi::nid_t node_id) const { return graph_.get_handle(node_id); }
  odgi::nid_t NodeId(handlegraph::handle_t handle) const { return graph_.get_id(handle); }

  bool HasPath(const std::string& path_name) const { return graph_.has_path(path_name); }
  HandleSeq PathHandles(const std::string& path_name) const;
  NodeIdSeq PathNodes(handlegraph::path_handle_t path_handle) const;
  NodeIdSeq PathNodes(const std::string& path_name) const;

  std::vector<std::string> SamplesIncluding(const NodeIdSeq& nodes) const;

  void FromSource(odgi::nid_t start_id, odgi::nid_t end_id) const;

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
  NonRefAlleleOverlappingError() : std::runtime_error("Non-reference allele overlaps reference allele without explicit '*' allele") {}
};

class Polytype {
 public:
  template <typename... Args>
  Polytype(int ploidy, Args&&... haplotype_args) : current_phase_(Phase::kImplicit) {
    for (int i = 0; i < ploidy; ++i) {
      haplotypes_.emplace_back(i, std::forward<Args>(haplotype_args)...);
    }
  }

  void AddGenotype(const Variant& variant, const Graph::PathHandleSeq& allele_paths, const Graph::NodeIdRange& ref_allele_indices, const Variant::Genotype& genotype, int star_allele_index);
  std::tuple<Phase, bool> NextPhase(const Phase& current_phase) const;
  void FinalizePaths();
  

  friend Haplotype;

 private:
  std::vector<Haplotype> haplotypes_;
  Phase current_phase_;
};

class Haplotype {
 public:
  Haplotype(int index, Graph& graph, const Graph::NodeIdSeq& ref_nodes, const std::string& sample,
            const ContigName& contig);

  void AddGenotypeAllele(const Variant& variant,const Graph::NodeIdRange& ref_allele_indices, int allele_index, const Graph::PathHandleSeq::value_type& allele_path, bool break_before);
  void FinalizePaths();

 private:
  const int index_;

  Graph& graph_;
  const Graph::NodeIdSeq& ref_nodes_;
  std::string sample_;
  const ContigName& contig_;

  int curr_segment_ = 0;
  handlegraph::path_handle_t current_segment_handle_;
  size_t next_ref_index_ = 0;

  odgi::graph_t& graph_t() const { return graph_.graph_; }
  std::string PathName() const;
};

}  // namespace detail

}  // namespace npsv3
