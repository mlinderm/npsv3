#include <array>
#include <vector>

#include <boost/dynamic_bitset.hpp>

#include "graph.hpp"
#include "kmer.hpp"

namespace npsv3 {

/**
 * @brief Overlay that greedily samples up to n haplotypes from the graph using
 *        graph-unique k-mer zygosity scores (Sirén et al. approach).
 */
class HaplotypeSamplerOverlay {
 public:
  using Haplotype = Graph::NodeIdSeq;
  
  struct Diplotype {
    size_t h1, h2; ///< Haplotype indices
    double score; ///< Score of this diplotype (higher is better)
  };

  struct Params {
    double homozygous_score = 1.0; ///< Initial score for HOMOZYGOUS k-mers
    double absent_score = -0.8; ///< Initial score for ABSENT k-mers
    double heterozygous_score = 0.0; ///< Initial score for HETEROZYGOUS k-mers
    double homozygous_discount = 0.9; ///< After selection: HOMOZYGOUS score *= this
    double het_adjustment = 0.05; ///< After selection: decrement HETEROZYGOUS score by this if on path, increment if not on path

    Params() {};
  };

  /**
   * @param graph        The variant graph (node IDs must be in topological order)
   * @param unique_kmers Pre-computed graph-unique k-mer map (from Graph::UniqueKmers())
   * @param counts       KmerCounts instance used to classify k-mer sequences
   * @param params       Scoring parameters (optional)
   */
  explicit HaplotypeSamplerOverlay(const Graph& graph, const UniqueKmersOverlay& unique_kmers, const Params& params = {});

  /**
   * @brief Construct with inference-VCF path filtering active.
   *
   * Paths returned by FindBestPaths will be restricted to those that differ at
   * variants in @p inference_vcf whose allele length change is >= @p min_size.
   */
  HaplotypeSamplerOverlay(const Graph& graph, const UniqueKmersOverlay& unique_kmers,
                          const std::string& inference_vcf,
                          const Range& region,
                          size_t min_size = 50,
                          const Params& params = {});

  /// Initialize (or reset) scores prior to sampling based on supplied k-mer counts without reconstructing the entire overlay.
  void InitializeScores(const KmerCounts& counts);

  /// Greedily select up to n haplotypes; returns one NodeIdSeq per haplotype.
  std::vector<Haplotype> SampleHaplotypes(size_t n);

  /// Return the up to n unique highest-scoring distinct paths through the graph using the current k-mer scores.
  std::vector<Haplotype> FindBestPaths(size_t n) const;

  /// Return the top n highest-scoring diplotypes from all pairs of the given
  /// candidate haplotypes, sorted by descending score.
  std::vector<Diplotype> SampleDiplotypes(const std::vector<Haplotype>& candidates, size_t n = 1) const;

  /// Number of unique non-universal k-mers collected during construction.
  size_t NumKmers() const { return kmers_.size(); }

  /// True if any k-mers are entirely within a single node.
  bool HasNodeKmers() const { return !node_kmers_.empty(); }

  /// True if any k-mers span multiple nodes (recorded as explict edges).
  bool HasEdgeKmers() const { return !edge_kmers_.empty(); }

 private:
  struct KmerInfo {
    KmerZygosity zygosity;
    double score;
  };

  struct Edge {
    odgi::nid_t from;
    odgi::nid_t to;

    bool operator<(const Edge& other) const noexcept{
      return std::tie(from, to) < std::tie(other.from, other.to);
    }
  };

  struct EdgeInfo {
    std::vector<odgi::nid_t> intermediate_nodes;  ///< [h1 ... h_{n-1}]
    std::vector<size_t> kmers; ///< All k-mer indices credited on traversal
  };

  const Graph& graph_;
  Params params_;

  Graph::NodeIdSet inference_node_mask_; ///< Variant nodes whose allele coverage is tracked in covered_paths
  Graph::PathIdSet inference_path_mask_; ///< Path-ID bits for tracked alleles (masks node_variant_paths_ lookups)
  bool apply_path_filter_ = false; ///< When true, skip paths with no covered inference alleles (set by inference-VCF constructor)

  std::vector<std::string> kmer_sequences_; ///< k-mer sequences
  std::vector<KmerInfo> kmers_; ///< k-mer zygosity and score information, parallel to kmer_sequences_
  std::unordered_map<odgi::nid_t, std::vector<size_t>> node_kmers_; ///< nid to k-mer indices
  std::map<Edge, std::vector<EdgeInfo>> edge_kmers_; ///< edge (from,to) to edge info (w/ intermediate nodes and k-mer indices)
  
  /// Return a bitset of length kmers_.size() indicating which k-mers lie on the given path.
  boost::dynamic_bitset<> KmersOnPath(const Graph::NodeIdSeq& path) const;

  /// Update the scores of k-mers having sampled @p path
  void UpdateScores(const Graph::NodeIdSeq& path);
};

}
