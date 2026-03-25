#include <vector>

#include "graph.hpp"
#include "kmer.hpp"

namespace npsv3 {

/**
 * @brief Overlay that greedily samples up to n haplotypes from the graph using
 *        graph-unique k-mer zygosity scores (Sirén et al. approach).
 */
class HaplotypeSamplerOverlay {
 public:
  struct Params {
    double homozygous_score = 1.0; ///< Initial score for HOMOZYGOUS k-mers
    double absent_score = -0.8; ///< Initial score for ABSENT k-mers
    double heterozygous_score = 0.0; ///< Initial score for HETEROZYGOUS k-mers
    double homozygous_discount = 0.9; ///< After selection: HOMOZYGOUS score *= this
    double het_adjustment = 0.05; ///< After selection: decrement HETEROZYGOUS score by this if on path, increment if not on path
  };

  /**
   * @param graph   The variant graph (node IDs must be in topological order)
   * @param k       K-mer length
   * @param max_edge Maximum edges traversed per k-mer
   * @param counts  KmerCounts instance used to classify k-mer sequences
   * @param params  Scoring parameters (optional)
   */
  HaplotypeSamplerOverlay(const Graph& graph, size_t k, size_t max_edge,
                          const KmerCounts& counts);
  HaplotypeSamplerOverlay(const Graph& graph, size_t k, size_t max_edge,
                          const KmerCounts& counts, Params params);

  /// Greedily select up to n haplotypes; returns one NodeIdSeq per haplotype.
  std::vector<Graph::NodeIdSeq> SampleHaplotypes(size_t n);

  /// Return the highest-scoring path through the graph using the current k-mer scores.
  Graph::NodeIdSeq FindBestPath() const;

  /// Number of unique non-universal k-mers collected during construction.
  size_t NumKmers() const { return kmers_.size(); }

  /// True if any k-mers are entirely within a single node.
  bool HasNodeKmers() const { return !node_kmers_.empty(); }

  /// True if any k-mers span multiple nodes (recorded as explict edges).
  bool HasEdgeKmers() const { return !edge_kmers_.empty(); }

 private:
  struct KmerInfo {
    std::string sequence;
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

  std::vector<KmerInfo> kmers_;
  std::unordered_map<odgi::nid_t, std::vector<size_t>> node_kmers_; ///< nid to k-mer indices
  std::map<Edge, std::vector<EdgeInfo>> edge_kmers_;  ///< edge (from,to) to edge info (w/ intermediate nodes and k-mer indices)
  
  void UpdateScores(const Graph::NodeIdSeq& path);

};

}