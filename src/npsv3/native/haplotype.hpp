#include <algorithm>
#include <array>
#include <vector>

#include <boost/dynamic_bitset.hpp>
#include <boost/core/span.hpp>

#include "graph.hpp"
#include "kmer.hpp"

namespace npsv3 {

/**
 * Overlay that greedily samples haplotypes from the graph using graph-unique k-mer zygosity scores (adapted from Sirén et al.)
 */
class HaplotypeSamplerOverlay {
 public:
  using Haplotype = Graph::NodeIdSeq;
  using KmerNodeIdSeq = std::vector<odgi::nid_t>;
  using KmerIdSet = boost::dynamic_bitset<>;

  // Transparent comparator to enable heterogeneous lookup in PathKmerMap using
  // boost::span or std::array keys without constructing a KmerNodeIdSeq.
  struct NodeIdSeqLess {
    using is_transparent = void;
    template<typename L, typename R>
    bool operator()(const L& lhs, const R& rhs) const {
      return std::lexicographical_compare(lhs.begin(), lhs.end(), rhs.begin(), rhs.end());
    }
  };

  struct PathKmerEntry {
    KmerIdSet kmer_set_;
    Graph::PathIdSet intermediate_paths_;
    explicit PathKmerEntry(size_t kmer_count = 0) : kmer_set_(kmer_count) {}
  };
  using PathKmerMap = std::map<KmerNodeIdSeq, PathKmerEntry, NodeIdSeqLess>;
  
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
   * @param graph The variant graph (node IDs must be in topological order)
   * @param sequences K-mer sequences (must be in the same order as @p locations)
   * @param locations K-mer locations on the graph (must be in the same order as @p sequences)
   * @param params Scoring parameters (optional)
   */
  HaplotypeSamplerOverlay(const Graph& graph, const std::vector<std::string>& sequences,
                          const std::vector<std::vector<UniqueKmersOverlay::KmerLocation>>& locations, const Params& params);

  /**
   * @param graph The variant graph (node IDs must be in topological order)
   * @param unique_kmers Pre-computed graph-unique k-mer map (from Graph::UniqueKmers())
   * @param counts KmerClassify instance used to classify k-mer sequences
   * @param params Scoring parameters (optional)
   */
  explicit HaplotypeSamplerOverlay(const Graph& graph, const UniqueKmersOverlay& unique_kmers, const Params& params = {});

  /**
   * Construct with inference-VCF path filtering active.
   *
   * Sampled path will be restricted to those that differ at variants in @p inference_vcf whose
   * allele length change is >= @p min_size.
   */
  HaplotypeSamplerOverlay(const Graph& graph, const UniqueKmersOverlay& unique_kmers,
                          const std::string& inference_vcf,
                          const Range& region,
                          size_t min_size = 50,
                          const Params& params = {});

  /// Initialize (or reset) scores prior to sampling based on k-mer @p counts without reconstructing the overlay.
  void InitializeScores(const KmerClassify& counts);

  /// Return up to @p n unique highest-scoring distinct paths through the graph using the current k-mer scores.
  std::vector<Haplotype> FindBestPaths(size_t n) const;

  /// Return the top @p n haplotypes, sorted by descending score, sampled greedily from the graph using k-mer coverage.
  std::vector<Haplotype> SampleHaplotypes(size_t n);

  /// Return the top @p n highest-scoring diplotypes from all pairs of the @p candidate haplotypes, sorted by descending score.
  std::vector<Diplotype> SampleDiplotypes(const std::vector<Haplotype>& candidates, size_t n = 1) const;

  /// Return the variant_id-allele pairs a @p haplotype traversed by a haplotype or is corresponding @p covered_paths set
  std::vector<std::pair<std::string, size_t>> DecodeHaplotype(const Haplotype& haplotype) const;
  std::vector<std::pair<std::string, size_t>> DecodeHaplotype(const Graph::PathIdSet& covered_paths) const;

  /// Number of unique k-mers used in sampling
  size_t NumKmers() const { return kmers_.size(); }

  /// Return the set of k-mers that lie on @p path.
  KmerIdSet KmersOnPath(const Graph::NodeIdSeq& path) const;

  const PathKmerMap& PathKmers() const { return path_kmers_; }

 private:
 
  struct KmerScore {
    KmerZygosity zygosity;
    double score;
  };

  struct Backpointer {
    double score;
    odgi::nid_t pred_node;
    PathKmerMap::const_iterator edge_it;
    size_t pred_path_idx;
    Graph::PathIdSet covered_paths;
  };

  using BestPathState = std::vector<std::vector<Backpointer>>;
  using PathWithCoverage = std::pair<Haplotype, Graph::PathIdSet>;

  /// Compute BestPathState backpointers for up to @p n paths for the current scores
  BestPathState PropagateBestPathState(size_t n) const;

  /// Backtrack using BestPathState to return complete path with its covered_paths set
  PathWithCoverage BacktrackPath(const BestPathState& path_state, size_t back_idx) const;

  /// Update the scores of k-mers having sampled @p path
  void UpdateScores(const Graph::NodeIdSeq& path);

  const Graph& graph_;
  Params params_;

  Graph::NodeIdSet inference_node_mask_; ///< Nodes that differentiate inference alleles
  Graph::PathIdSet inference_path_mask_; ///< Paths for inference alleles
  bool apply_path_filter_ = false; ///< When true, skip paths with no covered inference alleles (set by inference-VCF constructor)

  std::vector<std::string> kmer_sequences_; ///< k-mer sequences
  std::vector<KmerScore> kmers_; ///< k-mer zygosity and score information (parallel to kmer_sequences_)
  
  /** 
   * Map from path (sequence of nids) to k-mer indices. We leverage sorted order to efficiently query for path segments
    * that have associated k-mers. This could potentially be replaced with a trie, if it becomes a performance bottleneck.
   */
  PathKmerMap path_kmers_; ///< path (sequence of nids) to k-mer indices
  static_assert(!std::is_same<typename PathKmerMap::key_compare, void>::value, "HaplotypeSamplerOverlay requires PathKmerMap to be ordered");

};

}
