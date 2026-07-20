#include <algorithm>
#include <unordered_map>
#include <vector>

#include <boost/dynamic_bitset.hpp>

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

  /// Score @p haplotype under the current k-mer scores, i.e. the same score used to rank haplotypes while sampling.
  double Score(const Haplotype& haplotype) const;

  /// Return the variant_id-allele pairs a @p haplotype traversed by a haplotype or is corresponding @p covered_paths set
  std::vector<std::pair<std::string, size_t>> DecodeHaplotype(const Haplotype& haplotype) const;
  std::vector<std::pair<std::string, size_t>> DecodeHaplotype(const Graph::PathIdSet& covered_paths) const;

  /// Number of unique k-mers used in sampling
  size_t NumKmers() const { return kmers_.size(); }

  /// Return the set of k-mers that lie on @p path
  KmerIdSet KmersOnPath(const Graph::NodeIdSeq& path) const;

  /// Current score for k-mer @p idx (as used in Score()/PropagateBestPathState).
  double KmerScoreAt(size_t idx) const { return kmers_[idx].score; }
  /// Sequence for k-mer @p idx.
  const std::string& KmerSequenceAt(size_t idx) const { return kmer_sequences_[idx]; }

  using AutomatonIndex = size_t;
  static constexpr size_t kAutomatonRoot = 0;

  /// A state in the Aho-Corasick automaton mapping k-mer locations (path snippets) to k-mer indices.
  /// Exposed for direct unit testing of goto/fail/output construction.
  struct AutomatonState {
    std::unordered_map<odgi::nid_t, AutomatonIndex> goto_; ///< Trie edges: node id -> child state
    AutomatonIndex fail = kAutomatonRoot; ///< Failure link: state for the longest proper suffix of this state's path that is still a live prefix
    KmerIdSet output_kmers_; ///< Every k-mer whose location ends exactly here, including via failure links
    explicit AutomatonState(size_t kmer_count = 0) : output_kmers_(kmer_count) {}
  };

  const std::vector<AutomatonState>& Automaton() const { return automaton_; }

 private:

  struct KmerScore {
    KmerZygosity zygosity;
    double score;
  };

  struct Backpointer {
    double score;
    odgi::nid_t pred_node;
    AutomatonIndex pred_automaton_state; ///< Automaton state pool at pred_node this entry backtracks into
    size_t pred_path_idx; ///< Index within that (pred_node, pred_automaton_state) pool
    Graph::PathIdSet covered_paths;
  };

  /// For haplotype sampling, maintain G x (AutomationIndex -> N x Backpointer), where G is number of graph nodes.
  using StatePool = std::vector<Backpointer>;
  using NodeState = std::unordered_map<AutomatonIndex, StatePool>;
  using BestPathState = std::vector<NodeState>;
  
  using PathWithCoverage = std::pair<Haplotype, Graph::PathIdSet>;
  
  /// Num nodes by Num automaton states table 
  using ScoreToGoTable = std::vector<std::vector<double>>;

  /// Return the Aho-Corasick automaton state reached from @p state on @p node_id.
  AutomatonIndex AutomatonGoto(AutomatonIndex state, odgi::nid_t node_id) const;

  /// Compute BestPathState backpointers for up to @p n distinct-covered_paths paths per settlement point,
  /// for the current k-mer scores.
  ///
  /// When @p score_to_go, a nodes by automaton states table of possible future scores, is non-null, every entry
  /// discarded at a settlement point updates @p max_escaped_bound with an admissible upper bound on what that
  /// discarded branch could have gone on to score.
  BestPathState PropagateBestPathState(size_t n, const std::vector<std::vector<double>>* score_to_go = nullptr,
                                       double* max_escaped_bound = nullptr) const;

  /// Compute BestPathState backpointers for up to @p n distinct-covered_paths paths per settlement point,
  /// for the current k-mer scores, using adaptive beam search (increasing @p n up to a factor of @p max_widening)
  /// to guarantee that the top scoring paths are returned.                                      
  BestPathState PropagateBestPathStateAdaptively(size_t n, size_t max_widening = 8) const;

  /// Compute best achievable *additional* score [v][s] from Graph node v with pending automaton state s
  /// to the sink for the current k-mer scores. Used as an admissible bound in adaptive beam search.
  ScoreToGoTable ComputeScoreToGo() const;

  /// Return complete path using @p path_state starting at (max_node, @p back_automaton_state), at index @p back_idx
  PathWithCoverage BacktrackPath(const BestPathState& path_state, AutomatonIndex back_automaton_state,
                                 size_t back_idx) const;

  /// Return the additive score delta for every k-mer in @p set
  double KmerSetScoreDelta(const KmerIdSet& set) const;

  /// Update the scores of k-mers having sampled @p path
  void UpdateScores(const Graph::NodeIdSeq& path);

  const Graph& graph_;
  Params params_;

  Graph::NodeIdSet inference_node_mask_; ///< Nodes that differentiate inference alleles
  Graph::PathIdSet inference_path_mask_; ///< Paths for inference alleles
  bool apply_path_filter_ = false; ///< When true, skip paths with no covered inference alleles (set by inference-VCF constructor)

  std::vector<std::string> kmer_sequences_; ///< k-mer sequences
  std::vector<KmerScore> kmers_; ///< k-mer zygosity and score information (parallel to kmer_sequences_)

  std::vector<AutomatonState> automaton_; ///< Aho-Corasick automaton over every k-mer's recorded location, rooted at automaton_[kAutomatonRoot]
};

}
