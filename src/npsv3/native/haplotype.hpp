#include <algorithm>
#include <array>
#include <memory>
#include <unordered_map>
#include <vector>

#include <boost/dynamic_bitset.hpp>
#include <boost/intrusive_ptr.hpp>
#include <boost/smart_ptr/intrusive_ref_counter.hpp>

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

  /// Return up to @p n unique highest-scoring distinct paths through the graph using the current k-mer
  /// scores, via a single forward pass at exactly width @p n .
  std::vector<Haplotype> FindBestPaths(size_t n) const;

  /// Return the top @p n haplotypes, sorted by descending score, sampled greedily from the graph using k-mer coverage.
  std::vector<Haplotype> SampleHaplotypes(size_t n);

  /// Return the top @p n highest-scoring diplotypes from all pairs of the @p candidate haplotypes, sorted by descending score.
  std::vector<Diplotype> SampleDiplotypes(const std::vector<Haplotype>& candidates, size_t n = 1) const;

  /// Score @p haplotype under the current k-mer scores, i.e. the same score used to rank haplotypes while sampling.
  double Score(const Haplotype& haplotype) const;

  ///@{
  /// Return the variant_id-allele pairs traversed by a @p haplotype or its corresponding @p covered_paths set
  std::vector<std::pair<std::string, size_t>> DecodeHaplotype(const Haplotype& haplotype) const;
  std::vector<std::pair<std::string, size_t>> DecodeHaplotype(const Graph::PathIdSet& covered_paths) const;
  ///@}

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

  /// Purpose-implemented replacement "small" map optimized for nodes with small number of outgoing edges.
  /// The narrow case uses linear scan in a fixed inline array, while wide states fall back to an unordered
  /// map.
  ///
  /// Profiling indicated most non-root states have <= 2 goto_ entries, however attempts to use generic
  /// flat_map with small_vector regressed performance due to their generic indirection and binary search
  /// overhead. Exposed here to facilitate testing.
  class GotoMap {
   public:
    static constexpr size_t kInlineCapacity = 4;

    GotoMap() = default;
    GotoMap(GotoMap&&) = default;
    GotoMap& operator=(GotoMap&&) = default;
    // unique_ptr makes the implicit copy operations deleted; restore value semantics (deep-copying
    // overflow_ when present) so AutomatonState/HaplotypeSamplerOverlay remain copyable.
    GotoMap(const GotoMap& other);
    GotoMap& operator=(const GotoMap& other);

    /// Return a pointer to the child state for @p node_id, or nullptr if there is no such edge.
    const AutomatonIndex* find(odgi::nid_t node_id) const;

    /// Insert @p node_id -> @p child. Precondition: no edge for node_id already exists (matches trie
    /// construction, which only inserts after a failed find()).
    void emplace(odgi::nid_t node_id, AutomatonIndex child);

    size_t size() const;

    /// Visit every (node_id, child) edge. Construction-only use (BFS over goto_ to build fail links);
    /// not on the query-time hot path.
    template <typename Fn>
    void for_each(Fn&& fn) const;

   private:
    std::array<odgi::nid_t, kInlineCapacity> inline_keys_{};
    std::array<AutomatonIndex, kInlineCapacity> inline_values_{};
    size_t size_ = 0;
    std::unique_ptr<std::unordered_map<odgi::nid_t, AutomatonIndex>> overflow_;
  };

  /// A state in the Aho-Corasick automaton mapping k-mer locations (path snippets) to k-mer indices.
  /// Exposed for direct unit testing of goto/fail/output construction.
  struct AutomatonState {
    GotoMap goto_; ///< Trie edges: node id -> child state
    AutomatonIndex fail = kAutomatonRoot; ///< Failure link: state for the longest proper suffix of this state's path that is still a live prefix
    /// K-mer indices whose location ends exactly here, including via failure links, in ascending order.
    /// Stored as a flat sparse list rather than a per-state KmerIdSet as output sets are extremely sparse in
    /// practice (median 1 set bit, ~9 mean, measured on a dense 30k-k-mer/25k-state automaton). Empty for
    /// states with no pending/completing k-mer match.
    std::vector<uint32_t> output_kmer_indices_;
  };

  const std::vector<AutomatonState>& Automaton() const { return automaton_; }

 private:

  struct HaplotypeSamplerCommonCtor {};
  inline static constexpr HaplotypeSamplerCommonCtor kCommonCtor{};

  HaplotypeSamplerOverlay(HaplotypeSamplerCommonCtor, const Graph& graph, const std::vector<std::string>& sequences,
                          const std::vector<std::vector<UniqueKmersOverlay::KmerLocation>>& locations, const Params& params);

  struct KmerScore {
    KmerZygosity zygosity;
    double score;
  };

  /// Deterministic per-index finalizer (SplitMix64) used to build an order-independent hash of a covered_paths
  /// bit set as hash(S) = XOR over i in S of PathHash(i). 
  /// 
  /// Because XOR is commutative and this is a pure function
  /// of the *set* of indices, equal bit sets always hash equal regardless of how they were constructed. Thus
  /// the hash of a PathIdSetHolder can be updated incrementally (see the trusted-hash constructor below)
  /// instead of recomputed from scratch on every transition. This incremental use is only valid because
  /// covered_paths bits are monotonically added and never cleared (see accumulate_covered_paths).
  static size_t PathHash(size_t idx) {
    size_t x = idx + 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
  }

  /// Hash every set bit of `bits` from scratch via PathHash. Only needed where there's no predecessor hash to
  /// update incrementally from (e.g., the DP's seed node).
  static size_t HashPathIdSet(const Graph::PathIdSet& bits) {
    size_t h = 0;
    for (auto idx = bits.find_first(); idx != Graph::PathIdSet::npos; idx = bits.find_next(idx)) {
      h ^= PathHash(idx);
    }
    return h;
  }

  /// Immutable covered_paths bitset plus its hash, computed once when the set is finalized, to enable DP transitions
  /// that don't add any new path bits to share their predecessor's bitset via pointer copy (instead of a deep copy).
  ///
  /// Reference counted via boost::intrusive_ptr using a thread unsafe counter. We assume this will only be used
  /// in a single-threaded context (e.g., distinct Ray worker processes).
  struct PathIdSetHolder : public boost::intrusive_ref_counter<PathIdSetHolder, boost::thread_unsafe_counter> {
    Graph::PathIdSet bits;
    size_t hash;

    explicit PathIdSetHolder(Graph::PathIdSet b) : bits(std::move(b)), hash(HashPathIdSet(bits)) {}

    /// Trusted-hash constructor for incremental updates: caller must supply the hash of `b` themselves (see
    /// PathHash above), typically as predecessor_hash ^ (XOR of PathHash(i) for each bit i newly set vs. the
    /// predecessor). Only valid when `b`'s bits are a superset of whatever bits the caller's hash accounts for.
    PathIdSetHolder(Graph::PathIdSet b, size_t h) : bits(std::move(b)), hash(h) {}
  };
  using SharedPathIdSet = boost::intrusive_ptr<const PathIdSetHolder>;

  /// Backpointer for a single haplotype path at a settlement point (graph node + automaton state) in the 
  /// DP sampling algorithm.
  struct Backpointer {
    double score;
    odgi::nid_t pred_node;
    AutomatonIndex pred_automaton_state; ///< Automaton state pool at pred_node this entry backtracks into
    size_t pred_path_idx; ///< Index within that (pred_node, pred_automaton_state) pool
    SharedPathIdSet covered_paths; ///< Immutable; shared across backpointers
  };

  /// For sampling N haplotypes, maintain G x (AutomationIndex -> N x Backpointer), where G is number of graph nodes.
  using StatePool = std::vector<Backpointer>;
  using NodeState = std::unordered_map<AutomatonIndex, StatePool>;
  using BestPathState = std::vector<NodeState>;
  
  using PathWithCoverage = std::pair<Haplotype, Graph::PathIdSet>;

  /// Return the Aho-Corasick automaton state reached from @p state on @p node_id.
  AutomatonIndex AutomatonGoto(AutomatonIndex state, odgi::nid_t node_id) const;

  /// Compute BestPathState backpointers for up to @p n distinct-covered_paths paths per settlement point,
  /// for the current k-mer scores, via a single forward pass at exactly width @p n. Every (node,
  /// automaton_state) pool is trimmed to n candidates, including pending (mid k-mer-match) states.
  BestPathState PropagateBestPathState(size_t n) const;

  /// Extract the final top-n distinct paths (applying the path filter, if active) from a completed @p path_state.
  std::vector<Haplotype> ExtractBestPaths(const BestPathState& path_state) const;

  /// Return complete path using @p path_state starting at (max_node, @p back_automaton_state), at index @p back_idx
  PathWithCoverage BacktrackPath(const BestPathState& path_state, AutomatonIndex back_automaton_state,
                                 size_t back_idx) const;

  /// Return the additive score delta for every k-mer output at automaton state @p state
  double KmerSetScoreDelta(AutomatonIndex state) const;

  /// Update the scores of k-mers having sampled @p path
  void UpdateScores(const Graph::NodeIdSeq& path);

  const Graph& graph_;
  Params params_;

  Graph::NodeIdSet inference_node_mask_; ///< Nodes that differentiate inference alleles
  Graph::PathIdSet inference_path_mask_; ///< Paths for inference alleles
  bool apply_path_filter_ = false; ///< When true, skip paths with no covered inference alleles (set by inference-VCF constructor)

  std::vector<std::string> kmer_sequences_; ///< k-mer sequences
  std::vector<KmerScore> kmers_; ///< k-mer zygosity and score information (parallel to kmer_sequences_)

  /// Aho-Corasick automaton over every k-mer's recorded location, rooted at `automaton_[kAutomatonRoot]`
  std::vector<AutomatonState> automaton_;

  /// Node ids that appear *anywhere* in some k-mer's recorded location, i.e., the automaton's full symbol
  /// alphabet across every state. If a node_id is absent here, `AutomatonGoto(state, node_id) == kAutomatonRoot`
  /// for *every* state and we can skip the fail-chasing walk and its associated costs.
  Graph::NodeIdSet trie_symbol_mask_;

  /// Node ids whose arrival actually changes a path's covered_paths set, i.e. `graph_.node_variant_paths_[id]`
  /// (masked by `inference_path_mask_`/`inference_node_mask_` when `apply_path_filter_` is set) is non-empty.
  /// Computed once and then used for propagate `covered_paths` via cheap pointer copy when it doesn't change.
  Graph::NodeIdSet contributes_paths_mask_;

  void InitializeContributesPathsMask();
};

}
