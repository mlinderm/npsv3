#include <algorithm>
#include <array>
#include <limits>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

#include <boost/dynamic_bitset.hpp>
#include <boost/intrusive_ptr.hpp>
#include <boost/smart_ptr/intrusive_ref_counter.hpp>
#include <gbwt/gbwt.h>

#include "graph.hpp"
#include "kmer.hpp"

namespace npsv3 {

/**
 * @brief Overlay managing the population-prior "likely path state" over the background cohort's genotype paths already
 * materialized in a Graph:
 *
 *   Tier 1 (GBWT panel-consistency): The panel indexed as a GBWT (Sirén et al.), giving
 *     Find()/Extend() panel-consistency tracking plus a precomputed backward ScoreToGo() table over the
 *     induced state graph.
 *   Tier 2 (adjacent-distinguishing-node Markov chain): Observed counts of adjacent pairs of
 *     *distinguishing* nodes (nodes where node_variant_paths_[node_id] is non-empty), the backoff scored
 *     via TransitionProbability()/LogTransitionProbability() once a walk diverges from every panel member.
 *
 * A PanelState (v, [lo, hi)) denotes the contiguous range of background haplotypes consistent with a *specific walk* up
 * through node `v`. Width()/ScoreToGo() are precomputed only for states reachable by walking from start-of-haplotype
 * point (the region's entry node or a phase break) forward via Find() once, then Extend() for every subsequent node.
 *
 * Width() is the *deduplicated* panel-haplotype count for a state not raw GBWT interval size (separate paths for
 * phase-break segments are only counted once). ScoreToGo() is the offline backward max-recursion over log-width-ratios,
 * in unweighted units to provide a heuristic ranking aid for a beam search.
 *
 * For each background path the constructor walks its literal node sequence and accumulates counts of adjacent pairs of
 * *distinguishing* nodes, skipping non-distinguishing (plain shared reference) nodes in between. The transition table
 * maintains raw observed counts only. TransitionProbability()/LogTransitionProbability() do apply per-edge Laplace
 * smoothing, since that only needs data this class already owns.
 */
class HaplotypePriorOverlay {
 public:
  using PanelState = gbwt::SearchState;

  /// One observed (to_node, count) tier-2 transition, used as a sparse per-source adjacency-list entry.
  struct NodeTransition {
    odgi::nid_t to_node;
    uint32_t count;
  };

  /// @param excluded_samples Sample names (the '#'-delimited first component of Sample#HaplotypeIndex#Contig#SegmentIndex
  ///                          background paths) to omit entirely from both tiers, e.g. the sample being genotyped so
  ///                          its own truth haplotypes can't leak into its own prior.
  explicit HaplotypePriorOverlay(const Graph& graph, const std::vector<std::string>& excluded_samples = {});

  /// Return a search state starting from @p node, covering every panel haplotype that traverses that node, or an empty
  /// state if @p node is not part of any indexed path.
  ///
  /// Only use at a genuine walk start (the region's entry node, or after a phase-break), never mid-walk.
  PanelState Find(odgi::nid_t node) const;

  /// Return search state @p state extended by an edge to @p node or Empty if no such paths exist.
  PanelState Extend(const PanelState& state, odgi::nid_t node) const;

  /// Return True if @p state has no consistent panel members left.
  bool Empty(const PanelState& state) const { return state.empty(); }

  /// Deduplicated panel-haplotype count for @p state: distinct (sample, haplotype_index) pairs, not raw GBWT interval
  /// width. Returns 0 for the empty state and any state not reached by walking from a genuine walk start.
  size_t Width(const PanelState& state) const;

  /// Offline backward max-recursion score-to-go for @p state, in unweighted units as a heuristic ranking aid, not an
  /// exact bound. Returns 0.0 for the empty state, for any state with no further panel-consistent continuation
  /// (including the region's last node) and any state not reached from a genuine walk start.
  double ScoreToGo(const PanelState& state) const;


  /// CSR row offsets, dense-indexed by (node_id - graph.min_node_id()): transitions()[transition_starts()[i]
  /// .. transition_starts()[i+1]) are the observed (to_node, count) transitions from the node with id
  /// (graph.min_node_id() + i), sorted by to_node. Size == graph's node-id range + 1.
  const std::vector<size_t>& transition_starts() const { return transition_starts_; }

  /// Flat CSR transition entries, grouped by source node per transition_starts().
  const std::vector<NodeTransition>& transitions() const { return transitions_; }

  /// Total observed outgoing transitions from source node @p from_node (sum of that row's counts).
  /// Dense-indexed 1:1 with transition_starts() (excluding its trailing sentinel).
  const std::vector<uint32_t>& node_totals() const { return node_totals_; }

  /// Return P(to_node|from_node) expressed `(count(from,to)+alpha) / (node_totals[from]+alpha*K_to)`, where K_to is the
  /// number of distinct successor nodes observed from from_node. A `from_node` with zero observed successors (or not
  /// itself a distinguishing node at all) yields P=1.0 (no information, no penalty) rather than a cold-start branch.
  ///@{
  double TransitionProbability(odgi::nid_t from_node, odgi::nid_t to_node, double alpha) const;
  double LogTransitionProbability(odgi::nid_t from_node, odgi::nid_t to_node, double alpha) const;
  ///@}

  /// Serialize the overlay
  ///@{
  void Save(std::ostream& out) const;
  void Save(const std::string& path) const;
  ///@}

  /// Deserialize into a pre-allocated HaplotypePriorOverlay via placement new.
  ///@{
  static void Load(HaplotypePriorOverlay* target, const Graph& graph, std::istream& in);
  static void Load(HaplotypePriorOverlay* target, const Graph& graph, const std::string& path);
  ///@}
 private:
  // A dense-indexed (node, [lo,hi)) state key used for Width()/ScoreToGo() lookup. Functionally identical to PanelState,
  // but re-implemented to facilitate (de)serialization and decouple from GBWT library.
  struct StateKey {
    gbwt::node_type node;
    gbwt::size_type lo, hi;
    bool operator==(const StateKey& other) const {
      return node == other.node && lo == other.lo && hi == other.hi;
    }
  };
  struct StateKeyHash {
    size_t operator()(const StateKey& k) const noexcept;
  };
  static StateKey KeyOf(const PanelState& state) { return StateKey{state.node, state.range.first, state.range.second}; }

  // Used by Load() to construct from pre-built tables.
  HaplotypePriorOverlay(const Graph& graph, gbwt::GBWT index, std::vector<StateKey> state_keys,
                        std::unordered_map<StateKey, uint32_t, StateKeyHash> state_index, std::vector<uint32_t> width,
                        std::vector<double> score_to_go, std::vector<size_t> transition_starts,
                        std::vector<NodeTransition> transitions, std::vector<uint32_t> node_totals);

  // Constructor helpers, each a single pass over the background paths (see haplotype.cpp).
  void BuildTransitionTable(const std::unordered_set<std::string>& excluded_samples);
  void BuildGBWTIndex(const std::unordered_set<std::string>& excluded_samples);

  // Dense row index for from_node, or npos if from_node is out of the graph's node-id range.
  size_t RowIndex(odgi::nid_t from_node) const;

  const Graph& graph_;

  gbwt::GBWT index_;

  // Every non-empty state reachable by walking some panel haplotype's own node sequence via Find/Extend.
  // state_keys_/width_/score_to_go_ are parallel, dense-indexed arrays (the same dense index that is
  // state_index_'s mapped value); state_index_ is the O(1) average key -> dense-index lookup used by
  // Width()/ScoreToGo() at query time.
  std::vector<StateKey> state_keys_;
  std::unordered_map<StateKey, uint32_t, StateKeyHash> state_index_;
  std::vector<uint32_t> width_;
  std::vector<double> score_to_go_;

  std::vector<size_t> transition_starts_;
  std::vector<NodeTransition> transitions_;
  std::vector<uint32_t> node_totals_;
};

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

    double haplotype_prior_weight = 0.0; ///< Prior weight (0.0 disables the prior entirely).
    double panel_fallback_penalty = 0.0; ///< One-time cost, unweighted units, applied when walk diverges from panel haplotype 
    double transition_prior_alpha = 1.0; ///< Laplace smoothing for tier 2's P(to|from)
    size_t population_state_pool_cap = 8; ///< Pool search beam-width separate from sampling beam width

    constexpr Params() {}
  };


  /**
   * @param graph The variant graph (node IDs must be in topological order)
   * @param sequences K-mer sequences (must be in the same order as @p locations)
   * @param locations K-mer locations on the graph (must be in the same order as @p sequences)
   * @param prior Non-owning population-prior overlay (HAPLOTYPE_PRIOR_PROPOSAL.md §4); nullptr disables the
   *              prior entirely. Must outlive the sampler.
   * @param params Scoring hyperparameters (optional)
   */
  HaplotypeSamplerOverlay(const Graph& graph, const std::vector<std::string>& sequences,
                          const std::vector<std::vector<UniqueKmersOverlay::KmerLocation>>& locations,
                          const HaplotypePriorOverlay* prior = nullptr, const Params& params = {});

  /**
   * @param graph The variant graph (node IDs must be in topological order)
   * @param unique_kmers Pre-computed graph-unique k-mer map (from Graph::UniqueKmers())
   * @param prior Non-owning population-prior overlay (HAPLOTYPE_PRIOR_PROPOSAL.md §4); nullptr disables the
   *              prior entirely. Must outlive the sampler.
   * @param params Scoring hyperparameters (optional)
   */
  explicit HaplotypeSamplerOverlay(const Graph& graph, const UniqueKmersOverlay& unique_kmers,
                                   const HaplotypePriorOverlay* prior = nullptr, const Params& params = {});

  /**
   * Construct with inference-VCF path filtering active.
   *
   * Sampled path will be restricted to those that differ at variants in @p inference_vcf whose
   * allele length change is >= @p min_size.
   */
  HaplotypeSamplerOverlay(const Graph& graph, const UniqueKmersOverlay& unique_kmers, const std::string& inference_vcf,
                          const Range& region, size_t min_size = 50, const HaplotypePriorOverlay* prior = nullptr,
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
                          const std::vector<std::vector<UniqueKmersOverlay::KmerLocation>>& locations,
                          const HaplotypePriorOverlay* prior, const Params& params);

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

  /// Population-prior state carried per Backpointer. Tagged union of "still tracking full GBWT panel history" vs.
  /// "panel consistency exhausted, scoring via the adjacent-distinguishing-node Markov chain".
  ///
  /// last_distinguishing_node is tracked unconditionally to enable a "fallback" to the Markov chain at any point. When
  /// the population prior is disabled every Backpointer carries an identical default-constructed PopulationState (with
  /// no impact on deduplication/trimming during the beam search).
  struct PopulationState {
    struct Gbwt { HaplotypePriorOverlay::PanelState state{}; };  // Still tracking the panel
    struct FellBack {};                                          // Panel consistency exhausted
    std::variant<Gbwt, FellBack> tier{Gbwt{}};
    odgi::nid_t last_distinguishing_node = kNoNode; ///< Last graph node crossed with a non-empty
                                                     ///< node_variant_paths_ entry, tracked regardless.
    static constexpr odgi::nid_t kNoNode = -1; ///< odgi::nid_t is signed; real graph node ids are >= 1.
  };

  /// Backpointer for a single haplotype path at a settlement point (graph node + automaton state) in the
  /// DP sampling algorithm.
  struct Backpointer {
    double score;
    odgi::nid_t pred_node;
    AutomatonIndex pred_automaton_state; ///< Automaton state pool at pred_node this entry backtracks into
    size_t pred_path_idx; ///< Index within that (pred_node, pred_automaton_state) pool
    SharedPathIdSet covered_paths; ///< Immutable; shared across backpointers
    PopulationState population_state; ///< Population-prior tier + tracked context (see PopulationState)
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

  /// Apply one graph edge's population-prior transition to @p state in place returning the realized,
  /// already-haplotype_prior_weight-scaled score delta for that edge. @p next_node is the node being entered; @p
  /// distinguishes must be population_distinguishing_mask_.test(next_node) (already-hoisted by the caller, once per
  /// edge).
  double ApplyPopulationEdge(PopulationState& state, odgi::nid_t next_node, bool distinguishes) const;

  /// Heuristic score-to-go ranking for @p state's population-prior tail, composing full haplotype prior's with Markov
  /// Chain fallabck. Only used inside SortAndTrimBacktrack's rank-key. 0.0 when the prior isn't configured.
  double PopulationScoreToGo(const PopulationState& state) const;

  /// Tier 2's backward score-to-go recursion, keyed on distinguishing graph node id. Lazily built and memoized on first
  /// use into transition_score_to_go_ (a single backward sweep in decreasing node-id order). 0.0 if prior_ is null.
  double TransitionScoreToGo(odgi::nid_t from) const;

  void EnsureTransitionScoreToGo() const;

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
  
  /// Non-owning population-prior overlay. nullptr fully disables the prior, independent of
  /// params_.haplotype_prior_weight. Must outlive the sampler (see the binding keep_alive).
  const HaplotypePriorOverlay* prior_ = nullptr;

  /// "Distinguishing" node ids with a non-empty `graph_.node_variant_paths_[id]` without inference path filtering
  /// applied. The HaplotypePriorOverlay's transition table is built offline without any knowledge of inference paths.
  /// Used only for population-prior bookkeeping; covered_paths bookkeeping keeps using contributes_paths_mask_
  /// unchanged.
  Graph::NodeIdSet population_distinguishing_mask_;

  /// Lazily-built backward score-to-go table (TransitionScoreToGo), dense-indexed by (node_id - graph_.min_node_id());
  /// empty until first needed and prior_ is set.
  mutable std::vector<double> transition_score_to_go_;
  mutable bool transition_score_to_go_built_ = false;

  void InitializeContributesPathsMask();
  void InitializePopulationDistinguishingMask();
};

}
