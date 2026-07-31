#include "haplotype.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <numeric>
#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include <boost/dynamic_bitset.hpp>
#include <fmt/std.h>
#include <fmt/ranges.h>

#include "variant.hpp"

namespace npsv3 {

namespace {
// Ad hoc, opt-in (NPSV3_HAPLOTYPE_PROFILE=1) timing breakdown for PropagateBestPathStateAdaptively, to separate
// the higher-level algorithmic cost drivers documented in PERFORMANCE_NOTES.md: (1) the per-sampled-haplotype
// repeat of the *entire* DP (SampleHaplotypes calls this once per haplotype), (2) adaptive-widening retries
// within a single call (each a full extra forward pass), (3) beam-width growth's effect on a single forward
// pass's cost, and (4) ComputeScoreToGo's fixed backward-sweep cost, paid once per call regardless of width.
// Negligible overhead when disabled (one getenv call, cached in a function-local static).
bool HaplotypeProfilingEnabled() {
  static const bool enabled = std::getenv("NPSV3_HAPLOTYPE_PROFILE") != nullptr;
  return enabled;
}

using ProfileClock = std::chrono::steady_clock;
double ElapsedMs(ProfileClock::time_point start) {
  return std::chrono::duration<double, std::milli>(ProfileClock::now() - start).count();
}

// Process-wide resident memory, sampled from /proc/self/status. rss_kb is the *current* resident set
// (drops when the allocator returns freed pages to the OS, which it often doesn't promptly -- so this can
// stay elevated after a discarded widening attempt's C++ objects are destroyed); hwm_kb ("high water mark")
// is the peak resident set since process start and is monotonically non-decreasing, i.e. the number that
// actually predicts OOM-kill risk. Linux-only (matches this project's documented environment); returns
// zeros if /proc/self/status is unavailable rather than failing the (opt-in, diagnostic-only) measurement.
struct RssSample {
  long rss_kb = 0;
  long hwm_kb = 0;
};

RssSample CurrentRss() {
  RssSample sample;
  std::ifstream status("/proc/self/status");
  std::string line;
  while (std::getline(status, line)) {
    if (line.compare(0, 6, "VmRSS:") == 0) {
      sample.rss_kb = std::strtol(line.c_str() + 6, nullptr, 10);
    } else if (line.compare(0, 6, "VmHWM:") == 0) {
      sample.hwm_kb = std::strtol(line.c_str() + 6, nullptr, 10);
    }
  }
  return sample;
}
}  // namespace

// -----------------------------------------------------------------------------------------------
// The Aho-Corasick automaton: structure and how the forward DP queries it
// -----------------------------------------------------------------------------------------------
//
// automaton_ is a trie over every k-mer's recorded node-id-sequence location ("key"), turned into a
// standard Aho-Corasick automaton so the forward DP (PropagateBestPathState)
// can find every key that occurs as a contiguous substring of a walked path, including keys that
// overlap each other, by consuming the path one real graph node at a time.
//
// Each AutomatonState holds:
//   goto_            trie edges: node id -> child state (only the *outgoing* edges actually inserted)
//   fail             the state for the longest proper suffix of this state's own path that is also
//                    some other state's path (root if none) -- lets matching resume without rescanning
//                    the text after a goto_ miss
//   output_kmer_indices_  every k-mer whose location ends exactly here, including via failure links --
//                    built during construction as own_kmer_set_[state] (k-mers whose location is *exactly*
//                    this state's path) unioned with the failure target's already-computed output set,
//                    then flattened to a sparse index list to improve performance and memory usage.
//
// AutomatonGoto(state, node_id) is the *effective* (fail-chasing) transition: follow goto_ if present,
// otherwise walk fail links until one has a goto_ for node_id, otherwise land on root. This is the
// classic non-completed Aho-Corasick query -- automaton_ only stores edges that were actually inserted,
// so goto_ is not "completed" into a total function ahead of time.
//
// The DP consumes a path by calling, once per real graph node it visits:
//     state = AutomatonGoto(state, node_id);
//     score += KmerSetScoreDelta(state);
// starting from state = kAutomatonRoot. Because output_kmer_indices_ already includes every key ending at
// that state (via failure links), this single call correctly credits *all* keys, short or long,
// nested or overlapping, which end at this node, without the DP ever needing to special-case a
// multi-node key as an atomic edge. KmersOnPath does the same walk over a complete path to compute the
// same set (used by Score/UpdateScores/SampleDiplotypes).
//
// Concrete example (this is exactly the fixture in OverlappingKeyTest, tests/native/test_haplotype.cpp):
// two keys that overlap by two nodes: P at [1,3,4] and R at [3,4,6] (node ids on a graph
// 1(prefix) -> {3(ref)|2(alt)} -> 4(mid) -> {6(ref)|5(alt)} -> 7(suffix)). Trie + failure links:
//
//         (root=0)
//        1/      \3
//       1          4  <- own={}
//      3/           \4
//     2               5  <- own={}
//    4/                 \6
//   3  <- own={P}         6  <- own={R}
//
//   state  path      goto_        fail    output_kmer_indices_
//   0      []        {1:1, 3:4}   -       {}
//   1      [1]       {3:2}        0       {}
//   4      [3]       {4:5}        0       {}
//   2      [1,3]     {4:3}        4       {}                 (fail: "3" is also state 4's path)
//   5      [3,4]     {6:6}        0       {}
//   3      [1,3,4]   {}           5       {P}                (fail: "3,4" is also state 5's path)
//   6      [3,4,6]   {}           0       {R}
//
// Walking the reference path [1,3,4,6,7] one node at a time from root (state 0):
//   node 1: goto_(0,1) = 1                              output={}         (nothing ends here)
//   node 3: goto_(1,3) = 2                               output={}         (P not complete yet)
//   node 4: goto_(2,4) = 3                               output={P}        (P completes: [1,3,4])
//   node 6: state 3 has no goto_ for 6, so fail-chase:
//           state 3 -> fail -> state 5; state 5 *does* have goto_(5,6) = 6
//           => AutomatonGoto(3, 6) = 6                   output={R}        (R completes: [3,4,6])
//   node 7: state 6 has no goto_ for 7; fail-chases to root; root has no goto_ for 7 either
//           => AutomatonGoto(6, 7) = 0                   output={}
//
// A single DP lineage walking node-by-node therefore credits *both* P (at node 4) and R (at node 6)
// on the same path even though R's key starts partway *through* P's key.

// -----------------------------------------------------------------------------------------------
// GotoMap: out-of-line implementation (declared in haplotype.hpp, see the class comment there)
// -----------------------------------------------------------------------------------------------

HaplotypeSamplerOverlay::GotoMap::GotoMap(const GotoMap& other)
    : inline_keys_(other.inline_keys_), inline_values_(other.inline_values_), size_(other.size_),
      overflow_(other.overflow_ ? std::make_unique<std::unordered_map<odgi::nid_t, AutomatonIndex>>(*other.overflow_)
                                 : nullptr) {}

HaplotypeSamplerOverlay::GotoMap& HaplotypeSamplerOverlay::GotoMap::operator=(const GotoMap& other) {
  if (this == &other) return *this;
  inline_keys_ = other.inline_keys_;
  inline_values_ = other.inline_values_;
  size_ = other.size_;
  overflow_ = other.overflow_ ? std::make_unique<std::unordered_map<odgi::nid_t, AutomatonIndex>>(*other.overflow_)
                               : nullptr;
  return *this;
}

const HaplotypeSamplerOverlay::AutomatonIndex* HaplotypeSamplerOverlay::GotoMap::find(odgi::nid_t node_id) const {
  if (overflow_) {
    auto it = overflow_->find(node_id);
    return it != overflow_->end() ? &it->second : nullptr;
  }
  for (size_t i = 0; i < size_; ++i) {
    if (inline_keys_[i] == node_id) return &inline_values_[i];
  }
  return nullptr;
}

void HaplotypeSamplerOverlay::GotoMap::emplace(odgi::nid_t node_id, AutomatonIndex child) {
  if (overflow_) {
    overflow_->emplace(node_id, child);
    return;
  }
  if (size_ < kInlineCapacity) {
    inline_keys_[size_] = node_id;
    inline_values_[size_] = child;
    ++size_;
    return;
  }
  overflow_ = std::make_unique<std::unordered_map<odgi::nid_t, AutomatonIndex>>();
  for (size_t i = 0; i < size_; ++i) overflow_->emplace(inline_keys_[i], inline_values_[i]);
  overflow_->emplace(node_id, child);
}

size_t HaplotypeSamplerOverlay::GotoMap::size() const { return overflow_ ? overflow_->size() : size_; }

template <typename Fn>
void HaplotypeSamplerOverlay::GotoMap::for_each(Fn&& fn) const {
  if (overflow_) {
    for (const auto& [node_id, child] : *overflow_) fn(node_id, child);
  } else {
    for (size_t i = 0; i < size_; ++i) fn(inline_keys_[i], inline_values_[i]);
  }
}

HaplotypeSamplerOverlay::HaplotypeSamplerOverlay(
    HaplotypeSamplerCommonCtor,
    const Graph& graph, const std::vector<std::string>& sequences,
    const std::vector<std::vector<UniqueKmersOverlay::KmerLocation>>& locations, const Params& params)
    : graph_(graph), params_(params), apply_path_filter_(false), kmer_sequences_(sequences) {
  // Maintain internal kmers_ in the same order as unique_kmers for consistent indexing.
  const size_t num_kmers = sequences.size();

  kmers_.reserve(num_kmers);
  for (size_t kmer_idx = 0; kmer_idx < num_kmers; ++kmer_idx) {
    kmers_.push_back({ KmerZygosity::ABSENT, params_.absent_score }); // C++20 required for parenthesized initialization in emplace_back
  }

  // Build the Aho-Corasick automaton directly from each k-mer's own recorded location(s) to enable matching
  // k-mers that span multiple handles.
  automaton_.emplace_back(); // root = kAutomatonRoot (0)
  trie_symbol_mask_.resize(graph_.max_node_id() + 1);

  // k-mers whose location is *exactly* this state's path. Made sparse since not all states are terminal for some k-mer.
  std::unordered_map<size_t, KmerIdSet> own_kmer_set;

  // Build the trie: One path from root per k-mer location, recording which k-mer indices share that
  // exact location.
  for (size_t kmer_idx = 0; kmer_idx < num_kmers; ++kmer_idx) {
    for (const auto& [handles, offset] : locations[kmer_idx]) {
      size_t state = kAutomatonRoot;
      for (const auto& handle : handles) {
        auto node_id = graph_.get_id(handle);
        trie_symbol_mask_.set(node_id);
        const auto* child_ptr = automaton_[state].goto_.find(node_id);
        size_t child;
        if (!child_ptr) {
          child = automaton_.size();
          automaton_.emplace_back(); // May reallocate automaton_; index (not reference) into it afterward
          automaton_[state].goto_.emplace(node_id, child);
        } else {
          child = *child_ptr;
        }
        state = child;
      }
      own_kmer_set.try_emplace(state, num_kmers).first->second.set(kmer_idx);
    }
  }

  // BFS to compute failure links and output_kmer_indices_ (standard Aho-Corasick construction, adapted to
  // store each state's output set as a sparse index list rather than a dense per-state bitset). A state's
  // full output set is the union of its failure target's (already-finalized) output set and its own k-mers.
  // We build this as a scratch bitset then convert to sparse indices. BFS visits states in non-decreasing
  // depth order, and a state's failure link always points to a strictly shallower state, so each state's
  // failure target's full_output entry is already finalized when needed.
  std::vector<KmerIdSet> full_output(automaton_.size(), KmerIdSet(num_kmers));
  std::queue<size_t> to_visit;
  automaton_[kAutomatonRoot].goto_.for_each([&](odgi::nid_t /*node_id*/, size_t child) {
    automaton_[child].fail = kAutomatonRoot;
    to_visit.push(child);
  });
  while (!to_visit.empty()) {
    size_t state = to_visit.front();
    to_visit.pop();

    KmerIdSet& state_output = full_output[state];
    state_output = full_output[automaton_[state].fail];
    if (auto it = own_kmer_set.find(state); it != own_kmer_set.end()) {
      state_output |= it->second;
    }
    auto& indices = automaton_[state].output_kmer_indices_;
    for (size_t kmer_idx = state_output.find_first(); kmer_idx != KmerIdSet::npos; kmer_idx = state_output.find_next(kmer_idx)) {
      indices.push_back(static_cast<uint32_t>(kmer_idx));
    }

    // Standard construction: fail[child] = goto*(fail[state], node_id) is the state reached by taking
    // the same node_id edge from state's own failure state, using the identical fail-chasing lookup
    // the DP uses at query time (AutomatonGoto). Safe to call here: goto_ is fully built for every
    // state before this BFS starts, and fail[state] always points to a strictly shallower state, whose
    // own .fail chain BFS has already been finalized.
    automaton_[state].goto_.for_each([&](odgi::nid_t node_id, size_t child) {
      automaton_[child].fail = AutomatonGoto(automaton_[state].fail, node_id);
      to_visit.push(child);
    });
  }
}

HaplotypeSamplerOverlay::HaplotypeSamplerOverlay(
    const Graph& graph, const std::vector<std::string>& sequences,
    const std::vector<std::vector<UniqueKmersOverlay::KmerLocation>>& locations, const Params& params)
    : HaplotypeSamplerOverlay(kCommonCtor, graph, sequences, locations, params) {
  InitializeContributesPathsMask();
}

HaplotypeSamplerOverlay::HaplotypeSamplerOverlay(const Graph& graph, const UniqueKmersOverlay& unique_kmers,
                                                 const Params& params)
    : HaplotypeSamplerOverlay(kCommonCtor, graph, unique_kmers.sequences(), unique_kmers.locations(), params) {
  InitializeContributesPathsMask();
}

HaplotypeSamplerOverlay::HaplotypeSamplerOverlay(const Graph& graph, const UniqueKmersOverlay& unique_kmers,
                                                 const std::string& inference_vcf, const Range& region,
                                                 size_t min_size, const Params& params)
    : HaplotypeSamplerOverlay(kCommonCtor, graph, unique_kmers.sequences(), unique_kmers.locations(), params) {
  // Initialize inference VCF filtering
  apply_path_filter_ = true;
  graph.PopulateNodeAndPathMasks(inference_vcf, region, min_size, inference_node_mask_, inference_path_mask_);
  assert(inference_path_mask_.any());

  // Initialize the contributes_paths_mask_ based on the inference masks, so that we can do fast
  // propagation of covered_paths when the current node doesn't contribute any new paths.
  InitializeContributesPathsMask();
}

void HaplotypeSamplerOverlay::InitializeContributesPathsMask() {
  const odgi::nid_t min_id = graph_.min_node_id();
  const odgi::nid_t max_id = graph_.max_node_id();

  contributes_paths_mask_.clear();
  contributes_paths_mask_.resize(max_id + 1);
  for (odgi::nid_t i = min_id; i <= max_id; ++i) {
    if (!graph_.has_node(i)) continue;
    // Mirrors accumulate_covered_paths's condition in PropagateBestPathState exactly: a node "contributes"
    // iff that lambda would actually set a new bit for it.
    if (!apply_path_filter_) {
      if (graph_.node_variant_paths_[i].any()) contributes_paths_mask_.set(i);
    } else if (inference_node_mask_.test(i) && (graph_.node_variant_paths_[i] & inference_path_mask_).any()) {
      contributes_paths_mask_.set(i);
    }
  }
}

void HaplotypeSamplerOverlay::InitializeScores(const KmerClassify& counts) {
  // Re-classify k-mers to reset scores based on current parameters. FREQUENT or otherwise unknown k-mers
  // are set to 0 (a neutral score) and not updated during sampling.
  counts.ClassifySorted(kmer_sequences_, [&](size_t idx, KmerZygosity zyg) {
    double initial_score = 0.0; // A neutral score for unknown k-mers, which are effectively ignored during sampling
    switch (zyg) {
      default: // FREQUENT or unknown k-mers receive neutral score
        break;
      case KmerZygosity::HOMOZYGOUS:
        initial_score = params_.homozygous_score;
        break;
      case KmerZygosity::HETEROZYGOUS:
        initial_score = params_.heterozygous_score;
        break;
      case KmerZygosity::ABSENT:
        initial_score = params_.absent_score;
        break;
    }
    kmers_[idx] = { zyg, initial_score };
  });
}

namespace {
  // Unconditionally merge exact covered_paths duplicates within backtrack (keep the highest-scoring
  // representative per class)
  template<typename T>
  void DedupCoveredPaths(T& backtrack) {
    if (backtrack.empty()) return;
    // covered_paths is an immutable, refcounted PathIdSetHolder to enable fast "shallow" copies when
    // there are no changes. First check for pointer equality, then hash equality and then bits. Only
    // entries with *different* holders fall back to comparing ->hash and, on a hash tie, ->bits (potentially
    // thousands of bits). Since equal covered_paths always hash equal, this can never merge or split groups
    // differently than comparing covered_paths directly.
    std::sort(backtrack.begin(), backtrack.end(), [](const auto& a, const auto& b) {
      if (a.covered_paths == b.covered_paths) return a.score > b.score;
      if (a.covered_paths->hash != b.covered_paths->hash) return a.covered_paths->hash > b.covered_paths->hash;
      // Group by covered_paths first, then sort by descending score within groups
      if (a.covered_paths->bits == b.covered_paths->bits) {
        return a.score > b.score;
      }
      return a.covered_paths->bits > b.covered_paths->bits;
    });
    // Retain only the highest-scoring representative per inference equivalence class.
    auto last = std::unique(backtrack.begin(), backtrack.end(), [](const auto& a, const auto& b) {
      return a.covered_paths == b.covered_paths ||
             (a.covered_paths->hash == b.covered_paths->hash && a.covered_paths->bits == b.covered_paths->bits);
    });
    backtrack.resize(std::distance(backtrack.begin(), last));
  }

  /// Apply the settlement-point rule. 
  ///
  /// At the automaton root (or the final sink, where no k-mer match can still be pending), it's safe to collapse
  /// to the top @p n *distinct* covered_paths classes. At any other (pending-match) automaton state, collapsing
  /// to @p n classes could discard a class that would have gone on to have a better score when its pending match
  /// resolves, so only exact-duplicate merging (node, automaton_state, covered_paths) is applied there.
  ///
  /// When @p max_discarded_score is non-null and this trim actually discards entries (settled and
  /// backtrack.size() > n), it is updated to the highest score among the discarded entries -- the caller
  /// combines this with an admissible score_to_go bound to certify whether widening @p n could still matter.
  template <typename T>
  void SortAndTrimBacktrack(T& backtrack, size_t n, bool settled = true, double* max_discarded_score = nullptr) {
    DedupCoveredPaths(backtrack);
    if (!settled) return;

    size_t new_size = std::min(n, backtrack.size());
    std::partial_sort(backtrack.begin(), backtrack.begin() + new_size, backtrack.end(),
                      [](const auto& a, const auto& b) { return a.score > b.score; });
    if (max_discarded_score && backtrack.size() > new_size) {
      auto discarded_best = std::max_element(backtrack.begin() + new_size, backtrack.end(),
                                             [](const auto& a, const auto& b) { return a.score < b.score; });
      *max_discarded_score = std::max(*max_discarded_score, discarded_best->score);
    }
    backtrack.resize(new_size);
  }
}

size_t HaplotypeSamplerOverlay::AutomatonGoto(size_t state, odgi::nid_t node_id) const {
  // node_id never appears in any k-mer's recorded location, at any depth of any state's goto_/fail chain,
  // so every goto_.find(node_id) below is guaranteed to miss regardless of starting state. Skip straight
  // to the root, the only possible result, instead of walking the fail chain to discover the same result.
  if (!trie_symbol_mask_.test(node_id)) return kAutomatonRoot;
  while (state != kAutomatonRoot) {
    if (const auto* child = automaton_[state].goto_.find(node_id)) return *child;
    state = automaton_[state].fail;
  }
  const auto* child = automaton_[kAutomatonRoot].goto_.find(node_id);
  return child ? *child : kAutomatonRoot;
}

double HaplotypeSamplerOverlay::KmerSetScoreDelta(AutomatonIndex state) const {
  // Empty for the (overwhelming majority of) states with no pending/completing k-mer match.
  double total = 0.0;
  for (auto kmer_idx : automaton_[state].output_kmer_indices_) {
    total += kmers_[kmer_idx].score;
  }
  return 2.0 * total;
}

std::vector<HaplotypeSamplerOverlay::Haplotype> HaplotypeSamplerOverlay::ExtractBestPaths(const BestPathState& path_state) const {
  const auto & best_paths = path_state.back().at(kAutomatonRoot); // Sink pools are always merged under this one key

  std::vector<Haplotype> result;
  result.reserve(best_paths.size());
  for (size_t back_idx = 0; back_idx < best_paths.size(); ++back_idx) {
    if (apply_path_filter_ && best_paths[back_idx].covered_paths->bits.none()) {
      continue; // Skip paths that do not cover any inference paths when filtering is active
    }
    result.push_back(std::move(BacktrackPath(path_state, kAutomatonRoot, back_idx).first));
  }
  return result;
}

std::vector<HaplotypeSamplerOverlay::Haplotype> HaplotypeSamplerOverlay::FindBestPaths(size_t n) const {
  return FindBestPaths(n, nullptr);
}

std::vector<HaplotypeSamplerOverlay::Haplotype> HaplotypeSamplerOverlay::FindBestPaths(size_t n, size_t* attempts_used) const {
  auto result = ExtractBestPaths(PropagateBestPathStateAdaptively(n, /*max_widening=*/8, attempts_used));
  // The adaptive loop's settled width can exceed n (widening enlarges the beam to *certify* the top n,
  // not to deliberately return more); trim back to the documented "up to n" contract. Safe because
  // ExtractBestPaths' backing pool is already sorted by descending score, so this keeps the best n.
  if (result.size() > n) result.resize(n);
  return result;
}

std::vector<HaplotypeSamplerOverlay::Haplotype> HaplotypeSamplerOverlay::FindBestPathsFixedWidth(size_t n) const {
  return ExtractBestPaths(PropagateBestPathState(n));
}

HaplotypeSamplerOverlay::BestPathState HaplotypeSamplerOverlay::PropagateBestPathState(
    size_t n, const std::vector<std::vector<double>>* score_to_go, double* max_escaped_bound) const {
  const odgi::nid_t min_id = graph_.min_node_id();
  const odgi::nid_t max_id = graph_.max_node_id();
  const size_t covered_paths_size = graph_.node_variant_paths_[min_id].size(); // All nodes should have the same size path sets

  // dp[v - min_id][s] holds up to n backpointer entries for distinct paths from min_id to v that are
  // currently pending automaton state s upon arrival at v. pred_path_idx indexes into
  // dp[pred_node - min_id][pred_automaton_state], which is frozen before propagation so indices stay
  // stable.
  BestPathState dp(max_id - min_id + 1);
  if (n == 0) {
    dp[max_id - min_id][kAutomatonRoot]; // Ensure the sink always has a (possibly empty) root pool
    return dp;
  }

  // Minimum possible score is if none of the k-mers are present in the haplotype, i.e., pH(x) = -1
  double min_score = std::accumulate(kmers_.begin(), kmers_.end(), 0.0, [](double acc, const KmerScore& kmer) {
    return acc - kmer.score;
  });

  auto accumulate_covered_paths = [&](Graph::PathIdSet& covered_paths, odgi::nid_t node_id) {
    if (!apply_path_filter_) {
      covered_paths |= graph_.node_variant_paths_[node_id];
    } else if (inference_node_mask_.test(node_id)) {
      covered_paths |= (graph_.node_variant_paths_[node_id] & inference_path_mask_);
    }
  };
  auto make_covered_paths = [](Graph::PathIdSet bits) -> SharedPathIdSet {
    return SharedPathIdSet(new PathIdSetHolder(std::move(bits)));
  };

  // Seed the source node: a virtual transition from the automaton's root consuming min_id itself, exactly
  // mirroring how every other node's arrival is processed below.
  {
    size_t seed_state = AutomatonGoto(kAutomatonRoot, min_id);
    Graph::PathIdSet seed_covered_paths(covered_paths_size);
    accumulate_covered_paths(seed_covered_paths, min_id);
    dp[0][seed_state].push_back({
      min_score + KmerSetScoreDelta(seed_state),
      0,  // no predecessor node
      kAutomatonRoot, // irrelevant without predecessor node
      0, // irrelevant without predecessor node
      make_covered_paths(std::move(seed_covered_paths))
    });
  }

  for (odgi::nid_t i = min_id; i <= max_id; ++i) {
    if (!graph_.has_node(i)) continue;

    auto& node_state = dp[i - min_id];
    assert(!node_state.empty());  // Should have at least one path to every reachable node

    // At the automaton root (or the final sink, where no k-mer match can still be pending), it's safe to collapse
    // to the top n *distinct* covered_paths classes. At any other (pending-match) automaton state, collapsing
    // to n classes could discard a class that would have gone on to have a better score when its pending match
    // resolves, so only exact-duplicate merging (node, automaton_state, covered_paths) is applied there.    
    for (auto& [automaton_state, pool] : node_state) {
      double discarded = -std::numeric_limits<double>::infinity();
      SortAndTrimBacktrack(pool, n, /*settled=*/automaton_state == kAutomatonRoot,
                            score_to_go ? &discarded : nullptr);
      if (score_to_go && std::isfinite(discarded)) {
        *max_escaped_bound = std::max(*max_escaped_bound, discarded + (*score_to_go)[i - min_id][automaton_state]);
      }
    }

    // Propagate along every real forward graph edge
    graph_.follow_edges(graph_.get_handle(i), false /* forward */, [&](const handlegraph::handle_t& next) {
      auto next_node = graph_.get_id(next);
      auto& next_node_state = dp[next_node - min_id];
      // contributes_paths_mask_ is a static property of next_node (independent of any predecessor's
      // covered_paths), so it's safe to test once per edge rather than once per (automaton_state, b_idx).
      // When false, no b_idx below can possibly gain a new path bit at next_node -- share the predecessor's
      // PathIdSetHolder (cheap refcount bump, reuses its cached hash) instead of copying, et al.
      const bool contributes = contributes_paths_mask_.test(next_node);

      for (auto& [automaton_state, pool] : node_state) {
        size_t new_state = AutomatonGoto(automaton_state, next_node);
        double weight_delta = KmerSetScoreDelta(new_state);
        auto& next_pool = next_node_state[new_state];
        for (size_t b_idx = 0; b_idx < pool.size(); ++b_idx) {
          SharedPathIdSet new_covered_paths;
          if (contributes) {
            Graph::PathIdSet bits = pool[b_idx].covered_paths->bits;
            accumulate_covered_paths(bits, next_node);
            new_covered_paths = make_covered_paths(std::move(bits));
          } else {
            new_covered_paths = pool[b_idx].covered_paths; // unchanged -- share, don't copy
          }
          next_pool.push_back({
            pool[b_idx].score + weight_delta,
            i, // predecessor node
            automaton_state, // predecessor automaton state
            b_idx, // path index in predecessor node's pool
            std::move(new_covered_paths)
          });
        }
      }
      return true;
    });
  }

  // At the sink, no k-mer match can ever resolve further, so the pending automaton state does not carry any
  // forward-looking information. Merge every automaton-state pool into one, deduplicating by covered_paths, and
  // trim to the final top-n, i.e., same "settlement" treatment as a root state.
  auto& sink_state = dp[max_id - min_id];
  StatePool merged;
  for (auto& [automaton_state, pool] : sink_state) {
    merged.insert(merged.end(), std::make_move_iterator(pool.begin()), std::make_move_iterator(pool.end()));
  }
  {
    double discarded = -std::numeric_limits<double>::infinity();
    SortAndTrimBacktrack(merged, n, /*settled=*/true, score_to_go ? &discarded : nullptr);
    // No k-mer match can complete after the sink, so score_to_go is 0 there regardless of automaton state.
    if (score_to_go && std::isfinite(discarded)) {
      *max_escaped_bound = std::max(*max_escaped_bound, discarded);
    }
  }
  sink_state.clear();
  sink_state.emplace(kAutomatonRoot, std::move(merged));

  return dp;
}

HaplotypeSamplerOverlay::ScoreToGoTable HaplotypeSamplerOverlay::ComputeScoreToGo() const {
  const odgi::nid_t min_id = graph_.min_node_id();
  const odgi::nid_t max_id = graph_.max_node_id();
  const size_t num_states = automaton_.size();

  // score_to_go[v - min_id][s] mirrors PropagateBestPathState's forward transitions exactly, but
  // backward: since node ids are topologically sorted, every real edge / automaton transition only ever
  // points to a strictly higher node id, so a single reverse pass over node id suffices.
  ScoreToGoTable score_to_go(max_id - min_id + 1, std::vector<double>(num_states, 0.0));

  for (odgi::nid_t i = max_id - 1; i >= min_id; --i) {
    if (!graph_.has_node(i)) continue;

    auto& row = score_to_go[i - min_id];
    std::fill(row.begin(), row.end(), -std::numeric_limits<double>::infinity());

    graph_.follow_edges(graph_.get_handle(i), false /* forward */, [&](const handlegraph::handle_t& next) {
      auto next_node = graph_.get_id(next);
      const auto& next_row = score_to_go[next_node - min_id];
      if (!trie_symbol_mask_.test(next_node)) {
        // next_node never appears in any k-mer location, so AutomatonGoto(s, next_node) == kAutomatonRoot
        // (and KmerSetScoreDelta(kAutomatonRoot) == 0.0, since output_kmer_indices_ is always empty at the root)
        // for *every* state s -- every row[s] is max'd with the identical constant. Skip the per-state
        // AutomatonGoto/KmerSetScoreDelta calls (the dominant cost of this function, since it otherwise
        // runs them for every state on every edge) in favor of a plain branchless max sweep.
        double val = next_row[kAutomatonRoot];
        for (size_t s = 0; s < num_states; ++s) row[s] = std::max(row[s], val);
        return true;
      }
      for (size_t s = 0; s < num_states; ++s) {
        size_t new_state = AutomatonGoto(s, next_node);
        double weight_delta = KmerSetScoreDelta(new_state);
        row[s] = std::max(row[s], weight_delta + next_row[new_state]);
      }
      return true;
    });
  }

  return score_to_go;
}

HaplotypeSamplerOverlay::BestPathState HaplotypeSamplerOverlay::PropagateBestPathStateAdaptively(size_t n, size_t max_widening, size_t* attempts_used) const {
  const bool profiling = HaplotypeProfilingEnabled();
  static size_t call_index = 0;  // single-threaded per PERFORMANCE_NOTES.md's existing thread-unsafe-counter assumption
  size_t this_call = profiling ? call_index++ : 0;
  auto call_start = ProfileClock::now();
  RssSample call_start_rss = profiling ? CurrentRss() : RssSample{};

  if (profiling) {
    const odgi::nid_t min_id = graph_.min_node_id();
    const odgi::nid_t max_id = graph_.max_node_id();
    const size_t num_graph_nodes = max_id - min_id + 1;
    const size_t num_automaton_states = automaton_.size();
    const size_t covered_paths_bits = graph_.node_variant_paths_[min_id].size();
    // ScoreToGoTable is vector<vector<double>> (haplotype.hpp): num_graph_nodes separate heap allocations
    // (one per row), each holding num_automaton_states doubles. Payload bytes below is just the doubles;
    // it excludes each row's own std::vector control-block + allocator bookkeeping, which a single flat
    // allocation would avoid entirely (num_graph_nodes-1 fewer allocations).
    const size_t score_to_go_payload_bytes = num_graph_nodes * num_automaton_states * sizeof(double);
    // A single PathIdSetHolder's boost::dynamic_bitset block storage alone (excludes the holder's own
    // hash/refcount/dynamic_bitset-object overhead) -- one of these is allocated per DP transition where
    // contributes_paths_mask_ is set (see PropagateBestPathState), shared thereafter via intrusive_ptr.
    const size_t covered_paths_bitset_bytes = ((covered_paths_bits + 63) / 64) * 8;
    fmt::print(stderr,
               "HAP_PROFILE call={} n={} stage=sizes graph_nodes={} automaton_states={} "
               "score_to_go_payload_bytes={} covered_paths_bits={} covered_paths_bitset_bytes={} "
               "rss_kb={} hwm_kb={}\n",
               this_call, n, num_graph_nodes, num_automaton_states, score_to_go_payload_bytes,
               covered_paths_bits, covered_paths_bitset_bytes, call_start_rss.rss_kb, call_start_rss.hwm_kb);
  }

  auto score_to_go_start = ProfileClock::now();
  auto score_to_go = ComputeScoreToGo();
  if (profiling) {
    RssSample rss = CurrentRss();
    fmt::print(stderr, "HAP_PROFILE call={} n={} stage=score_to_go ms={:.3f} rss_kb={} hwm_kb={} rss_delta_kb={}\n",
               this_call, n, ElapsedMs(score_to_go_start), rss.rss_kb, rss.hwm_kb, rss.rss_kb - call_start_rss.rss_kb);
  }

  // Adaptive beam search. Start by maintaining n backpointers, then double with the width until the top scoring paths
  // are guaranteed to be included. max_escaped_bound reports an admissible upper bound on what a discarded branch
  // could still have scored. If the weakest_kept_score is >= that bound, no discarded branch could have beaten the
  // worst kept result, so the top-n set must contain the true top-n and the search can stop. Otherwise, some discarded
  // branch *might* have beaten the worst kept result, so the width is doubled and the search is redone from scratch at
  // the wider beam.
  for (size_t width = n, attempt = 0; ; width *= 2, ++attempt) {
    auto forward_start = ProfileClock::now();
    double max_escaped_bound = -std::numeric_limits<double>::infinity();
    auto path_state = PropagateBestPathState(width, &score_to_go, &max_escaped_bound);

    const auto& results = path_state.back().at(kAutomatonRoot);
    double weakest_kept_score = results.empty() ? -std::numeric_limits<double>::infinity() : results.back().score;
    bool settled = weakest_kept_score >= max_escaped_bound || width > max_widening * std::max<size_t>(n, 1);
    if (profiling) {
      // Sampled with path_state (this attempt's full BestPathState, i.e. its backpointer pools/PathIdSetHolders)
      // still alive: reflects this attempt's own footprint. Any *previous* attempt's path_state was already
      // destroyed when that loop iteration's scope ended, before this attempt's PropagateBestPathState call --
      // so rss_delta_kb isolates this attempt, but if the allocator doesn't return freed pages to the OS
      // promptly, rss_kb/hwm_kb can still show elevated (non-dropping) memory carried over from earlier,
      // already-destroyed attempts in this same call.
      RssSample rss = CurrentRss();
      fmt::print(stderr,
                 "HAP_PROFILE call={} n={} stage=forward attempt={} width={} ms={:.3f} settled={} weakest={:.4f} bound={:.4f} "
                 "rss_kb={} hwm_kb={} rss_delta_kb={}\n",
                 this_call, n, attempt, width, ElapsedMs(forward_start), settled ? 1 : 0, weakest_kept_score, max_escaped_bound,
                 rss.rss_kb, rss.hwm_kb, rss.rss_kb - call_start_rss.rss_kb);
    }
    if (settled) {
      if (attempts_used) *attempts_used = attempt + 1;
      if (profiling) {
        RssSample rss = CurrentRss();
        fmt::print(stderr, "HAP_PROFILE call={} n={} stage=total ms={:.3f} final_width={} attempts={} rss_kb={} hwm_kb={}\n",
                   this_call, n, ElapsedMs(call_start), width, attempt + 1, rss.rss_kb, rss.hwm_kb);
      }
      return path_state;
    }
  }
}

HaplotypeSamplerOverlay::KmerIdSet HaplotypeSamplerOverlay::KmersOnPath(const Haplotype& path) const {
  // Walk the automaton one node at a time for path accumulating output_kmer_indices_, the unique k-mers
  // for sub-path match ending at that node.
  KmerIdSet on_path(kmers_.size());
  size_t state = kAutomatonRoot;
  for (auto node_id : path) {
    state = AutomatonGoto(state, node_id);
    for (auto kmer_idx : automaton_[state].output_kmer_indices_) on_path.set(kmer_idx);
  }
  return on_path;
}

double HaplotypeSamplerOverlay::Score(const Haplotype& haplotype) const {
  // Mirrors the DP scoring in PropagateBestPathState: each k-mer contributes its (signed) current
  // score if it lies on the haplotype, or its negation otherwise.
  auto on_path = KmersOnPath(haplotype);

  double score = 0.0;
  for (size_t i = 0; i < kmers_.size(); ++i) {
    score += on_path.test(i) ? kmers_[i].score : -kmers_[i].score;
  }
  return score;
}

void HaplotypeSamplerOverlay::UpdateScores(const Haplotype& path) {
  auto on_path = KmersOnPath(path);

  for (size_t i = 0; i < kmers_.size(); ++i) {
    auto& km = kmers_[i];
    switch (km.zygosity) {
      default:
        break;
      case KmerZygosity::HOMOZYGOUS:
        if (on_path.test(i)) km.score *= params_.homozygous_discount;
        break;
      case KmerZygosity::HETEROZYGOUS:
        km.score -= (on_path.test(i) ? params_.het_adjustment : -params_.het_adjustment);
        break;
    }
  }
}

HaplotypeSamplerOverlay::PathWithCoverage HaplotypeSamplerOverlay::BacktrackPath(const BestPathState& path_state, size_t back_automaton_state, size_t back_idx) const {
  const odgi::nid_t min_id = graph_.min_node_id();
  const odgi::nid_t max_id = graph_.max_node_id();

  Haplotype path;
  odgi::nid_t current_node = max_id;
  size_t current_automaton_state = back_automaton_state;
  size_t current_back_idx = back_idx;

  Graph::PathIdSet covered_paths = path_state[max_id - min_id].at(back_automaton_state).at(back_idx).covered_paths->bits;

  path.push_back(current_node);
  while (current_node != min_id) {
    const auto& backpointer = path_state[current_node - min_id].at(current_automaton_state).at(current_back_idx);
    current_back_idx = backpointer.pred_path_idx;
    current_automaton_state = backpointer.pred_automaton_state;
    current_node = backpointer.pred_node; assert(current_node >= min_id && current_node <= max_id);
    path.push_back(current_node);
  }

  std::reverse(path.begin(), path.end());
  return std::make_pair(std::move(path), covered_paths);
}

std::vector<HaplotypeSamplerOverlay::Haplotype> HaplotypeSamplerOverlay::SampleHaplotypes(size_t n) {
  // OPTIMIZATION_PROPOSALS.md "Proposal 2": run each draw as a single forward pass at exactly the requested
  // width instead of going through PropagateBestPathStateAdaptively's score_to_go-certified doubling loop.
  // Empirically validated (see that doc's "Empirical validation of Proposal 2") across 450 synthetic + 60
  // real-small-region + 21 real worst-case-region trials to never change the sampled result versus the
  // certified/widened result: every discard SortAndTrimBacktrack performs compares entries within a single
  // (node, automaton_state) pool, and score_to_go is a pure function of that same pair, so a same-pool
  // comparison of current score is already exact, not an approximation -- widening was only ever needed to
  // *certify* that, never to find a different answer. This also drops the ~43%-of-runtime ComputeScoreToGo
  // sweep (PERFORMANCE_NOTES.md "Higher-level timing breakdown") and the memory-compounding effect of
  // discarded widening attempts (PERFORMANCE_NOTES.md "Memory consumption by driver") entirely, since
  // neither ever runs here now. See SampleHaplotypesAdaptive(n) for the pre-Proposal-2 behavior, kept for
  // comparison in tests.
  const bool profiling = HaplotypeProfilingEnabled();
  static size_t draw_index = 0;  // single-threaded per PERFORMANCE_NOTES.md's existing thread-unsafe-counter assumption
  return SampleHaplotypesImpl(n, [&](size_t width) {
    auto draw_start = ProfileClock::now();
    auto path_state = PropagateBestPathState(width);
    if (profiling) {
      RssSample rss = CurrentRss();
      fmt::print(stderr, "HAP_PROFILE call={} n={} width={} stage=fixed_width ms={:.3f} rss_kb={} hwm_kb={}\n",
                 draw_index++, n, width, ElapsedMs(draw_start), rss.rss_kb, rss.hwm_kb);
    }
    return path_state;
  });
}

std::vector<HaplotypeSamplerOverlay::Haplotype> HaplotypeSamplerOverlay::SampleHaplotypesAdaptive(size_t n) {
  return SampleHaplotypesImpl(n, [this](size_t width) { return PropagateBestPathStateAdaptively(width); });
}

std::vector<HaplotypeSamplerOverlay::Haplotype> HaplotypeSamplerOverlay::SampleHaplotypesImpl(
    size_t n, const std::function<BestPathState(size_t)>& propagate) {
  std::vector<PathWithCoverage> samples;
  samples.reserve(n);

  while (samples.size() < n) {
    // Request more paths than already selected. Since we could select a path with no covered paths, we sample the
    // top (|selected|+2) paths to ensure we can find a new distinct path that covers at least one inference path.
    auto path_state = propagate(samples.size() + 2);
    const auto & candidates = path_state.back().at(kAutomatonRoot);

    size_t back_idx = candidates.size();
    for (size_t i = 0; i < candidates.size(); ++i) {
      const auto& candidate = candidates[i];
      if (apply_path_filter_ && candidate.covered_paths->bits.none()) {
        continue; // Skip paths that do not cover any inference paths when filtering is active
      }
      auto matching_sample = std::find_if(samples.begin(), samples.end(), [&](const PathWithCoverage& result) {
        return candidate.covered_paths->bits == result.second;
      });
      if (matching_sample == samples.end()) {
        back_idx = i;  // Found a new candidate that is not already in samples
        break;
      }
    }
    if (back_idx == candidates.size()) {
      // We did not find any new distinct paths, so stop sampling
      break;
    }

    // Extract and save the path of interest and its covered path set
    samples.push_back(std::move(BacktrackPath(path_state, kAutomatonRoot, back_idx)));
    UpdateScores(samples.back().first);
  }

  // Extract just the paths from the sampled results
  std::vector<Haplotype> results;
  results.reserve(samples.size());
  for (auto& [path, covered_paths] : samples) {
    results.push_back(std::move(path));
  }
  return results;
}

std::vector<std::pair<std::string, size_t>> HaplotypeSamplerOverlay::DecodeHaplotype(const Haplotype& haplotype) const {
  if (haplotype.empty()) {
    return {};
  }

  // Re-derive the same covered_paths bitset PropagateBestPathState accumulates during sampling (a simple linear
  // union, since we already have the complete path rather than needing to search for it).
  const size_t covered_paths_size = graph_.node_variant_paths_[haplotype.front()].size(); // All nodes should have the same size path sets
  Graph::PathIdSet covered_paths(covered_paths_size);
  for (auto node_id : haplotype) {
    if (!apply_path_filter_) {
      covered_paths |= graph_.node_variant_paths_[node_id];
    } else if (inference_node_mask_.test(node_id)) {
      covered_paths |= (graph_.node_variant_paths_[node_id] & inference_path_mask_);
    }
  }
  return DecodeHaplotype(covered_paths);
}

std::vector<std::pair<std::string, size_t>> HaplotypeSamplerOverlay::DecodeHaplotype(const Graph::PathIdSet& covered_paths) const {
  std::vector<std::pair<std::string, size_t>> result;
  for (auto path_idx = covered_paths.find_first(); path_idx != Graph::PathIdSet::npos;
       path_idx = covered_paths.find_next(path_idx)) {
    // Path names have the form "_alt_{variant_id}_{allele}" (see Graph::AltPathName); variant_id is a fixed-format
    // hex digest with no underscores, so the last two underscore-delimited fields are unambiguous.
    const auto path_name = graph_.get_path_name(handlegraph::as_path_handle(path_idx));
    auto allele_sep = path_name.rfind('_');
    auto variant_sep = path_name.rfind('_', allele_sep - 1);
    result.emplace_back(path_name.substr(variant_sep + 1, allele_sep - variant_sep - 1),
                         std::stoi(path_name.substr(allele_sep + 1)));
  }
  return result;
}

std::vector<HaplotypeSamplerOverlay::Diplotype> HaplotypeSamplerOverlay::SampleDiplotypes(
    const std::vector<Haplotype>& candidates, size_t n) const {
  if (candidates.empty() || n == 0) {
    return {};
  }

  // Pre-compute on-path bitsets once per candidate.
  std::vector<boost::dynamic_bitset<>> on_paths;
  on_paths.reserve(candidates.size());
  for (const auto& cand : candidates)
    on_paths.push_back(KmersOnPath(cand));

  // Score all pairs with replacement (j >= i).
  std::vector<Diplotype> scored;
  scored.reserve(candidates.size() * (candidates.size() + 1) / 2);

  for (size_t ci = 0; ci < candidates.size(); ++ci) {
    for (size_t cj = ci; cj < candidates.size(); ++cj) {
      // w(H, H') = sum over all k-mers of (1 - |observed_copy_count - expected_copy_count|)
      double score = 0.0;
      for (size_t k = 0; k < kmers_.size(); ++k) {
        int copy_count = (on_paths[ci].test(k) ? 1 : 0) + (on_paths[cj].test(k) ? 1 : 0);
        int expected;
        switch (kmers_[k].zygosity) {
          case KmerZygosity::ABSENT:       expected = 0; break;
          case KmerZygosity::HETEROZYGOUS: expected = 1; break;
          case KmerZygosity::HOMOZYGOUS:   expected = 2; break;
          default: continue;  // skip FREQUENT
        }
        score += 1.0 - std::abs(copy_count - expected);
      }
      scored.push_back({ci, cj, score});
    }
  }

  size_t keep = std::min(n, scored.size());
  std::partial_sort(scored.begin(), scored.begin() + keep, scored.end(),
      [](const Diplotype& a, const Diplotype& b) { return a.score > b.score; });
  scored.resize(keep);
  scored.shrink_to_fit();
  return scored;
}

}  // namespace npsv3
