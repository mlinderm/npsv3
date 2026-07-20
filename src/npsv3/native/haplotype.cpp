#include "haplotype.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <numeric>
#include <queue>
#include <unordered_map>
#include <unordered_set>

#include <boost/dynamic_bitset.hpp>
#include <fmt/std.h>
#include <fmt/ranges.h>

#include "variant.hpp"

namespace npsv3 {

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
//   output_kmers_    every k-mer whose location ends exactly here, including via failure links -- built
//                    during construction as own_kmer_set_[state] (k-mers whose location is *exactly*
//                    this state's path) unioned with the failure target's already-computed
//                    output_kmers_.
//
// AutomatonGoto(state, node_id) is the *effective* (fail-chasing) transition: follow goto_ if present,
// otherwise walk fail links until one has a goto_ for node_id, otherwise land on root. This is the
// classic non-completed Aho-Corasick query -- automaton_ only stores edges that were actually inserted,
// so goto_ is not "completed" into a total function ahead of time.
//
// The DP consumes a path by calling, once per real graph node it visits:
//     state = AutomatonGoto(state, node_id);
//     score += KmerSetScoreDelta(automaton_[state].output_kmers_);
// starting from state = kAutomatonRoot. Because output_kmers_ already includes every key ending at
// that state (via failure links), this single call correctly credits *all* keys -- short or long,
// nested or overlapping -- that end at this node, without the DP ever needing to special-case a
// multi-node key as an atomic edge. KmersOnPath does the same walk over a complete path to compute the
// same set (used by Score/UpdateScores/SampleDiplotypes) -- it is not a separate implementation.
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
//   state  path      goto_        fail    output_kmers_
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

HaplotypeSamplerOverlay::HaplotypeSamplerOverlay(
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
  automaton_.emplace_back(num_kmers); // root = kAutomatonRoot (0)

  // k-mers whose location is *exactly* this state's path. Made sparse since not all states are terminal for some k-mer.
  std::unordered_map<size_t, KmerIdSet> own_kmer_set;

  // Build the trie: One path from root per k-mer location, recording which k-mer indices share that
  // exact location.
  for (size_t kmer_idx = 0; kmer_idx < num_kmers; ++kmer_idx) {
    for (const auto& [handles, offset] : locations[kmer_idx]) {
      size_t state = kAutomatonRoot;
      for (const auto& handle : handles) {
        auto node_id = graph_.get_id(handle);
        auto child_it = automaton_[state].goto_.find(node_id);
        size_t child;
        if (child_it == automaton_[state].goto_.end()) {
          child = automaton_.size();
          automaton_.emplace_back(num_kmers); // May reallocate automaton_; index (not reference) into it afterward
          automaton_[state].goto_.emplace(node_id, child);
        } else {
          child = child_it->second;
        }
        state = child;
      }
      own_kmer_set.try_emplace(state, num_kmers).first->second.set(kmer_idx);
    }
  }

  // BFS to compute failure links and output_kmers_ (standard Aho-Corasick construction). BFS visits
  // states in non-decreasing depth order, and a state's failure link always points to a strictly
  // shallower state, so each state's failure target's output_kmers_ is already finalized when needed.
  std::queue<size_t> to_visit;
  for (auto& [node_id, child] : automaton_[kAutomatonRoot].goto_) {
    automaton_[child].fail = kAutomatonRoot;
    to_visit.push(child);
  }
  while (!to_visit.empty()) {
    size_t state = to_visit.front();
    to_visit.pop();

    automaton_[state].output_kmers_ = automaton_[automaton_[state].fail].output_kmers_;
    if (auto it = own_kmer_set.find(state); it != own_kmer_set.end()) {
      automaton_[state].output_kmers_ |= it->second;
    }

    // Standard construction: fail[child] = goto*(fail[state], node_id) is the state reached by taking
    // the same node_id edge from state's own failure state, using the identical fail-chasing lookup
    // the DP uses at query time (AutomatonGoto). Safe to call here: goto_ is fully built for every
    // state before this BFS starts, and fail[state] always points to a strictly shallower state, whose
    // own .fail chain BFS has already been finalized.
    for (auto& [node_id, child] : automaton_[state].goto_) {
      automaton_[child].fail = AutomatonGoto(automaton_[state].fail, node_id);
      to_visit.push(child);
    }
  }
}

HaplotypeSamplerOverlay::HaplotypeSamplerOverlay(const Graph& graph, const UniqueKmersOverlay& unique_kmers, const Params& params)
    : HaplotypeSamplerOverlay(graph, unique_kmers.sequences(), unique_kmers.locations(), params) {
}

HaplotypeSamplerOverlay::HaplotypeSamplerOverlay(const Graph& graph, const UniqueKmersOverlay& unique_kmers,
                                                 const std::string& inference_vcf, const Range& region,
                                                 size_t min_size, const Params& params)
    : HaplotypeSamplerOverlay(graph, unique_kmers.sequences(), unique_kmers.locations(), params) {
  // Initialize inference VCF filtering
  apply_path_filter_ = true;
  graph.PopulateNodeAndPathMasks(inference_vcf, region, min_size, inference_node_mask_, inference_path_mask_);
  assert(inference_path_mask_.any());
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
    std::sort(backtrack.begin(), backtrack.end(), [](const auto& a, const auto& b) {
      // Group by covered_paths first, then sort by descending score within groups
      if (a.covered_paths == b.covered_paths) {
        return a.score > b.score;
      }
      return a.covered_paths > b.covered_paths;
    });
    // Retain only the highest-scoring representative per inference equivalence class.
    auto last = std::unique(backtrack.begin(), backtrack.end(), [](const auto& a, const auto& b) {
      return a.covered_paths == b.covered_paths;
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
  while (state != kAutomatonRoot) {
    auto it = automaton_[state].goto_.find(node_id);
    if (it != automaton_[state].goto_.end()) return it->second;
    state = automaton_[state].fail;
  }
  auto it = automaton_[kAutomatonRoot].goto_.find(node_id);
  return it != automaton_[kAutomatonRoot].goto_.end() ? it->second : kAutomatonRoot;
}

double HaplotypeSamplerOverlay::KmerSetScoreDelta(const KmerIdSet& set) const {
  double total = 0.0;
  for (size_t kmer_idx = set.find_first(); kmer_idx != KmerIdSet::npos; kmer_idx = set.find_next(kmer_idx)) {
    total += kmers_[kmer_idx].score;
  }
  return 2.0 * total;
}

std::vector<HaplotypeSamplerOverlay::Haplotype> HaplotypeSamplerOverlay::FindBestPaths(size_t n) const {
  auto path_state = PropagateBestPathStateAdaptively(n);
  const auto & best_paths = path_state.back().at(kAutomatonRoot); // Sink pools are always merged under this one key

  std::vector<Haplotype> result;
  result.reserve(best_paths.size());
  for (size_t back_idx = 0; back_idx < best_paths.size(); ++back_idx) {
    if (apply_path_filter_ && best_paths[back_idx].covered_paths.none()) {
      continue; // Skip paths that do not cover any inference paths when filtering is active
    }
    result.push_back(std::move(BacktrackPath(path_state, kAutomatonRoot, back_idx).first));
  }
  return result;
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

  // Seed the source node: a virtual transition from the automaton's root consuming min_id itself, exactly
  // mirroring how every other node's arrival is processed below.
  {
    size_t seed_state = AutomatonGoto(kAutomatonRoot, min_id);
    Graph::PathIdSet seed_covered_paths(covered_paths_size);
    accumulate_covered_paths(seed_covered_paths, min_id);
    dp[0][seed_state].push_back({
      min_score + KmerSetScoreDelta(automaton_[seed_state].output_kmers_),
      0,  // no predecessor node
      kAutomatonRoot, // irrelevant without predecessor node
      0, // irrelevant without predecessor node
      std::move(seed_covered_paths)
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

      for (auto& [automaton_state, pool] : node_state) {
        size_t new_state = AutomatonGoto(automaton_state, next_node);
        double weight_delta = KmerSetScoreDelta(automaton_[new_state].output_kmers_);
        auto& next_pool = next_node_state[new_state];
        for (size_t b_idx = 0; b_idx < pool.size(); ++b_idx) {
          Graph::PathIdSet new_covered_paths = pool[b_idx].covered_paths;
          accumulate_covered_paths(new_covered_paths, next_node);
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
      for (size_t s = 0; s < num_states; ++s) {
        size_t new_state = AutomatonGoto(s, next_node);
        double weight_delta = KmerSetScoreDelta(automaton_[new_state].output_kmers_);
        row[s] = std::max(row[s], weight_delta + next_row[new_state]);
      }
      return true;
    });
  }

  return score_to_go;
}

HaplotypeSamplerOverlay::BestPathState HaplotypeSamplerOverlay::PropagateBestPathStateAdaptively(size_t n, size_t max_widening) const {
  auto score_to_go = ComputeScoreToGo();

  // Adaptive beam search. Start by maintaining n backpointers, then double with the width until the top scoring paths
  // are guaranteed to be included. max_escaped_bound reports an admissible upper bound on what a discarded branch
  // could still have scored. If the weakest_kept_score is >= that bound, no discarded branch could have beaten the
  // worst kept result, so the top-n set must contain the true top-n and the search can stop. Otherwise, some discarded
  // branch *might* have beaten the worst kept result, so the width is doubled and the search is redone from scratch at
  // the wider beam.
  for (size_t width = n; ; width *= 2) {
    double max_escaped_bound = -std::numeric_limits<double>::infinity();
    auto path_state = PropagateBestPathState(width, &score_to_go, &max_escaped_bound);

    const auto& results = path_state.back().at(kAutomatonRoot);
    double weakest_kept_score = results.empty() ? -std::numeric_limits<double>::infinity() : results.back().score;
    if (weakest_kept_score >= max_escaped_bound || width > max_widening * std::max<size_t>(n, 1)) {
      return path_state;
    }
  }
}

HaplotypeSamplerOverlay::KmerIdSet HaplotypeSamplerOverlay::KmersOnPath(const Haplotype& path) const {
  // Walk the automaton one node at a time for path accumulating output_kmers_, the unique k-mers for
  // sub-path match ending at that node.
  KmerIdSet on_path(kmers_.size());
  size_t state = kAutomatonRoot;
  for (auto node_id : path) {
    state = AutomatonGoto(state, node_id);
    on_path |= automaton_[state].output_kmers_;
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

  Graph::PathIdSet covered_paths = path_state[max_id - min_id].at(back_automaton_state).at(back_idx).covered_paths;

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
  std::vector<PathWithCoverage> samples;
  samples.reserve(n);

  while (samples.size() < n) {
    // Request more paths than already selected. Since we could select a path with no covered paths, we sample the
    // top (|selected|+2) paths to ensure we can find a new distinct path that covers at least one inference path.
    auto path_state = PropagateBestPathStateAdaptively(samples.size() + 2);
    const auto & candidates = path_state.back().at(kAutomatonRoot);

    size_t back_idx = candidates.size();
    for (size_t i = 0; i < candidates.size(); ++i) {
      const auto& candidate = candidates[i];
      if (apply_path_filter_ && candidate.covered_paths.none()) {
        continue; // Skip paths that do not cover any inference paths when filtering is active
      }
      auto matching_sample = std::find_if(samples.begin(), samples.end(), [&](const PathWithCoverage& result) {
        return candidate.covered_paths == result.second;
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
