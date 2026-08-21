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

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/dynamic_bitset.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/serialization/vector.hpp>
#include <fmt/std.h>
#include <fmt/ranges.h>
#include <gbwt/dynamic_gbwt.h>
#include <spdlog/spdlog.h>

#include <optional>

#include "variant.hpp"

namespace boost {
namespace serialization {

template <class Archive>
void serialize(Archive& ar, npsv3::HaplotypePriorOverlay::NodeTransition& t, unsigned int) {
  ar & t.to_node;
  ar & t.count;
}

}  // namespace serialization
}  // namespace boost

namespace npsv3 {

namespace {
// Ad hoc, opt-in (NPSV3_HAPLOTYPE_PROFILE=1) timing/memory breakdown for SampleHaplotypes, documented in
// PERFORMANCE_NOTES.md: since it re-runs the DP from scratch once per sampled haplotype, per-draw wall time
// and resident memory (rss_kb/hwm_kb) are printed to stderr as one HAP_PROFILE line per draw. Negligible
// overhead when disabled (one getenv call, cached in a function-local static).
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
// stay elevated for a while after a large draw's C++ objects are destroyed); hwm_kb ("high water mark")
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

namespace {

/// Combining hash for a (from_node, to_node) pair key
struct NodePairHash {
  size_t operator()(const std::pair<odgi::nid_t, odgi::nid_t>& p) const noexcept {
    size_t h = std::hash<odgi::nid_t>()(p.first);
    h ^= std::hash<odgi::nid_t>()(p.second) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    return h;
  }
};

}  // namespace

HaplotypePriorOverlay::HaplotypePriorOverlay(const Graph& graph, const std::vector<std::string>& excluded_samples)
    : graph_(graph) {
  const std::unordered_set<std::string> excluded(excluded_samples.begin(), excluded_samples.end());
  BuildTransitionTable(excluded);
  BuildGBWTIndex(excluded);
}

void HaplotypePriorOverlay::BuildTransitionTable(const std::unordered_set<std::string>& excluded_samples) {
  const odgi::nid_t min_id = graph_.min_node_id();
  const odgi::nid_t max_id = graph_.max_node_id();
  const size_t node_space = static_cast<size_t>(max_id - min_id + 1);

  std::unordered_map<std::pair<odgi::nid_t, odgi::nid_t>, uint32_t, NodePairHash> counts;
  std::vector<uint32_t> node_totals(node_space, 0);

  graph_.for_each_path_handle([&](const handlegraph::path_handle_t& path_handle) {
    auto path_name = graph_.get_path_name(path_handle);
    auto hash_pos = path_name.find('#');
    if (hash_pos == std::string::npos) {
      return;  // Not a background sample genotype/segment path (Sample#HaplotypeIndex#Contig#SegmentIndex)
    }
    if (!excluded_samples.empty() && excluded_samples.count(path_name.substr(0, hash_pos))) {
      return;  // Caller-requested sample exclusion (e.g. the sample being genotyped)
    }

    // Walk the path's literal node sequence, skipping non-distinguishing (shared reference) nodes
    odgi::nid_t prev = -1;  // Sentinel for no distinguishing node seen yet on this path
    for (auto node_id : graph_.PathNodes(path_handle)) {
      if (graph_.node_variant_paths_[node_id].none()) continue;
      if (prev != -1) {
        counts[{prev, node_id}]++;
        node_totals[static_cast<size_t>(prev - min_id)]++;
      }
      prev = node_id;
    }
  });

  // Compact into a sorted-by-(from, to) CSR structure for deterministic, cache-friendly per-source lookup.
  std::vector<std::pair<std::pair<odgi::nid_t, odgi::nid_t>, uint32_t>> sorted_counts(counts.begin(), counts.end());
  std::sort(sorted_counts.begin(), sorted_counts.end());

  transition_starts_.assign(node_space + 1, 0);
  for (const auto& [key, count] : sorted_counts) {
    transition_starts_[static_cast<size_t>(key.first - min_id) + 1]++;
  }
  for (size_t i = 0; i < node_space; i++) {
    transition_starts_[i + 1] += transition_starts_[i];
  }

  transitions_.reserve(sorted_counts.size());
  for (const auto& [key, count] : sorted_counts) {
    transitions_.push_back(NodeTransition{key.second, count});
  }

  node_totals_ = std::move(node_totals);
}

HaplotypePriorOverlay::HaplotypePriorOverlay(const Graph& graph, gbwt::GBWT index, std::vector<StateKey> state_keys,
                                             std::unordered_map<StateKey, uint32_t, StateKeyHash> state_index,
                                             std::vector<uint32_t> width, std::vector<double> score_to_go,
                                             std::vector<size_t> transition_starts,
                                             std::vector<NodeTransition> transitions,
                                             std::vector<uint32_t> node_totals)
    : graph_(graph),
      index_(std::move(index)),
      state_keys_(std::move(state_keys)),
      state_index_(std::move(state_index)),
      width_(std::move(width)),
      score_to_go_(std::move(score_to_go)),
      transition_starts_(std::move(transition_starts)),
      transitions_(std::move(transitions)),
      node_totals_(std::move(node_totals)) {}

size_t HaplotypePriorOverlay::RowIndex(odgi::nid_t from_node) const {
  const odgi::nid_t min_id = graph_.min_node_id();
  const odgi::nid_t max_id = graph_.max_node_id();
  if (from_node < min_id || from_node > max_id) return static_cast<size_t>(-1);
  return static_cast<size_t>(from_node - min_id);
}

double HaplotypePriorOverlay::TransitionProbability(odgi::nid_t from_node, odgi::nid_t to_node,
                                                    double alpha) const {
  const size_t row = RowIndex(from_node);
  if (row == static_cast<size_t>(-1)) return 1.0;  // outside the graph's node range: no information

  const size_t begin = transition_starts_[row];
  const size_t end = transition_starts_[row + 1];
  const size_t k_to = std::max<size_t>(end - begin, 1);  // floor of 1: see class docs

  auto it = std::lower_bound(transitions_.begin() + begin, transitions_.begin() + end, to_node,
                              [](const NodeTransition& t, odgi::nid_t to) { return t.to_node < to; });
  const uint32_t count = (it != transitions_.begin() + end && it->to_node == to_node) ? it->count : 0;
  const uint32_t total = node_totals_[row];

  return (count + alpha) / (total + alpha * static_cast<double>(k_to));
}

double HaplotypePriorOverlay::LogTransitionProbability(odgi::nid_t from_node, odgi::nid_t to_node,
                                                       double alpha) const {
  return std::log(TransitionProbability(from_node, to_node, alpha));
}

namespace {

struct BackgroundPathTag {
  std::string sample;
  int haplotype_index;
};

// Parses "Sample#HaplotypeIndex#Contig#SegmentIndex" -> {sample, haplotype_index}. Returns std::nullopt if
// path_name doesn't look like a background sample path (same '#'-name predicate as SamplesIncluding).
std::optional<BackgroundPathTag> ParseBackgroundPathName(const std::string& path_name) {
  auto first = path_name.find('#');
  if (first == std::string::npos) return std::nullopt;
  auto second = path_name.find('#', first + 1);
  if (second == std::string::npos) return std::nullopt;  // malformed; defensively skip
  return BackgroundPathTag{path_name.substr(0, first),
                            std::stoi(path_name.substr(first + 1, second - first - 1))};
}

}  // namespace

size_t HaplotypePriorOverlay::StateKeyHash::operator()(const StateKey& k) const noexcept {
  size_t h = std::hash<gbwt::node_type>()(k.node);
  h ^= std::hash<gbwt::size_type>()(k.lo) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
  h ^= std::hash<gbwt::size_type>()(k.hi) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
  return h;
}

void HaplotypePriorOverlay::BuildGBWTIndex(const std::unordered_set<std::string>& excluded_samples) {
  // gbwt::GBWTBuilder's node_width must cover the largest encoded node id (id << 1 | is_reverse) ever inserted or
  // searched (too small results in silent corruption). This graph is always forward-oriented but we still reserve the
  // reverse bit to match gbwt's node_width convention.
  gbwt::size_type node_width = gbwt::bit_length(gbwt::Node::encode(graph_.max_node_id(), true));
  gbwt::GBWTBuilder builder(node_width);

  std::unordered_map<std::string, uint32_t> sample_ids;
  std::unordered_map<uint32_t, uint32_t> panel_member_ids;  // packed (sample_id<<8|hap) -> dense member id
  std::vector<gbwt::vector_type> panel_sequences;
  std::vector<uint32_t> panel_sequence_members;  // parallel to panel_sequences: dense panel-member id

  graph_.for_each_path_handle([&](const handlegraph::path_handle_t& path_handle) {
    auto path_name = graph_.get_path_name(path_handle);
    auto tag = ParseBackgroundPathName(path_name);
    if (!tag) return;  // not a background sample genotype/segment path
    if (excluded_samples.count(tag->sample)) return;  // Caller-requested sample exclusion (e.g. the sample being genotyped)

    auto [sample_it, unused1] = sample_ids.try_emplace(tag->sample, static_cast<uint32_t>(sample_ids.size()));
    uint32_t packed = (sample_it->second << 8) | static_cast<uint32_t>(tag->haplotype_index);
    auto [member_it, unused2] = panel_member_ids.try_emplace(packed, static_cast<uint32_t>(panel_member_ids.size()));

    auto nodes = graph_.PathNodes(path_handle);
    gbwt::vector_type seq;
    seq.reserve(nodes.size());
    for (auto node_id : nodes) {
      seq.push_back(gbwt::Node::encode(node_id, false));
    }
    if (seq.empty()) return;

    builder.insert(seq, false);
    panel_sequences.push_back(std::move(seq));
    panel_sequence_members.push_back(member_it->second);
  });
  builder.finish();
  index_ = builder.index;

  // Walk every panel haplotype's own node sequence again, this time through the *finished* index's
  // Find()/Extend(), to enumerate the induced state graph (states as (node,[lo,hi)), edges via Extend) and
  // which panel members reach it.
  //
  // This walk is exhaustive over every state a caller could ever reach. Every panel path here is walked from its own
  // genuine start (the region's entry node, or a phase-break), so any caller who does the same only reaches states this
  // walk already covers.
  //
  // This also gives Width() without gbwt::locate(). Since the walk already knows which panel member it's
  // following, recording that membership directly as we go is cheaper than a separate locate()+dedup pass
  // and yields the same count.
  const size_t num_panel_members = panel_member_ids.size();
  std::vector<boost::dynamic_bitset<>> reaching_members;  // parallel to state_keys_, transient
  std::vector<std::vector<uint32_t>> successors;          // parallel to state_keys_: successor state indices

  auto GetOrCreateState = [&](const PanelState& state) -> uint32_t {
    auto key = KeyOf(state);
    auto [it, inserted] = state_index_.try_emplace(key, static_cast<uint32_t>(state_keys_.size()));
    if (inserted) {
      state_keys_.push_back(key);
      reaching_members.emplace_back(num_panel_members);
      successors.emplace_back();
    }
    return it->second;
  };

  for (size_t p = 0; p < panel_sequences.size(); p++) {
    const auto& seq = panel_sequences[p];
    uint32_t member = panel_sequence_members[p];

    PanelState state = index_.find(seq[0]);
    assert(!state.empty());
    uint32_t state_idx = GetOrCreateState(state);
    reaching_members[state_idx].set(member);

    for (size_t i = 1; i < seq.size(); i++) {
      PanelState next_state = index_.extend(state, seq[i]);
      assert(!next_state.empty());
      uint32_t next_idx = GetOrCreateState(next_state);
      reaching_members[next_idx].set(member);
      successors[state_idx].push_back(next_idx); // duplicate edges across panel members deduped below
      state = next_state;
      state_idx = next_idx;
    }
  }

  width_.resize(state_keys_.size());
  for (size_t i = 0; i < state_keys_.size(); i++) {
    width_[i] = static_cast<uint32_t>(reaching_members[i].count());
    auto& succ = successors[i];
    std::sort(succ.begin(), succ.end());
    succ.erase(std::unique(succ.begin(), succ.end()), succ.end());
  }

  // Backward max-recursion over log-width-ratios. Base case (score_to_go_ left at its default-initialized 0.0) is any
  // state with no recorded successors: either the region's last node, or for empty/ never-encountered states, handled
  // separately by ScoreToGo()'s own empty/lookup-miss checks, not by this table at all.
  score_to_go_.assign(state_keys_.size(), 0.0);
  std::vector<std::vector<uint32_t>> states_by_node_id(graph_.max_node_id() + 1);
  for (size_t i = 0; i < state_keys_.size(); i++) {
    states_by_node_id[gbwt::Node::id(state_keys_[i].node)].push_back(static_cast<uint32_t>(i));
  }
  for (auto nid = graph_.max_node_id(); nid >= graph_.min_node_id(); nid--) {
    for (uint32_t state_idx : states_by_node_id[nid]) {
      if (successors[state_idx].empty()) continue;  // base case: score_to_go_ already 0.0
      double best = -std::numeric_limits<double>::infinity();
      for (uint32_t next_idx : successors[state_idx]) {
        double delta = std::log(static_cast<double>(width_[next_idx]) / static_cast<double>(width_[state_idx]));
        best = std::max(best, delta + score_to_go_[next_idx]);
      }
      score_to_go_[state_idx] = best;
    }
  }
}

HaplotypePriorOverlay::PanelState HaplotypePriorOverlay::Find(odgi::nid_t node) const {
  return index_.find(gbwt::Node::encode(node, false));
}

HaplotypePriorOverlay::PanelState HaplotypePriorOverlay::Extend(const PanelState& state, odgi::nid_t node) const {
  return index_.extend(state, gbwt::Node::encode(node, false));
}

size_t HaplotypePriorOverlay::Width(const PanelState& state) const {
  if (state.empty()) return 0;
  auto it = state_index_.find(KeyOf(state));
  return it != state_index_.end() ? width_[it->second] : 0;
}

double HaplotypePriorOverlay::ScoreToGo(const PanelState& state) const {
  if (state.empty()) return 0.0;
  auto it = state_index_.find(KeyOf(state));
  return it != state_index_.end() ? score_to_go_[it->second] : 0.0;
}

// -----------------------------------------------------------------------------------------------
// HaplotypePriorOverlay serialization
// -----------------------------------------------------------------------------------------------
//
// The GBWT index writes itself first via its own native serialize(); everything else follows in a single Boost archive
// on the same stream.

void HaplotypePriorOverlay::Save(std::ostream& out) const {
  index_.serialize(out);

  std::vector<gbwt::node_type> nodes;
  std::vector<gbwt::size_type> los, his;
  nodes.reserve(state_keys_.size());
  los.reserve(state_keys_.size());
  his.reserve(state_keys_.size());
  for (const auto& key : state_keys_) {
    nodes.push_back(key.node);
    los.push_back(key.lo);
    his.push_back(key.hi);
  }

  boost::archive::binary_oarchive oa(out);
  oa << nodes << los << his << width_ << score_to_go_;
  oa << transition_starts_ << transitions_ << node_totals_;
}

void HaplotypePriorOverlay::Save(const std::string& path) const {
  std::ofstream out(path, std::ios::binary);
  if (!out) throw std::runtime_error("Cannot open haplotype prior file for writing: " + path);
  Save(out);
}

void HaplotypePriorOverlay::Load(HaplotypePriorOverlay* target, const Graph& graph, std::istream& in) {
  gbwt::GBWT index;
  index.load(in);

  boost::archive::binary_iarchive ia(in);
  std::vector<gbwt::node_type> nodes;
  std::vector<gbwt::size_type> los, his;
  std::vector<uint32_t> width;
  std::vector<double> score_to_go;
  ia >> nodes >> los >> his >> width >> score_to_go;

  std::vector<size_t> transition_starts;
  std::vector<NodeTransition> transitions;
  std::vector<uint32_t> node_totals;
  ia >> transition_starts >> transitions >> node_totals;

  std::vector<StateKey> state_keys(nodes.size());
  std::unordered_map<StateKey, uint32_t, StateKeyHash> state_index;
  state_index.reserve(nodes.size());
  for (size_t i = 0; i < nodes.size(); i++) {
    state_keys[i] = StateKey{nodes[i], los[i], his[i]};
    state_index.emplace(state_keys[i], static_cast<uint32_t>(i));
  }

  new (target) HaplotypePriorOverlay(graph, std::move(index), std::move(state_keys), std::move(state_index),
                                     std::move(width), std::move(score_to_go), std::move(transition_starts),
                                     std::move(transitions), std::move(node_totals));
}

void HaplotypePriorOverlay::Load(HaplotypePriorOverlay* target, const Graph& graph, const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("Cannot open haplotype prior file: " + path);
  Load(target, graph, in);
}

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
    HaplotypeSamplerCommonCtor, const Graph& graph, const std::vector<std::string>& sequences,
    const std::vector<std::vector<UniqueKmersOverlay::KmerLocation>>& locations, const HaplotypePriorOverlay* prior,
    const Params& params)
    : graph_(graph), params_(params), apply_path_filter_(false), kmer_sequences_(sequences), prior_(prior) {
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

  // Population-prior "distinguishing node" bookkeeping is deliberately *not* filtered by the inference-VCF
  // mask.
  InitializePopulationDistinguishingMask();
}

HaplotypeSamplerOverlay::HaplotypeSamplerOverlay(
    const Graph& graph, const std::vector<std::string>& sequences,
    const std::vector<std::vector<UniqueKmersOverlay::KmerLocation>>& locations,
    const HaplotypePriorOverlay* prior, const Params& params)
    : HaplotypeSamplerOverlay(kCommonCtor, graph, sequences, locations, prior, params) {
  // Initialize the contributes_paths_mask_ based on the inference masks, so that we can do fast
  // propagation of covered_paths when the current node doesn't contribute any new paths.
  InitializeContributesPathsMask();
}

HaplotypeSamplerOverlay::HaplotypeSamplerOverlay(const Graph& graph, const UniqueKmersOverlay& unique_kmers,
                                                 const HaplotypePriorOverlay* prior, const Params& params)
    : HaplotypeSamplerOverlay(kCommonCtor, graph, unique_kmers.sequences(), unique_kmers.locations(), prior, params) {
  // Initialize the contributes_paths_mask_ based on the inference masks, so that we can do fast
  // propagation of covered_paths when the current node doesn't contribute any new paths.
  InitializeContributesPathsMask();
}

HaplotypeSamplerOverlay::HaplotypeSamplerOverlay(const Graph& graph, const UniqueKmersOverlay& unique_kmers,
                                                 const std::string& inference_vcf, const Range& region,
                                                 size_t min_size, const HaplotypePriorOverlay* prior, const Params& params)
    : HaplotypeSamplerOverlay(kCommonCtor, graph, unique_kmers.sequences(), unique_kmers.locations(), prior, params) {
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

void HaplotypeSamplerOverlay::InitializePopulationDistinguishingMask() {
  const odgi::nid_t min_id = graph_.min_node_id();
  const odgi::nid_t max_id = graph_.max_node_id();

  population_distinguishing_mask_.clear();
  population_distinguishing_mask_.resize(max_id + 1);
  for (odgi::nid_t i = min_id; i <= max_id; ++i) {
    if (!graph_.has_node(i)) continue;
    if (graph_.node_variant_paths_[i].any()) population_distinguishing_mask_.set(i);
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
  // covered_paths is an immutable, refcounted PathIdSetHolder to enable fast "shallow" copies when there
  // are no changes. First check for pointer equality, then hash equality and then bits. Only entries with
  // *different* holders fall back to comparing ->hash and, on a hash tie, ->bits (potentially thousands of
  // bits). Since equal covered_paths always hash equal, this can never merge or split groups differently
  // than comparing covered_paths directly.
  template <typename T>
  bool SameCoveredPathsClass(const T& a, const T& b) {
    return a.covered_paths == b.covered_paths ||
           (a.covered_paths->hash == b.covered_paths->hash && a.covered_paths->bits == b.covered_paths->bits);
  }

  // Population-state equality. Deliberately conservative: under-merging is the safe default, the explicit
  // population_state_pool_cap bounds the cost, not this predicate. When the prior is disabled every entry
  // carries an identical default-constructed PopulationState, so this is always true and SamePoolClass
  // below degenerates structurally to SameCoveredPathsClass alone.
  template <typename P>
  bool SamePopulationState(const P& a, const P& b) {
    if (a.last_distinguishing_node != b.last_distinguishing_node) return false;
    if (const auto* ag = std::get_if<typename P::Gbwt>(&a.tier)) {
      const auto* bg = std::get_if<typename P::Gbwt>(&b.tier);
      return bg != nullptr && ag->state.node == bg->state.node && ag->state.range == bg->state.range;
    }
    return std::holds_alternative<typename P::FellBack>(b.tier);
  }

  template <typename T>
  bool SamePoolClass(const T& a, const T& b) {
    return SameCoveredPathsClass(a, b) && SamePopulationState(a.population_state, b.population_state);
  }

  /// Dedup @p backtrack by (covered_paths, population_state) and trim to the top @p n highest-scoring
  /// distinct classes, ranked by @p rank_key (defaults to identity on .score).
  ///
  /// Use a single O(|backtrack| x n) pass against a small (<= n) `kept` buffer, instead of a full O(m log m)
  /// comparison sort purely to group-and-dedup followed by a second sort to rank by score. Every incoming
  /// pool was already trimmed to <= n (typically 8 or less) at the previous node, so |backtrack| here is small
  /// (bounded by roughly in-degree x n, or occupied-automaton-states x n at the sink merge), so a linear scan
  //// can be efficient.
  ///
  /// Applied at every (node, automaton_state) pool, even those with pending (mid k-mer) matches. `KmerSetScoreDelta`
  /// (the only per-edge k-mer score contribution) is a pure function of (source automaton_state, next_node), and so is
  /// applied identically to every entry in a pool for a given edge, regardless of covered_paths. The population-prior
  /// term is *not* independent of covered_paths/population_state the same way, so @p rank_key exists to let ranking
  /// incorporate PopulationScoreToGo as a heuristic aid, while the *stored* .score on every surviving entry
  /// remains the real, unmodified accumulated score.
  ///
  /// @p same_class is the dedup predicate. **The sink/final merge deliberately overrides this to SameCoveredPathsClass
  /// alone** (haplotype.cpp's PropagateBestPathState call site) rather than accepting the default: under an active
  /// inference-VCF filter, population_distinguishing_mask_ is deliberately *not* masked the same way
  /// contributes_paths_mask_ is, so two entries can reach the sink with identical (filtered) covered_paths but
  /// different population_state, e.g. two upstream branches at a *masked-out* variant, re-converging at a later
  /// *inference* variant. Population state is genuinely NOT provably determined by covered_paths in that case.
  /// FindBestPaths's output-distinctness contract is covered_paths-only. A covered_paths-only dedup at the sink
  /// preserves that, while `.score` (by then a real, fully-accumulated value, not a heuristic) still picks the
  /// population-aware best representative of each covered_paths class. Intermediate trimming (the other call site)
  /// still needs the population-aware predicate, or a branch the prior would favor later could be discarded prematurely
  /// before covered_paths alone would have distinguished it.
  ///
  /// The surviving entries' order is otherwise unspecified unless @p sort_result is true, which additionally
  /// sorts the (<= n, so cheap) result by descending rank_key. Callers whose result must be in
  /// descending-score order (currently: only the final sink/root pool, consumed by SampleHaplotypes's greedy
  /// per-draw walk and FindBestPaths's public sorted-order contract) must pass sort_result=true; every other
  /// call site's result only feeds further DP propagation, where entries are consumed via stable stored
  /// indices (Backpointer::pred_path_idx), not iteration order, so no order is needed there.
  template <typename T, typename RankKeyFn, typename SameClassFn>
  void SortAndTrimBacktrack(T& backtrack, size_t n, bool sort_result, RankKeyFn rank_key, SameClassFn same_class) {
    if (n == 0 || backtrack.empty()) {
      backtrack.clear();
      return;
    }

    T kept;
    kept.reserve(n);
    size_t worst_idx = 0; // valid only once kept.size() == n

    auto recompute_worst = [&]() {
      worst_idx = 0;
      for (size_t i = 1; i < kept.size(); ++i) {
        if (rank_key(kept[i]) < rank_key(kept[worst_idx])) worst_idx = i;
      }
    };

    for (auto& candidate : backtrack) {
      size_t match_idx = kept.size();
      for (size_t i = 0; i < kept.size(); ++i) {
        if (same_class(candidate, kept[i])) {
          match_idx = i;
          break;
        }
      }
      if (match_idx < kept.size()) {
        if (rank_key(candidate) > rank_key(kept[match_idx])) kept[match_idx] = std::move(candidate);
        continue;
      }
      if (kept.size() < n) {
        kept.push_back(std::move(candidate));
        if (kept.size() == n) recompute_worst();
        continue;
      }
      if (rank_key(candidate) > rank_key(kept[worst_idx])) {
        kept[worst_idx] = std::move(candidate);
        recompute_worst();
      }
    }

    if (sort_result) {
      std::sort(kept.begin(), kept.end(),
                [&](const auto& a, const auto& b) { return rank_key(a) > rank_key(b); });
    }
    backtrack = std::move(kept);
  }

  // Default dedup predicate: SamePoolClass (covered_paths AND population_state).
  template <typename T, typename RankKeyFn>
  void SortAndTrimBacktrack(T& backtrack, size_t n, bool sort_result, RankKeyFn rank_key) {
    SortAndTrimBacktrack(backtrack, n, sort_result, rank_key,
                         [](const auto& a, const auto& b) { return SamePoolClass(a, b); });
  }

  // Default rank-key: identity on .score, today's (pre-prior) behavior.
  template <typename T>
  void SortAndTrimBacktrack(T& backtrack, size_t n, bool sort_result = false) {
    SortAndTrimBacktrack(backtrack, n, sort_result, [](const auto& e) { return e.score; });
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
    if (apply_path_filter_) {
      // A haplotype that never explicitly selects a flagged allele for some inference variant still
      // implicitly declines it (i.e. carries the reference allele there), even if it doesn't explicitly
      // traverse the reference allele's own node(s).
      auto adjusted = graph_.ApplyReferenceFallback(best_paths[back_idx].covered_paths->bits, inference_path_mask_);
      if (adjusted.none()) {
        continue; // Skip paths that do not cover (or implicitly decline) any inference variant
      }
    }
    result.push_back(std::move(BacktrackPath(path_state, kAutomatonRoot, back_idx).first));
  }
  return result;
}

std::vector<HaplotypeSamplerOverlay::Haplotype> HaplotypeSamplerOverlay::FindBestPaths(size_t n) const {
  return ExtractBestPaths(PropagateBestPathState(n));
}

void HaplotypeSamplerOverlay::EnsureTransitionScoreToGo() const {
  if (transition_score_to_go_built_) return;
  transition_score_to_go_built_ = true;
  if (!prior_) return;

  const odgi::nid_t min_id = graph_.min_node_id();
  const odgi::nid_t max_id = graph_.max_node_id();
  transition_score_to_go_.assign(static_cast<size_t>(max_id - min_id + 1), 0.0);

  // Backward max-recursion over HaplotypePriorOverlay's node-keyed adjacency, in strictly decreasing node-id order.
  // Every recorded (from, to) pair has to_node's id strictly greater than from_node's, so score_to_go for a later node
  // never depends on an earlier one.
  const auto& starts = prior_->transition_starts();
  const auto& transitions = prior_->transitions();
  for (odgi::nid_t from = max_id; from >= min_id; --from) {
    const size_t row = static_cast<size_t>(from - min_id);
    if (row + 1 >= starts.size()) continue;
    double best = 0.0; // base case: no recorded successor (region end, or not a source at all)
    for (size_t i = starts[row]; i < starts[row + 1]; ++i) {
      const auto& t = transitions[i];
      const double log_p = std::log(prior_->TransitionProbability(from, t.to_node, params_.transition_prior_alpha));
      const double candidate = log_p + transition_score_to_go_[static_cast<size_t>(t.to_node - min_id)];
      if (i == starts[row] || candidate > best) best = candidate;
    }
    transition_score_to_go_[row] = best;
  }
}

double HaplotypeSamplerOverlay::TransitionScoreToGo(odgi::nid_t from) const {
  if (!prior_) return 0.0;
  EnsureTransitionScoreToGo();
  const odgi::nid_t min_id = graph_.min_node_id();
  const odgi::nid_t max_id = graph_.max_node_id();
  if (from < min_id || from > max_id) return 0.0;
  return transition_score_to_go_[static_cast<size_t>(from - min_id)];
}

double HaplotypeSamplerOverlay::PopulationScoreToGo(const PopulationState& state) const {
  if (!prior_) return 0.0;
  if (const auto* gbwt = std::get_if<PopulationState::Gbwt>(&state.tier)) {
    const double stay = prior_->ScoreToGo(gbwt->state);
    const double fall_back = (state.last_distinguishing_node != PopulationState::kNoNode)
        ? params_.panel_fallback_penalty + TransitionScoreToGo(state.last_distinguishing_node)
        : -std::numeric_limits<double>::infinity();
    return std::max(stay, fall_back);
  }
  return TransitionScoreToGo(state.last_distinguishing_node);
}

double HaplotypeSamplerOverlay::ApplyPopulationEdge(PopulationState& state, odgi::nid_t next_node,
                                                     bool distinguishes) const {
  if (!prior_) return 0.0;

  double delta = 0.0;
  if (auto* gbwt = std::get_if<PopulationState::Gbwt>(&state.tier)) {
    // Extend() fires once per edge, unconditionally.
    const auto extended = prior_->Extend(gbwt->state, next_node);
    if (!prior_->Empty(extended)) {
      delta = std::log(static_cast<double>(prior_->Width(extended)) /
                        static_cast<double>(prior_->Width(gbwt->state)));
      gbwt->state = extended;
    } else {
      // Pay the one-time fallback penalty on this edge, then score every *subsequent* edge using Markov model.
      delta = params_.panel_fallback_penalty;
      state.tier = PopulationState::FellBack{};
    }
  } else if (distinguishes && state.last_distinguishing_node != PopulationState::kNoNode) {
    delta = prior_->LogTransitionProbability(state.last_distinguishing_node, next_node,
                                             params_.transition_prior_alpha);
  }

  // Maintain last_distinguishing_node bookkeeping so any later fallback has a correct starting context
  if (distinguishes) state.last_distinguishing_node = next_node;

  // Guard the multiplication explicitly rather than relying on delta's own finiteness: PopulationScoreToGo
  // (a different function, ranking-only) can legitimately return -infinity, and 0.0 * -inf == NaN in
  // IEEE-754. This guard is what keeps an overlay-configured-but-weight-zero caller byte-identical to the
  // prior being fully disabled. Do not remove it as a "simplification".
  return params_.haplotype_prior_weight != 0.0 ? params_.haplotype_prior_weight * delta : 0.0;
}

HaplotypeSamplerOverlay::BestPathState HaplotypeSamplerOverlay::PropagateBestPathState(size_t n) const {
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

  // Population prior: active only when both an overlay and a nonzero weight are configured
  const bool population_prior_active = prior_ != nullptr && params_.haplotype_prior_weight != 0.0;
  auto rank_key = [&](const Backpointer& e) {
    double key = e.score;
    if (population_prior_active) key += params_.haplotype_prior_weight * PopulationScoreToGo(e.population_state);
    return key;
  };
  // A separate, typically-larger cap for the intermediate per-node trim only: n is frequently
  // deliberately small (SampleHaplotypes starts at samples.size()+2, FindBestPaths(1) passes 1), so without
  // widening, population-hypothesis diversity would have no room to compete with covered-paths diversity
  // for the same tiny slot count. Widens (never shrinks below the caller's requested n) only when the prior
  // is actually active; otherwise identical to n, so trimming stays byte-identical to the behavior without the prior.
  const size_t node_trim_cap = population_prior_active ? std::max(n, params_.population_state_pool_cap) : n;

  // Seed the source node: a virtual transition from the automaton's root consuming min_id itself, exactly
  // mirroring how every other node's arrival is processed below.
  {
    size_t seed_state = AutomatonGoto(kAutomatonRoot, min_id);
    Graph::PathIdSet seed_covered_paths(covered_paths_size);
    accumulate_covered_paths(seed_covered_paths, min_id);

    PopulationState seed_pop{};
    if (prior_) {
      // This is the only Find() call site in PropagateBestPathState. Every other use of the Gbwt tier below is
      // Extend()-only, via ApplyPopulationEdge. A stray Find() anywhere else silently degrades Width()/ScoreToGo() to
      // 0, not a crash. Do not add one.
      std::get<PopulationState::Gbwt>(seed_pop.tier).state = prior_->Find(min_id);
      if (population_distinguishing_mask_.test(min_id)) seed_pop.last_distinguishing_node = min_id;
    }

    dp[0][seed_state].push_back({
      min_score + KmerSetScoreDelta(seed_state),
      0,  // no predecessor node
      kAutomatonRoot, // irrelevant without predecessor node
      0, // irrelevant without predecessor node
      make_covered_paths(std::move(seed_covered_paths)),
      seed_pop
    });
  }

  for (odgi::nid_t i = min_id; i <= max_id; ++i) {
    if (!graph_.has_node(i)) continue;

    auto& node_state = dp[i - min_id];
    assert(!node_state.empty());  // Should have at least one path to every reachable node

    // Every pool at this node is width-trimmed to node_trim_cap distinct classes, ranked by rank_key (real
    // score, plus the heuristic population score-to-go when active). Dedup by (covered_paths,
    // population_state) only when the prior is actually active; otherwise population_state is real (Find/Extend
    // still run whenever prior_ is configured, regardless of weight -- see ApplyPopulationEdge) but
    // score-irrelevant, so classing by it alone would fragment node_trim_cap==n slots across population-state
    // variants of the same covered_paths and starve out genuinely distinct covered_paths candidates. Falling
    // back to SameCoveredPathsClass here keeps a weight-zero-but-prior-configured caller byte-identical to the
    // prior being fully disabled, matching node_trim_cap's own gating above.
    for (auto& [automaton_state, pool] : node_state) {
      if (population_prior_active) {
        SortAndTrimBacktrack(pool, node_trim_cap, /*sort_result=*/false, rank_key);
      } else {
        SortAndTrimBacktrack(pool, node_trim_cap, /*sort_result=*/false, rank_key,
                             [](const auto& a, const auto& b) { return SameCoveredPathsClass(a, b); });
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

      // Precompute the sparse set of path bits next_node itself contributes, once per edge (not once per
      // automaton_state/b_idx below): mirrors accumulate_covered_paths's masking exactly, and `contributes`
      // already guarantees inference_node_mask_.test(next_node) when apply_path_filter_ is set (see
      // InitializeContributesPathsMask). Reused both to extend covered_paths and, per b_idx below, to compute
      // the incremental hash delta (only the bits that are actually newly set for that b_idx's predecessor).
      Graph::PathIdSet contribution;
      if (contributes) {
        contribution = apply_path_filter_ ? (graph_.node_variant_paths_[next_node] & inference_path_mask_)
                                           : graph_.node_variant_paths_[next_node];
      }

      // Population-prior "distinguishing" is deliberately *not* the same flag as `contributes` above: it must stay
      // unfiltered by any active inference-VCF mask, since HaplotypePriorOverlay's Markov adjacency table was built
      // without knowledge of one.
      const bool distinguishes = population_distinguishing_mask_.test(next_node);

      for (auto& [automaton_state, pool] : node_state) {
        size_t new_state = AutomatonGoto(automaton_state, next_node);
        double weight_delta = KmerSetScoreDelta(new_state);
        auto& next_pool = next_node_state[new_state];
        for (size_t b_idx = 0; b_idx < pool.size(); ++b_idx) {
          SharedPathIdSet new_covered_paths;
          if (contributes) {
            const auto& parent = *pool[b_idx].covered_paths;
            size_t delta_hash = 0;
            for (auto idx = contribution.find_first(); idx != Graph::PathIdSet::npos; idx = contribution.find_next(idx)) {
              if (!parent.bits.test(idx)) delta_hash ^= PathHash(idx);
            }
            Graph::PathIdSet bits = parent.bits;
            bits |= contribution;
            new_covered_paths = SharedPathIdSet(new PathIdSetHolder(std::move(bits), parent.hash ^ delta_hash));
          } else {
            new_covered_paths = pool[b_idx].covered_paths; // unchanged -- share, don't copy
          }

          // Population-prior term is NOT hoistable like weight_delta above. Different b_idx entries sharing
          // (automaton_state, next_node) generally carry different PopulationStates (different GBWT intervals,
          // different last_distinguishing_node), so both the state transition and its score delta must be computed per
          // entry.
          PopulationState new_pop = pool[b_idx].population_state;
          double pop_delta = ApplyPopulationEdge(new_pop, next_node, distinguishes);

          next_pool.push_back({
            pool[b_idx].score + weight_delta + pop_delta,
            i, // predecessor node
            automaton_state, // predecessor automaton state
            b_idx, // path index in predecessor node's pool
            std::move(new_covered_paths),
            new_pop
          });
        }
      }
      return true;
    });
  }

  // At the sink, no k-mer match can ever resolve further, so the pending automaton state does not carry any
  // forward-looking information. Merge every automaton-state pool into one, deduplicating by covered_paths *alone*,
  // i.e., SameCoveredPathsClass, not the population-aware SamePoolClass used everywhere else, and trim to the final
  // top-n, i.e., same "settlement" treatment as a root state.
  //
  // This is NOT a no-op under an active inference-VCF filter (HAPLOTYPE_PRIOR_PROPOSAL.md finding #3):
  // population_distinguishing_mask_ is deliberately unfiltered, so two branches that diverged at a masked-out
  // (non-inference) variant can reach the sink with identical filtered covered_paths but genuinely different
  // population_state. Using SameCoveredPathsClass here on purpose preserves FindBestPaths's existing
  // output-distinctness contract instead of silently inflating output size whenever the prior happens to be active;
  // `.score`, a fully-accumulated value here, not a heuristic, still picks the population-aware best representative
  // within each covered_paths class. n (not node_trim_cap) is correct regardless, since node_trim_cap only exists to
  // give population-state diversity room during *intermediate* propagation, and the final output is deliberately not
  // keyed on population state at all.
  auto& sink_state = dp[max_id - min_id];
  StatePool merged;
  for (auto& [automaton_state, pool] : sink_state) {
    merged.insert(merged.end(), std::make_move_iterator(pool.begin()), std::make_move_iterator(pool.end()));
  }
  SortAndTrimBacktrack(
      merged, n, /*sort_result=*/true, [](const Backpointer& e) { return e.score; },
      [](const Backpointer& a, const Backpointer& b) { return SameCoveredPathsClass(a, b); }
  ); // feeds SampleHaplotypes's order-sensitive walk
  sink_state.clear();
  sink_state.emplace(kAutomatonRoot, std::move(merged));

  return dp;
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

  // Population-prior term, mirroring PropagateBestPathState's own per-edge accumulation via the same
  // ApplyPopulationEdge helper.
  if (prior_ && !haplotype.empty()) {
    PopulationState state{};
    // Here haplotype.front() is a "genuine walk start" so use `Find`.
    std::get<PopulationState::Gbwt>(state.tier).state = prior_->Find(haplotype.front());
    if (population_distinguishing_mask_.test(haplotype.front())) {
      state.last_distinguishing_node = haplotype.front();
    }

    for (size_t i = 1; i < haplotype.size(); ++i) {
      const bool distinguishes = population_distinguishing_mask_.test(haplotype[i]);
      score += ApplyPopulationEdge(state, haplotype[i], distinguishes);
    }
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
  // Each draw runs PropagateBestPathState as a single forward pass at exactly the requested width
  const bool profiling = HaplotypeProfilingEnabled();
  static size_t call_index = 0;  // single-threaded per PERFORMANCE_NOTES.md's existing thread-unsafe-counter assumption
  size_t this_call = profiling ? call_index++ : 0;
  RssSample call_start_rss = profiling ? CurrentRss() : RssSample{};
  if (profiling) {
    const odgi::nid_t min_id = graph_.min_node_id();
    const odgi::nid_t max_id = graph_.max_node_id();
    const size_t covered_paths_bits = graph_.node_variant_paths_[min_id].size();
    // A single PathIdSetHolder's boost::dynamic_bitset block storage alone (excludes the holder's own
    // hash/refcount/dynamic_bitset-object overhead) -- one of these is allocated per DP transition where
    // contributes_paths_mask_ is set (see PropagateBestPathState), shared thereafter via intrusive_ptr.
    const size_t covered_paths_bitset_bytes = ((covered_paths_bits + 63) / 64) * 8;
    fmt::print(stderr,
               "HAP_PROFILE call={} n={} stage=sizes graph_nodes={} automaton_states={} "
               "covered_paths_bits={} covered_paths_bitset_bytes={} rss_kb={} hwm_kb={}\n",
               this_call, n, max_id - min_id + 1, automaton_.size(), covered_paths_bits,
               covered_paths_bitset_bytes, call_start_rss.rss_kb, call_start_rss.hwm_kb);
  }

  std::vector<PathWithCoverage> samples;
  samples.reserve(n);

  const bool population_prior_active = prior_ != nullptr && params_.haplotype_prior_weight != 0.0;
  // Hard ceiling on how far a single draw's width is allowed to grow while searching for one more covering
  // haplotype under an active population prior (see the retry loop below), so a region that never yields one
  // can't widen indefinitely. Generous relative to n/pool_cap: in practice a genuine covering candidate should
  // surface long before this is reached. Unused (and irrelevant) when the prior is disabled.
  constexpr size_t kMaxWidthMultiplier = 8;
  const size_t max_width = kMaxWidthMultiplier * std::max(n, params_.population_state_pool_cap);

  while (samples.size() < n) {
    // Request more paths than already selected. Since we could select a path with no covered paths, we sample the
    // top (|selected|+2) paths to ensure we can find a new distinct path that covers at least one inference path.
    size_t width = samples.size() + 2;
    if (population_prior_active) {
      // PropagateBestPathState's intermediate beam (node_trim_cap) is clamped to at least
      // population_state_pool_cap regardless of width, so any width at or below that cap can only change the
      // *final* trim, never the search itself -- skip straight past that plateau instead of incrementing
      // through it one draw at a time.
      width = std::max(width, params_.population_state_pool_cap + 2);
    }

    BestPathState path_state;
    size_t back_idx = 0;
    size_t last_candidates_size = 0;
    bool found_new = false;
    Graph::PathIdSet adjusted_covered_paths;  // Set only when a new candidate is found below
    for (;;) {
      auto draw_start = ProfileClock::now();
      path_state = PropagateBestPathState(width);
      if (profiling) {
        RssSample rss = CurrentRss();
        fmt::print(stderr, "HAP_PROFILE call={} n={} width={} stage=fixed_width ms={:.3f} rss_kb={} hwm_kb={}\n",
                   this_call, n, width, ElapsedMs(draw_start), rss.rss_kb, rss.hwm_kb);
      }
      const auto& candidates = path_state.back().at(kAutomatonRoot);
      last_candidates_size = candidates.size();

      back_idx = candidates.size();
      for (size_t i = 0; i < candidates.size(); ++i) {
        const auto& candidate = candidates[i];
        // A haplotype that never explicitly selects a flagged allele for some inference variant still implicitly
        // declines it (i.e. carries the reference allele there), even if it doesn't explicitly traverse the reference
        // allele's own node(s). Compute once per candidate and use consistently for both the filter check and the dedup
        // comparison below, so two candidates that decline the same variant via different, otherwise-unflagged alleles
        // are correctly recognized as the same genotype call.
        auto adjusted = apply_path_filter_
            ? graph_.ApplyReferenceFallback(candidate.covered_paths->bits, inference_path_mask_)
            : candidate.covered_paths->bits;
        if (apply_path_filter_ && adjusted.none()) {
          continue; // Skip paths that do not cover (or implicitly decline) any inference variant
        }
        auto matching_sample = std::find_if(samples.begin(), samples.end(), [&](const PathWithCoverage& result) {
          return adjusted == result.second;
        });
        if (matching_sample == samples.end()) {
          back_idx = i;  // Found a new candidate that is not already in samples
          adjusted_covered_paths = std::move(adjusted);
          break;
        }
      }
      if (back_idx != candidates.size()) {
        found_new = true;
        break;
      }

      // No acceptable candidate at this width. Without the population prior, dedup is by covered_paths alone
      // and node_trim_cap == width has always meant "this many distinct covered_paths classes is already as
      // much diversity as this graph offers" in practice -- so preserve the original, cheap single-attempt
      // behavior here rather than paying for retries a large graph can make expensive.
      if (!population_prior_active) {
        break;
      }
      // With the prior active, a sink returning fewer entries than requested does NOT prove every distinct
      // (covered_paths, population_state) combination has been found: node_trim_cap (PropagateBestPathState's
      // intermediate beam) tracks width too, so a still-larger width can let upstream nodes retain combinations
      // that this width's narrower intermediate beam already pruned before they ever reached the sink -- i.e.
      // sink diversity can grow non-monotonically with width, not just saturate at some fixed graph-determined
      // value. So keep widening unconditionally up to max_width rather than trusting an under-full sink as
      // proof there is nothing left to find.
      if (width >= max_width) {
        break;  // Hit the search-width ceiling; give up for this draw.
      }
      width *= 2;
    }

    if (!found_new) {
      // Only log when the prior was active and the sink was still saturating (last_candidates_size >= width):
      // that's the case where giving up is a real loss (further widening was still finding new structure, we
      // just ran out of ceiling). The non-prior single-attempt path never logs -- it matches long-standing
      // behavior, not a new limitation worth flagging.
      if (population_prior_active && last_candidates_size >= width) {
        spdlog::info(
            "HaplotypeSamplerOverlay::SampleHaplotypes: {}:{}-{} reached the search-width ceiling ({}) with "
            "only {}/{} haplotypes sampled; additional distinct haplotypes may exist beyond this search width.",
            graph_.region().contig(), graph_.region().start(), graph_.region().end(), width, samples.size(), n);
      }
      break;
    }

    // Extract and save the path of interest and its covered path set
    samples.push_back(std::move(BacktrackPath(path_state, kAutomatonRoot, back_idx)));
    if (apply_path_filter_) {
      // Store the reference-fallback-adjusted bits (matching what the dedup comparison above used), not the
      // raw ones BacktrackPath returns, so later draws compare consistently.
      samples.back().second = std::move(adjusted_covered_paths);
    }
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
  Graph::PathIdSet covered_paths;
  if (!apply_path_filter_) {
    covered_paths = graph_.CoveredPaths(haplotype);
  } else {
    // Unlike the unfiltered case, only nodes that distinguish an *inference* allele (inference_node_mask_)
    // contribute -- and only their inference-masked bits -- so nodes that merely fall within a variant's
    // broader (non-distinguishing) reference span don't spuriously count. graph_.CoveredPaths() unions every
    // node unconditionally, so it isn't equivalent here.
    const size_t covered_paths_size = graph_.node_variant_paths_[haplotype.front()].size(); // All nodes should have the same size path sets
    covered_paths = Graph::PathIdSet(covered_paths_size);
    for (auto node_id : haplotype) {
      if (inference_node_mask_.test(node_id)) {
        covered_paths |= (graph_.node_variant_paths_[node_id] & inference_path_mask_);
      }
    }
    // A haplotype that never explicitly selects a flagged allele for some inference variant still implicitly
    // declines it (i.e. carries the reference allele there).
    covered_paths = graph_.ApplyReferenceFallback(covered_paths, inference_path_mask_);
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
