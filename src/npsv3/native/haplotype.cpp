#include "haplotype.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <numeric>
#include <unordered_set>

#include <boost/dynamic_bitset.hpp>
#include <fmt/std.h>
#include <fmt/ranges.h>

#include "variant.hpp"

namespace npsv3 {

HaplotypeSamplerOverlay::HaplotypeSamplerOverlay(
    const Graph& graph, const std::vector<std::string>& sequences,
    const std::vector<std::vector<UniqueKmersOverlay::KmerLocation>>& locations, const Params& params)
    : graph_(graph), params_(params), apply_path_filter_(false), kmer_sequences_(sequences) {
  // Maintain internal kmers_ in the same order as unique_kmers for consistent indexing.
  const size_t num_kmers = sequences.size();
  
  kmers_.reserve(num_kmers);
  for (size_t kmer_idx = 0; kmer_idx < num_kmers; ++kmer_idx) {
    const auto & kmer_locations = locations[kmer_idx];
    kmers_.push_back({ KmerZygosity::ABSENT, params_.absent_score }); // C++20 required for parenthesized initialization in emplace_back

    // Record k-mer presence on each "path" (sequence of node IDs). K-mers that span multiple nodes
    // are termed "explicit edges" or "shadow edges" that span all the handles in their path.
    for (const auto & [handles, offset] : kmer_locations) {
      KmerNodeIdSeq kmer_path(handles.size());
      std::transform(handles.begin(), handles.end(), kmer_path.begin(),
                     [&](const handlegraph::handle_t& handle) { return graph.get_id(handle); });
      auto [path_kmer_it, _] = path_kmers_.try_emplace(std::move(kmer_path), num_kmers);
      path_kmer_it->second.kmer_set_.set(kmer_idx);
    }
  }

  // Absorb k-mers of all sub-paths, including individuals nodes, into path kmers
  for (auto & [path, entry] : path_kmers_) {
    for (size_t start = 0; start < path.size(); ++start) {
      for (size_t end = start + 1; end <= path.size(); ++end) {
        if (start == 0 && end == path.size())
          continue; // Skip the full path itself
        auto sub_span = boost::span<const KmerNodeIdSeq::value_type>(path.data() + start, end - start);
        if (auto it = path_kmers_.find(sub_span); it != path_kmers_.end()) {
          entry.kmer_set_ |= it->second.kmer_set_;
        }
      }
    }
  }
}

HaplotypeSamplerOverlay::HaplotypeSamplerOverlay(const Graph& graph, const UniqueKmersOverlay& unique_kmers, const Params& params)
    : HaplotypeSamplerOverlay(graph, unique_kmers.sequences(), unique_kmers.locations(), params) {
  // Precompute intermediate path sets for k-mers that span multiple nodes.
  const size_t covered_paths_size = graph_.node_variant_paths_[graph_.min_node_id()].size();
  for (auto& [path, entry] : path_kmers_) {
    // This is only needed for edges with "intermediate" nodes, i.e., k-mers that span multiple nodes. 
    if (path.size() <= 2)
      continue;
    entry.intermediate_paths_ = Graph::PathIdSet(covered_paths_size);
    for (size_t i = 1, e = path.size() - 1; i < e; ++i) {
        entry.intermediate_paths_ |= graph_.node_variant_paths_[path[i]];
    }
  }
}

HaplotypeSamplerOverlay::HaplotypeSamplerOverlay(const Graph& graph, const UniqueKmersOverlay& unique_kmers,
                                                 const std::string& inference_vcf, const Range& region,
                                                 size_t min_size, const Params& params)
    : HaplotypeSamplerOverlay(graph, unique_kmers.sequences(), unique_kmers.locations(), params) {
  // Initialize inference VCF filtering
  apply_path_filter_ = true;
  graph.PopulateNodeAndPathMasks(inference_vcf, region, min_size, inference_node_mask_, inference_path_mask_);
  assert(inference_path_mask_.any());
  
  // Precompute intermediate path sets for k-mers that span multiple nodes accounting for inference filtering
  const size_t covered_paths_size = graph_.node_variant_paths_[graph_.min_node_id()].size();
  for (auto& [path, entry] : path_kmers_) {
    // This is only needed for edges with "intermediate" nodes, i.e., k-mers that span multiple nodes. 
    if (path.size() <= 2)
      continue;
    entry.intermediate_paths_ = Graph::PathIdSet(covered_paths_size);
    for (size_t i = 1, e = path.size() - 1; i < e; ++i) {
      if (inference_node_mask_.test(path[i])) {
        entry.intermediate_paths_ |= graph_.node_variant_paths_[path[i]];
      }
    }
    entry.intermediate_paths_ &= inference_path_mask_;
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
  template<typename T>
  void SortAndTrimBacktrack(T& backtrack, size_t n, bool dedup_covered_paths = true) {
    if (dedup_covered_paths && !backtrack.empty()) {
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

    size_t new_size = std::min(n, backtrack.size());
    std::partial_sort(backtrack.begin(), backtrack.begin() + new_size, backtrack.end(),
        [](const auto& a, const auto& b) { return a.score > b.score; });
    backtrack.resize(new_size);
  }
}

std::vector<HaplotypeSamplerOverlay::Haplotype> HaplotypeSamplerOverlay::FindBestPaths(size_t n) const {
  if (n == 0) return {};

  const odgi::nid_t min_id = graph_.min_node_id();
  const odgi::nid_t max_id = graph_.max_node_id();
  const size_t covered_paths_size = graph_.node_variant_paths_[min_id].size(); // All nodes should have the same size of path sets

  // Minimum possible score is if none of the k-mers are present in the haplotype, i.e., pH(x) = −1
  double min_score = std::accumulate(kmers_.begin(), kmers_.end(), 0.0, [](double acc, const KmerScore& kmer) {
    return acc - kmer.score;
  });

  // dp[v - min_id] holds up to n backpointer entries for distinct paths from min_id to v.
  // pred_path_idx indexes into dp[pred_node - min_id], which is frozen before propagation
  // so the indices remain stable.
  struct Backpointer {
    double score;
    odgi::nid_t pred_node;
    PathKmerMap::const_iterator edge_it;
    size_t pred_path_idx;
    Graph::PathIdSet covered_paths;
  };
  std::vector<std::vector<Backpointer>> dp(max_id - min_id + 1);

  // Seed the source node.
  dp[0].push_back({ 
    min_score,
    0,  // no predecessor node
    path_kmers_.end(), // no explicit edge 
    0, // irrelevant without predecessor node
    Graph::PathIdSet(covered_paths_size)
  });

  for (odgi::nid_t i = min_id; i <= max_id; ++i) {
    if (!graph_.has_node(i)) continue;

    auto& back_pointers = dp[i - min_id];
    assert(!back_pointers.empty());  // Should have at least one path to every reachable node

    // Accumulate inference paths for node i before trimming/deduplicating, so we can use covered_paths as the equivalence key.
    if (!apply_path_filter_) {
      const auto & node_paths = graph_.node_variant_paths_[i];
      for (auto& back : back_pointers)
        back.covered_paths |= node_paths;
    } else if (inference_node_mask_.test(i)) {
      auto node_paths = graph_.node_variant_paths_[i] & inference_path_mask_;
      for (auto& back : back_pointers)
        back.covered_paths |= node_paths;
    }

    // Trim to top 'n' before propagating. After this point dp[i] is frozen: we only push entries
    // into successor lists, never back into dp[i], so pred_path_idx values are stable.
    SortAndTrimBacktrack(back_pointers, n);

    // Find all k-mers starting at this node
    odgi::nid_t i_next = i + 1;
    auto path_kmer_it = path_kmers_.lower_bound(boost::span<const KmerNodeIdSeq::value_type>(&i, 1));
    auto path_kmer_end = path_kmers_.lower_bound(boost::span<const KmerNodeIdSeq::value_type>(&i_next, 1));

    // The first key in order will the individual node if it exists. If so, credit this node's own k-mers.
    if (path_kmer_it != path_kmer_end && path_kmer_it->first.size() == 1 && path_kmer_it->first.front() == i) {
      double node_score_delta = 0.0;
      const auto& kmer_set = path_kmer_it->second.kmer_set_;
      for (size_t kmer_idx = kmer_set.find_first(); kmer_idx != KmerIdSet::npos; kmer_idx = kmer_set.find_next(kmer_idx)) {
        node_score_delta += 2.0 * kmers_[kmer_idx].score;
      }
      for (auto& back : back_pointers)
        back.score += node_score_delta;
      
      ++path_kmer_it; // Move to the next path key, which will be the first sub-path of length > 1
    }

    // Propagate along real forward edges without explicit edge counterparts
    graph_.follow_edges(graph_.get_handle(i), false /* forward */, [&](const handlegraph::handle_t& next) {
      auto next_node = graph_.get_id(next);
      const odgi::nid_t edge_arr[2] = {i, next_node};
      if (path_kmers_.find(boost::span<const odgi::nid_t>(edge_arr, 2)) != path_kmers_.end()) {
        // Use the "explicit" edge that exists for i -> next_node (handled below)
        return true;
      }
      auto& next_back_pointers = dp[next_node - min_id];
      for (size_t b_idx = 0; b_idx < back_pointers.size(); ++b_idx) {
        next_back_pointers.push_back({
          back_pointers[b_idx].score, 
          i, // predecessor node
          path_kmers_.end(), // no explicit edge
          b_idx, // path index in predecessor node
          back_pointers[b_idx].covered_paths
        });
      }
      return true;
    });

     // Propagate along explicit edges starting at node i
    for (; path_kmer_it != path_kmer_end; ++path_kmer_it) {
      assert(path_kmer_it->first.size() > 1 && path_kmer_it->first.front() == i); // These must span multiple nodes

      double edge_score_delta = 0.0;
      const auto& kmer_set = path_kmer_it->second.kmer_set_;
      for (size_t kmer_idx = kmer_set.find_first(); kmer_idx != KmerIdSet::npos; kmer_idx = kmer_set.find_next(kmer_idx)) {
        edge_score_delta += 2.0 * kmers_[kmer_idx].score;
      }

      const auto& edge_intermediate_paths = path_kmer_it->second.intermediate_paths_;
      
      auto& next_back_pointers = dp[path_kmer_it->first.back() - min_id];
      for (size_t b_idx = 0; b_idx < back_pointers.size(); ++b_idx) {
        next_back_pointers.push_back({
          back_pointers[b_idx].score + edge_score_delta, 
          i, // predecessor node
          path_kmer_it, // explicit edge
          b_idx, // path index in predecessor node
          back_pointers[b_idx].covered_paths
        });
        if (!edge_intermediate_paths.empty()) {
          next_back_pointers.back().covered_paths |= edge_intermediate_paths;
        }
      }
    }
  }

  // Sort and trim the final node's backpointers.
  auto& final_backtrack = dp[max_id - min_id];
  SortAndTrimBacktrack(final_backtrack, n);

  // Backtrack from max_id to min_id for each of the (up to n) best paths.
  std::vector<Haplotype> result;
  result.reserve(final_backtrack.size());
  for (size_t back_idx = 0; back_idx < final_backtrack.size(); ++back_idx) {
    // When inference filtering is active, skip paths that don't traverse any inference nodes/paths.
    if (apply_path_filter_ && final_backtrack[back_idx].covered_paths.none()) {
      continue;
    }

    Haplotype path;
    odgi::nid_t current_node = max_id;
    size_t current_back_idx = back_idx;

    while (current_node != min_id) {
      path.push_back(current_node);
      const auto& backpointer = dp[current_node - min_id][current_back_idx];
      if (backpointer.edge_it != path_kmers_.end()) {
        // Populate the intermediate nodes of the explicit edge in reverse order
        const auto & edge_path = backpointer.edge_it->first; 
        assert(edge_path.front() == backpointer.pred_node && edge_path.back() == current_node);
        std::reverse_copy(edge_path.begin() + 1, edge_path.end() - 1, std::back_inserter(path));
      }
      current_back_idx = backpointer.pred_path_idx; assert(current_back_idx < n);
      current_node = backpointer.pred_node; assert(current_node >= min_id && current_node <= max_id);
    }
    path.push_back(current_node); // Include source node

    std::reverse(path.begin(), path.end());
    result.push_back(std::move(path));
  }
  return result;
}

HaplotypeSamplerOverlay::KmerIdSet HaplotypeSamplerOverlay::KmersOnPath(const Haplotype& path) const {
  KmerIdSet on_path(kmers_.size());
  for (auto path_it = path.begin(), path_end = path.end(); path_it != path_end; ++path_it) {
    auto prefix = boost::span<const KmerNodeIdSeq::value_type>(&(*path_it), std::distance(path_it, path_end));
    
    // Find the longest matching prefix which occurs just previous of the upper bound
    // since the map is sorted.
    auto match_it = path_kmers_.upper_bound(prefix);
    if (match_it == path_kmers_.begin()) {
      continue; // No prefix match found at all so advance to the next node in the path
    }
    match_it = std::prev(match_it);

    const auto & match_key = match_it->first;
    auto [path_mismatch_it, key_mismatch_it] = std::mismatch(path_it, path_end, match_key.begin(), match_key.end());
    if (key_mismatch_it == match_key.begin()) {
      continue; // No matching prefix found so advance to the next node in the path
    } else if (key_mismatch_it != match_key.end()) {
      // Requery with the longest matching prefix, which occurs at the point of divergence within the original query.
      auto trimmed = prefix.subspan(0, std::distance(path_it, path_mismatch_it));
      match_it = path_kmers_.find(trimmed);
      assert(match_it != path_kmers_.end());
    }
      
    // Update on_path from the k-mers in the longest matching key
    on_path |= match_it->second.kmer_set_;
  }
  return on_path;
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

std::vector<HaplotypeSamplerOverlay::Haplotype> HaplotypeSamplerOverlay::SampleHaplotypes(size_t n) {
  std::vector<Haplotype> results;
  results.reserve(n);

  for (size_t i = 0; i < n; ++i) {
    // Request one more path than already selected: by pigeonhole, the best
    // unselected path (if any) must appear within the top (|selected|+1).
    auto paths = FindBestPaths(results.size() + 1);

    // Find the first path not already in results.
    const Haplotype* new_path = nullptr;
    for (const auto& path : paths) {
      if (std::find(results.begin(), results.end(), path) == results.end()) {
        new_path = &path;
        break;
      }
    }
    if (!new_path) break;  // no more distinct paths in the graph

    results.push_back(*new_path);
    UpdateScores(results.back());
  }
  return results;
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
