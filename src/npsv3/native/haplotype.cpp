#include "haplotype.hpp"

#include <algorithm>
#include <cassert>
#include <limits>
#include <numeric>

#include "variant.hpp"

namespace npsv3 {

HaplotypeSamplerOverlay::HaplotypeSamplerOverlay(const Graph& graph, size_t k, size_t max_edge,
                                                 const KmerCounts& counts, const Params& params)
    : graph_(graph), params_(params), apply_path_filter_(false) {

  graph.UniqueKmers(
      k, max_edge,
      [&](const std::string& seq, const std::vector<handlegraph::handle_t>& handles, [[maybe_unused]] uint64_t offset) {
        auto zyg = counts.Classify(seq);
        double initial_score = params_.absent_score;
        switch (zyg) {
          default:
            return;  // skip FREQUENT or otherwise unknown k-mers
          case KmerZygosity::HOMOZYGOUS:
            initial_score = params_.homozygous_score;
            break;
          case KmerZygosity::HETEROZYGOUS:
            initial_score = params_.heterozygous_score;
            break;
          case KmerZygosity::ABSENT:
            break;
        }

        // Need to wait for C++20 for parenthesized initialization in emplace_back
        auto kmer_idx = kmers_.size();
        kmers_.push_back({seq, zyg, initial_score});

        // Record k-mer presence. K-mers within a node are associated with that node. K-mers that span multiple nodes
        // are associated with explicit (shadow) edges spanning all the handles in their path.
        if (handles.size() == 1) {
          node_kmers_[graph.get_id(handles.front())].push_back(kmer_idx);
        } else {
          // Find edge, insert if not found
          auto& edges = edge_kmers_[Edge{graph.get_id(handles.front()), graph.get_id(handles.back())}];
          auto it = std::find_if(edges.begin(), edges.end(), [&](const EdgeInfo& e) {
            // A matching node must have the same nodes (the first and last are already checked by the map key).
            return std::equal(e.intermediate_nodes.begin(), e.intermediate_nodes.end(), handles.begin() + 1, handles.end() - 1, [&](odgi::nid_t node_id, const handlegraph::handle_t& handle) {
              return node_id == graph.get_id(handle);
            });
          });

          if (it == edges.end()) {
            // No existing edges match this handle path, create a new one
            EdgeInfo new_edge_info{ {}, {kmer_idx} };

            new_edge_info.intermediate_nodes.reserve(handles.size() - 2);
            std::transform(handles.begin() + 1, handles.end() - 1, std::back_inserter(new_edge_info.intermediate_nodes),
                           [&](const handlegraph::handle_t& handle) { return graph.get_id(handle); });

            edges.push_back(std::move(new_edge_info));
          } else {
            // Existing edge matches this handle path, add k-mer index to it
            it->kmers.push_back(kmer_idx);
          }
        }
      },
      true /* exclude universal k-mers */);

  // Absorb intermediate-node k-mers into each explicit edge's k-mer list.
  for (auto& [_, edges] : edge_kmers_) {
    for (auto& edge : edges) {
      for (auto& intermediate_node : edge.intermediate_nodes) {
        if (auto it = node_kmers_.find(intermediate_node); it != node_kmers_.end())
          edge.kmers.insert(edge.kmers.end(), it->second.begin(), it->second.end());
      }
    }
  }
}

HaplotypeSamplerOverlay::HaplotypeSamplerOverlay(const Graph& graph, size_t k, size_t max_edge,
                                                 const KmerCounts& counts,
                                                 const std::string& inference_vcf,
                                                 const Range& region,
                                                 size_t min_size,
                                                 const Params& params)
    : HaplotypeSamplerOverlay(graph, k, max_edge, counts, params) {
  apply_path_filter_ = true;
  graph.PopulateNodeAndPathMasks(inference_vcf, region, min_size, inference_node_mask_, inference_path_mask_);
  assert(inference_path_mask_.any()); // Should have at least one path to track, otherwise no paths will be selected
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

std::vector<Graph::NodeIdSeq> HaplotypeSamplerOverlay::FindBestPaths(size_t n) const {
  if (n == 0) return {};

  const odgi::nid_t min_id = graph_.min_node_id();
  const odgi::nid_t max_id = graph_.max_node_id();
  const size_t covered_paths_size = graph_.node_variant_paths_[min_id].size(); // All nodes should have the same size of path sets

  // Minimum possible score is if none of the k-mers are present in the haplotype, i.e., pH(x) = −1
  double min_score = std::accumulate(kmers_.begin(), kmers_.end(), 0.0, [](double acc, const KmerInfo& kmer) {
    return acc - kmer.score;
  });

  // dp[v - min_id] holds up to n backpointer entries for distinct paths from min_id to v.
  // pred_path_idx indexes into dp[pred_node - min_id], which is frozen before propagation
  // so the indices remain stable.
  struct Backpointer {
    double score = -std::numeric_limits<double>::infinity();
    odgi::nid_t pred_node = 0;
    int edge_idx = -1; // -1 = real edge; >= 0 = index into edge_kmers_[{pred_node, cur}]
    size_t pred_path_idx = std::numeric_limits<size_t>::max(); // index into dp[pred_node - min_id]
    Graph::PathIdSet covered_paths = {}; // inference allele coverage; empty (size 0) when filtering is off
  };
  std::vector<std::vector<Backpointer>> dp(max_id - min_id + 1);

  // Seed the source node.
  dp[0].push_back({ min_score });
  dp[0].back().covered_paths.resize(covered_paths_size);

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

    { // Credit this node's own k-mers: switching pH(x) from -1 to +1 adds 2*score per k-mer.
      double node_score_delta = 0.0;
      if (auto it = node_kmers_.find(i); it != node_kmers_.end()) {
          for (auto idx : it->second)
            node_score_delta += 2.0 * kmers_[idx].score;
      }
      for (auto& back : back_pointers)
        back.score += node_score_delta;
    }

    // Propagate along real forward edges without explicit edge counterparts
    graph_.follow_edges(graph_.get_handle(i), false /* forward */, [&](const handlegraph::handle_t& next) {
      auto next_node = graph_.get_id(next);
      if (auto it = edge_kmers_.find(Edge{i, next_node});
          it != edge_kmers_.end() && std::any_of(it->second.begin(), it->second.end(), [](const EdgeInfo& edge) {
            return edge.intermediate_nodes.empty();
          })) {
        // Explicit edge exists for this implicit edge (i -> next_node, w/ no intermediate nodes), use that explicit
        // edge instead.
        return true;
      }
      auto& next_back_pointers = dp[next_node - min_id];
      for (size_t b_idx = 0; b_idx < back_pointers.size(); ++b_idx) {
        next_back_pointers.push_back({back_pointers[b_idx].score, i /* predecessor node */, -1 /* no explicit edge */,
                                      b_idx /* path index in predecessor node */, back_pointers[b_idx].covered_paths});
      }
      return true;
    });

    // Propagate along explicit edges starting at i; each EdgeInfo is a distinct path option.
    for (auto it = edge_kmers_.lower_bound(Edge{i, min_id}), end = edge_kmers_.upper_bound(Edge{i, max_id}); it != end;
         ++it) {
      const auto& edges = it->second;
      for (size_t e_idx = 0; e_idx < edges.size(); ++e_idx) {
        const double edge_delta =
            std::accumulate(edges[e_idx].kmers.begin(), edges[e_idx].kmers.end(), 0.0,
                            [&](double acc, size_t k_idx) { return acc + 2.0 * kmers_[k_idx].score; });

        // Accumulate inference paths for intermediate nodes of this explicit edge.
        Graph::PathIdSet edge_intermediate_paths(covered_paths_size);
        if (!apply_path_filter_) {
          for (auto nid : edges[e_idx].intermediate_nodes) {
            edge_intermediate_paths |= graph_.node_variant_paths_[nid];
          }
        } else {
          for (auto nid : edges[e_idx].intermediate_nodes) {
            if (inference_node_mask_.test(nid))
              edge_intermediate_paths |= graph_.node_variant_paths_[nid];
          }
          edge_intermediate_paths &= inference_path_mask_;
        }

        auto& next_back_pointers = dp[it->first.to - min_id];
        for (size_t b_idx = 0; b_idx < back_pointers.size(); ++b_idx) {
          // Graph::PathIdSet combined = back_pointers[b_idx].covered_paths;
          // if (filtering) {
          //   combined |= edge_intermediate_paths;
          // }
          next_back_pointers.push_back({back_pointers[b_idx].score + edge_delta, i /* predecessor node */,
                                        static_cast<int>(e_idx) /* explicit edge */,
                                        b_idx /* path index in predecessor node */,
                                        back_pointers[b_idx].covered_paths | edge_intermediate_paths});
        }
      }
    }
  }

  // Sort and trim the final node's backpointers.
  auto& final_backtrack = dp[max_id - min_id];
  SortAndTrimBacktrack(final_backtrack, n);

  // Backtrack from max_id to min_id for each of the (up to n) best paths.
  std::vector<Graph::NodeIdSeq> result;
  result.reserve(final_backtrack.size());
  for (size_t back_idx = 0; back_idx < final_backtrack.size(); ++back_idx) {
    // When inference filtering is active, skip paths that don't traverse any inference nodes/paths.
    if (apply_path_filter_ && final_backtrack[back_idx].covered_paths.none()) {
      continue;
    }

    Graph::NodeIdSeq path;
    odgi::nid_t current_node = max_id;
    size_t current_back_idx = back_idx;

    while (current_node != min_id) {
      path.push_back(current_node);
      const auto& backpointer = dp[current_node - min_id][current_back_idx];
      if (backpointer.edge_idx >= 0) {
        // Expand explicit edge's intermediate nodes in reverse order.
        const auto& edge = edge_kmers_.at(Edge{backpointer.pred_node, current_node}).at(backpointer.edge_idx);
        for (auto nit = edge.intermediate_nodes.rbegin(); nit != edge.intermediate_nodes.rend(); ++nit)
          path.push_back(*nit);
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


void HaplotypeSamplerOverlay::UpdateScores(const Graph::NodeIdSeq& path) {
  const odgi::nid_t min_id = graph_.min_node_id();
  const odgi::nid_t max_id = graph_.max_node_id();
  std::unordered_set<size_t> on_path;

  for (auto path_it = path.begin(), path_end = path.end(); path_it != path_end;) {
    // Collect single-node k-mer indices for each node on the path.
    auto nid = *path_it;
    if (auto k_it = node_kmers_.find(nid); k_it != node_kmers_.end()) {
      const auto & kmer_indices = k_it->second;
      on_path.insert(kmer_indices.begin(), kmer_indices.end());
    }
    // Collect edge k-mer indices for explicit edges on the path
    bool found_edge = false;
    for (auto e_it = edge_kmers_.lower_bound(Edge{nid, min_id}), e_end = edge_kmers_.upper_bound(Edge{nid, max_id});
         e_it != e_end && !found_edge; ++e_it) {
      const auto& edges = e_it->second;
      for (const auto& edge : edges) {
        // Does the nodes on this explicit edge match the next nodes on the path?
        auto [edge_it, past_edge_it] =
            std::mismatch(edge.intermediate_nodes.begin(), edge.intermediate_nodes.end(), path_it + 1, path.end());
        if (edge_it == edge.intermediate_nodes.end() && *past_edge_it == e_it->first.to) {
          on_path.insert(edge.kmers.begin(), edge.kmers.end());
          path_it = past_edge_it;
          found_edge = true;
          break;  // Only one edge can match since they can't share intermediate nodes
        }
      }
    }
    if (!found_edge) ++path_it;
  }

  // Apply score adjustments based on zygosity.
  for (size_t i = 0; i < kmers_.size(); ++i) {
    auto& km = kmers_[i];
    switch (km.zygosity) {
      default:
        break;
      case KmerZygosity::HOMOZYGOUS:
        // "If x is a homozygous k-mer and x ∈ H, we discount its score by a multiplicative factor: w(x) ≔ 0.9 × w(x)."
        if (on_path.count(i) > 0) km.score *= params_.homozygous_discount;
        break;
      case KmerZygosity::HETEROZYGOUS:
        // "if x is a heterozygous k-mer, we adjust its score by an additive term to make the opposite outcome more likely: w(x) ≔ w(x) − 0.05 × pH(x)."
        km.score -= ((on_path.count(i) > 0) ? params_.het_adjustment : -params_.het_adjustment);  
        break;
    }
  }
}

std::vector<Graph::NodeIdSeq> HaplotypeSamplerOverlay::SampleHaplotypes(size_t n) {
  std::vector<Graph::NodeIdSeq> results;
  results.reserve(n);

  for (size_t i = 0; i < n; ++i) {
    // Request one more path than already selected: by pigeonhole, the best
    // unselected path (if any) must appear within the top (|selected|+1).
    auto paths = FindBestPaths(results.size() + 1);

    // Find the first path not already in results.
    const Graph::NodeIdSeq* new_path = nullptr;
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

}  // namespace npsv3
