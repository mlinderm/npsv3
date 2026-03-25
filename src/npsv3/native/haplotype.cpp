#include "haplotype.hpp"

#include <numeric>

namespace npsv3 {

HaplotypeSamplerOverlay::HaplotypeSamplerOverlay(const Graph& graph, size_t k, size_t max_edge,
                                                 const KmerCounts& counts)
    : HaplotypeSamplerOverlay(graph, k, max_edge, counts, Params{}) {}

HaplotypeSamplerOverlay::HaplotypeSamplerOverlay(const Graph& graph, size_t k, size_t max_edge,
                                                 const KmerCounts& counts, Params params)
    : graph_(graph), params_(params) {

  graph.UniqueKmers(
      k, max_edge,
      [&](const std::string& seq, const std::vector<handlegraph::handle_t>& handles, uint64_t offset) {
        auto zyg = counts.Classify(seq);
        double initial_score = params_.absent_score;
        switch (zyg) {
          default:
            return;  // skip FREQUENT or otherwise unknown k-mers, they don't contribute to haplotype distinction
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
        // are associated with (shadow) edges spanning all the handles in their path.
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
      true /* exclude 'universal' k-mers in each haplotype */);

  // Fill in k-mer indices for intermediate nodes in all explicit edges
  for (auto& [_, edges] : edge_kmers_) {
    for (auto& edge : edges) {
      for (auto & intermediate_node : edge.intermediate_nodes) {
        if (auto it = node_kmers_.find(intermediate_node); it != node_kmers_.end()) {
          edge.kmers.insert(edge.kmers.end(), it->second.begin(), it->second.end());
        }
      }
    }
  }
}

Graph::NodeIdSeq HaplotypeSamplerOverlay::FindBestPath() const {
  const odgi::nid_t min_id = graph_.min_node_id();
  const odgi::nid_t max_id = graph_.max_node_id();

  // Minimum possible score is if none of the k-mers are present in the haplotype, i.e., pH(x) = −1
  double min_score = std::accumulate(kmers_.begin(), kmers_.end(), 0.0, [](double acc, const KmerInfo& kmer) {
    return acc - kmer.score;
  });

  struct DPState {
    double score;
    odgi::nid_t pred_node = 0;
    int edge_idx = -1; // If pred_node is an explicit edge, index in edge_kmers_ value
  };
  std::vector<DPState> dp(max_id - min_id + 1, DPState{min_score});

  // Forward pass in topological, i.e., ascending node ID, order. 
  for (odgi::nid_t i = min_id; i <= max_id; ++i) {
    if (!graph_.has_node(i)) continue;
    auto& state = dp[i - min_id];

    // Credit this node's own (single-node) k-mers, since we are switching pH(x) = −1 to pH(x) = 1, we
    // add 2*score.
    if (auto it = node_kmers_.find(i); it != node_kmers_.end()) {
      for (auto idx : it->second)
        state.score += 2 * kmers_[idx].score;
    }

    // Propagate along real forward edges without explicit edge counterparts
    graph_.follow_edges(graph_.get_handle(i), false /* forward */, [&](const handlegraph::handle_t& next) {
      auto next_node = graph_.get_id(next);
      if (auto it = edge_kmers_.find(Edge{i, next_node});
          it != edge_kmers_.end() && std::any_of(it->second.begin(), it->second.end(), [](const EdgeInfo& edge) {
            return edge.intermediate_nodes.empty();
          })) {
        return true;  // Explicit edge exists, skip implicit edge
      }

      // >= to ensure we update the predecessor for ties, which can occur when edges don't have k-mers and thus don't
      // change the score
      if (auto& next_state = dp[graph_.get_id(next) - min_id]; state.score >= next_state.score) {
        next_state.score = state.score;
        next_state.pred_node = i;
      }
      return true;
    });

    // Propagate along explicit edges that start at node i
    for (auto it = edge_kmers_.lower_bound(Edge{i, min_id}), end = edge_kmers_.upper_bound(Edge{i, max_id}); it != end; ++it) {
      assert(it->first.from == i);
      const auto& edges = it->second;
      for (size_t e_idx = 0; e_idx < edges.size(); ++e_idx) {
        const auto& edge = edges[e_idx];
        double score = std::accumulate(edge.kmers.begin(), edge.kmers.end(), state.score, [&](double acc, size_t k_idx) {
          return acc + 2 * kmers_[k_idx].score;
        });
        
        if (auto & next_state = dp[it->first.to - min_id]; score >= next_state.score) {
          next_state.score = score;
          next_state.pred_node = i;
          next_state.edge_idx = e_idx;
        }
      }
    }
  }

  // Backtrack from max_id to min_id, inserting intermediate nodes for shadow-edge predecessors. 
  Graph::NodeIdSeq path;
  odgi::nid_t current = max_id;
  while (current != min_id) {
    const auto& state = dp[current - min_id]; 
    
    path.push_back(current);
    if ( state.edge_idx >= 0) {
      // Came via explict edge with nodes = [h1,...,h_{n-1}], add h_* in reverse order
      const auto& edge = edge_kmers_.at(Edge{state.pred_node, current}).at(state.edge_idx);
      for (auto nit = edge.intermediate_nodes.rbegin(); nit != edge.intermediate_nodes.rend(); ++nit) {
        path.push_back(*nit);
      }
    }

    current = state.pred_node;
  }
  path.push_back(current); // Add the starting node

  std::reverse(path.begin(), path.end());
  return path;
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
        // Does the nodes on this shadow edge match the next nodes on the path?
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
  std::vector<Graph::NodeIdSeq> result;
  result.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    auto path = FindBestPath();
    result.push_back(path);
    UpdateScores(path);
  }
  return result;
}

}