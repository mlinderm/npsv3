#include "kmer.hpp"

#include <algorithm>
#include <numeric>
#include <fstream>
#include <map>
#include <stdexcept>
#include <utility>

#include <boost/scope/scope_exit.hpp>
#include <boost/dynamic_bitset.hpp>
#include <handlegraph/util.hpp>

namespace npsv3 {

namespace {
  // Map from node_id to the [start, end) range within that node covered by a k-mer.
using NodeCoverageMap = std::unordered_map<handlegraph::handle_t, std::pair<size_t, size_t>>;

/**
 * @brief Compute the coverage of a k-mer within each node of its handle path.
 *
 * @param graph The graph to query for node sequences.
 * @param handles The handle path of the k-mer occurrence, in order.
 * @param start_offset The offset within the first handle where the k-mer starts.
 * @param k The length of the k-mer.
 *
 * @return A map from handle to the half-open [start, end) range within that node covered by the k-mer. Zero-length
 * (deletion) nodes are included with the sentinel value {0, 0}.
 */
NodeCoverageMap ComputeKmerCoverage(const handlegraph::HandleGraph& graph,
                                    const std::vector<handlegraph::handle_t>& handles,
                                    uint64_t start_offset, size_t k) {
  NodeCoverageMap coverage;
  for (size_t i = 0, remaining = k; i < handles.size() && remaining > 0; ++i) {
    auto seq = graph.get_sequence(handles[i]);
    size_t seq_length = seq.size();
    if (seq_length == 0) {
      // Deletion node: Record traversal with sentinel {0,0} so it participates
      // in the common-intersection check across occurrences.
      coverage[handles[i]] = {0, 0};
      continue;
    }
    size_t node_start = (i == 0) ? static_cast<size_t>(start_offset) : 0;
    assert(node_start < seq_length); // node_start should always be within the node sequence
    size_t covered = std::min(remaining, seq_length - node_start);
    if (covered > 0) {
      coverage[handles[i]] = {node_start, node_start + covered};
      remaining -= covered;
    }
  }
  return coverage;
}


/**
 * @brief Intersect k-mer coverage maps in place
 */
void IntersectCoverage(NodeCoverageMap& current, const NodeCoverageMap& other) {
  for (auto it = current.begin(); it != current.end();) {
    auto jt = other.find(it->first);
    if (jt == other.end()) {
      it = current.erase(it);
    } else if (it->second == NodeCoverageMap::mapped_type({0, 0}) &&
               jt->second == NodeCoverageMap::mapped_type({0, 0})) {
      // Both occurrences traversed the same deletion node
      ++it;
    } else {
      const auto& [start1, end1] = it->second;
      const auto& [start2, end2] = jt->second;
      size_t new_start = std::max(start1, start2);
      size_t new_end = std::min(end1, end2);
      if (new_start >= new_end) {
        it = current.erase(it);
      } else {
        it->second = {new_start, new_end};
        ++it;
      }
    }
  }
}

template<typename T>
void apply_permutation_in_place(const std::vector<size_t>& perm, std::vector<T>& vec) {
  // https://stackoverflow.com/a/17074810
  assert(perm.size() == vec.size());
  boost::dynamic_bitset<> done(perm.size(), 0);
  for (size_t i = 0; i < vec.size(); ++i) {
    if (done.test(i)) {
        continue;
    }
    done.set(i);
    std::size_t prev_j = i;
    std::size_t j = perm[i];
    while (i != j) {
      std::swap(vec[prev_j], vec[j]);
      done.set(j);
      prev_j = j;
      j = perm[j];
    }
  }
}

std::string CanonicalKmer(const std::string& seq) {
  std::string rc = odgi::reverse_complement(seq);
  return rc < seq ? rc : seq;
}

} // anonymous namespace

UniqueKmersOverlay::UniqueKmersOverlay(const Graph& graph, size_t k, size_t max_edges, bool exclude_universal, bool canonicalize) : graph_(graph), k_(k) {
  struct Location : public KmerLocation {
    NodeCoverageMap coverage_;  // running intersection across all occurrences seen so far
  };
  std::unordered_map<std::string, Location> seen;
  std::unordered_set<std::string> non_unique;

  graph_.Kmers(k_, max_edges, [&](const std::string& seq, const std::vector<handlegraph::handle_t>& handles, uint64_t offset) {
    // When canonicalize is set, forward and reverse-complement occurrences share one key so
    // their coverage maps are intersected together, just like multiple forward occurrences are.
    // TODO: Optimize canonicalization to not create copy when not canonicalizing or when the reverse complement is greater than the forward sequence.
    std::string key = canonicalize ? CanonicalKmer(seq) : seq;
    if (non_unique.count(key)) return;

    // A k-mer is **graph-unique** if all occurrences of its sequence in the graph share a non-empty coverage intersection
    // of at least one node — i.e., every occurrence traverses the same portion of some common node. This allows the k-mer
    // to start or end in different nodes across haplotypes (e.g., when multiple predecessor paths converge into a shared
    // node), as long as all occurrences pass through a shared node segment.

    auto curr_coverage = ComputeKmerCoverage(graph_, handles, offset, k);
    auto it = seen.find(key);
    if (it == seen.end()) {
      seen.emplace(key, Location{ handles, offset, std::move(curr_coverage) });
    } else {
      auto& entry = it->second;
      IntersectCoverage(entry.coverage_, curr_coverage);
      if (entry.coverage_.empty()) {
        non_unique.insert(key);
        seen.erase(it);
      }
    }
  });

  // Collect survivors applying universal filter if requested.
  sequences_.reserve(seen.size());
  locations_.reserve(seen.size());
  for (auto it = seen.begin(); it != seen.end(); ) {
    auto node = seen.extract(it++); // Extract invalidates iterator, so post-increment before processing
    if (exclude_universal) {
      // A k-mer is "universal" if all nodes uniquely covered by this k-mer are not part of any variants and thus must
      // appear on any traversal of the graph.
      bool is_universal = std::all_of(node.mapped().coverage_.begin(), node.mapped().coverage_.end(),
        [&](const auto& kv) {
          auto node_id = graph_.get_id(kv.first);
          assert(node_id < graph_.node_variant_paths_.size());
          return graph_.node_variant_paths_[node_id].none();
        });
      
      if (is_universal) continue;
    }
    sequences_.push_back(std::move(node.key()));
    locations_.push_back(std::move(static_cast<KmerLocation>(node.mapped())));
  }
  sequences_.shrink_to_fit();
  locations_.shrink_to_fit();

  // Sort the sequences by creating a permutation and applying it to both vectors
  {
    std::vector<size_t> perm(sequences_.size());
    std::iota(perm.begin(), perm.end(), 0);
    std::sort(perm.begin(), perm.end(), [&](size_t i, size_t j) {
      return sequences_[i] < sequences_[j];
    });
    apply_permutation_in_place(perm, sequences_);
    apply_permutation_in_place(perm, locations_);
  }
}

void UniqueKmersOverlay::SaveFasta(const std::string& fasta_path) const {
  // TODO: Write compressed FASTA
  std::ofstream fasta(fasta_path);
  for (size_t i = 0; i < sequences_.size(); ++i) {
    fasta << ">" << i << "\n";
    fasta << sequences_[i] << "\n";
  }
}

KmerCounts::KmerCounts(const std::string& db_path, double coverage, const KmerCounts::Params& params)
    : db_path_(db_path) {
  assert(params.absent_fraction < params.heterozygous_fraction &&
         params.heterozygous_fraction < params.homozygous_fraction);
  absent_coverage_ = params.absent_fraction * coverage;
  heterozygous_coverage_ = params.heterozygous_fraction * coverage;
  homozygous_coverage_ = params.homozygous_fraction * coverage;
}

// ---------------------------------------------------------------------------

KmerZygosity KmerCounts::ClassifyCount(uint32_t count) const {
  if (count < absent_coverage_) {
    return KmerZygosity::ABSENT;
  } else if (count < heterozygous_coverage_) {
    return KmerZygosity::HETEROZYGOUS;
  } else if (count < homozygous_coverage_) {
    return KmerZygosity::HOMOZYGOUS;
  } else {
    return KmerZygosity::FREQUENT;
  }
}

void KmerCounts::ClassifySorted(const std::vector<std::string>& sequences, const std::function<void(size_t idx, KmerZygosity zyg)>& callback) const {
  CKMCFile db;
  if (!db.OpenForListing(db_path_)) {
    throw std::runtime_error("Cannot open KMC database for listing: " + db_path_);
  }
  auto cleanup = boost::scope::make_scope_exit([&db] {
    db.Close();
  });

  auto k = db.KmerLength();

  // Track which indices had a DB "hit", emitting ABSENT for the rest after the loop. We can't emit
  // ABSENT inline because the DB may only be locally sorted, not globally sorted.
  boost::dynamic_bitset<> matched(sequences.size(), 0);

  size_t pos = 0;
  std::string prev_kmer_str;

  CKmerAPI kmer(k);
  uint32_t count = 0;
  while (db.ReadNextKmer(kmer, count)) {
    std::string kmer_str = kmer.to_string();

    // Detect a locally-sorted chunk boundary and reset the merge position.
    if (kmer_str < prev_kmer_str) {
      pos = std::distance(sequences.begin(), std::lower_bound(sequences.begin(), sequences.begin() + pos, kmer_str));
    }

    // Advance past entries that sort before this k-mer, then emit any matches
    while (pos < sequences.size() && sequences[pos] < kmer_str) {
      ++pos;
    }
    for (; pos < sequences.size() && sequences[pos] == kmer_str; ++pos) {
      matched[pos] = true;
      callback(pos, ClassifyCount(count));
    }

    prev_kmer_str = std::move(kmer_str);
  }

  for (size_t i = 0; i < sequences.size(); ++i) {
    if (!matched.test(i)) {
      callback(i, KmerZygosity::ABSENT);
    }
  }
}

}  // namespace npsv3
