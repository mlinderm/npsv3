#include "kmer.hpp"

#include <algorithm>
#include <numeric>
#include <fstream>
#include <map>
#include <stdexcept>
#include <utility>

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/scope/scope_exit.hpp>
#include <boost/dynamic_bitset.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/vector.hpp>
#include <handlegraph/util.hpp>
#include <fmt/format.h>

namespace boost {
namespace serialization {

template <class Archive>
void serialize(Archive& ar, handlegraph::handle_t& h, unsigned int) {
  ar & handlegraph::as_integer(h);
}

template <class Archive>
void serialize(Archive& ar, npsv3::UniqueKmersOverlay::KmerLocation& loc, unsigned int) {
  ar & loc.handles_;
  ar & loc.starting_handle_offset_;
}

}  // namespace serialization
}  // namespace boost

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
void ApplyPermutationInPlace(const std::vector<size_t>& perm, std::vector<T>& vec) {
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

bool UniqueKmersOverlay::KmerLocation::operator==(const KmerLocation& other) const {
  return handles_ == other.handles_ && starting_handle_offset_ == other.starting_handle_offset_;
}

UniqueKmersOverlay::UniqueKmersOverlay(const Graph& graph, size_t k, size_t max_edges, bool exclude_universal, bool canonicalize, const KmerCounts* ref_kmer_counts) : graph_(graph), k_(k) {
  // Precompute the forward reachability of all nodes in the graph to determine whether two k-mer occurrences 
  // can co-occur on the same graph traversal.
  auto reachable = graph_.ForwardReachability();

  // A k-mer is **graph-unique** if no two of its occurrences can appear on the same graph
  // traversal. Each callback invocation from Kmers() describes one occurrence by its physical
  // start position (handle, offset). Two occurrences with the same (handle, offset)
  // are the same graph position with via different path continuations and so cannot
  // co-occur. Two occurrences with distinct start positions can co-occur when:
  //   (a) they are in the same node at different offsets (both always present on any
  //       traversal through that node), or
  //   (b) their start nodes are reachable from one another (one follows the other on some
  //       path through the graph).
  std::unordered_map<std::string, std::vector<KmerLocation>> seen;
  std::unordered_set<std::string> non_unique;

  // Construct a set of reference k-mers in this graph if filtering by reference
  std::unordered_set<std::string> graph_ref;
  if (ref_kmer_counts) {
    auto ref_seq = graph_.PathSequence(graph_.region_.contig());
    for (size_t i = 0; i + k <= ref_seq.size(); ++i) {
      graph_ref.insert(ref_seq.substr(i, k));
    }
  }

  graph_.Kmers(k_, max_edges, [&](const std::string& seq, const std::vector<handlegraph::handle_t>& handles, uint64_t offset) {
    // When canonicalize is set, forward and reverse-complement occurrences share one key.
    // TODO: Optimize canonicalization to not create copy when not canonicalizing or when the reverse complement is
    // greater than the forward sequence.
    std::string key = canonicalize ? CanonicalKmer(seq) : seq;
    if (non_unique.count(key)) return;

    auto it = seen.find(key);
    if (ref_kmer_counts && it == seen.end()) {
      // Reference k-mers that appear twice in the reference genome can't be graph-unique and a non-reference k-mer that appears
      // in the reference outside this region can't be graph-unique either.
      const uint32_t threshold = graph_ref.count(key) ? 2u : 1u;
      if (ref_kmer_counts->Count(key) >= threshold) {
        non_unique.insert(key);
        return;
      }
    }

    auto new_location = KmerLocation{ handles, offset };
    if (it == seen.end()) {
      seen.emplace(key, std::vector{ new_location });
    } else {
      auto& existing_locations = it->second;
      auto new_location_starting_node_id = graph_.get_id(new_location.handles_[0]);
      bool conflict = false;
      for (const auto& [existing_handles, existing_offset] : existing_locations) {
        if (existing_handles[0] == new_location.handles_[0]) {
          // k-mer starts in the same node as an existing instance. If the have the same offset, then they are at the same position
          // (i.e., having different path continuations) and can't "co-occur" on some traversal of the graph. If they have different
          // offsets, then they are both present on any traversal of that node and thus can co-occur.
          conflict = existing_offset != new_location.starting_handle_offset_;
          break;
        } else if (auto existing_starting_node_id = graph_.get_id(existing_handles[0]);
                   reachable[new_location_starting_node_id].test(existing_starting_node_id) ||
                   reachable[existing_starting_node_id].test(new_location_starting_node_id)) {
          // k-mer starts in a different node, but can reach or is reachable from another instance, and thus could co-occur on
          // some traversal of the graph.
          conflict = true;
          break;
        }
      }
      if (conflict) {
        non_unique.insert(key);
        seen.erase(it);
      } else {
        existing_locations.push_back(new_location);
      }
    }
  });

  // Collect survivors applying universal filter if requested.
  sequences_.reserve(seen.size());
  for (auto it = seen.begin(); it != seen.end(); ) {
    auto node = seen.extract(it++); // Extract invalidates iterator, so post-increment before processing
    auto& locations = node.mapped();
    if (exclude_universal) {
      // A k-mer is "universal" when it is present on every graph traversal. We can detect this when the intersection of all locations
      // for a k-mer include nodes not associated with any variant path.
      NodeCoverageMap common_coverage = ComputeKmerCoverage(graph_, locations[0].handles_, locations[0].starting_handle_offset_, k_);
      for (size_t i = 1; i < locations.size(); ++i) {
        NodeCoverageMap other_coverage = ComputeKmerCoverage(graph_, locations[i].handles_, locations[i].starting_handle_offset_, k_);
        IntersectCoverage(common_coverage, other_coverage);
      }

      bool is_universal = !common_coverage.empty() &&
          std::all_of(common_coverage.begin(), common_coverage.end(), [this](NodeCoverageMap::const_reference entry) {
            auto node_id = graph_.get_id(entry.first);
            assert(static_cast<size_t>(node_id) < graph_.node_variant_paths_.size());
            return graph_.node_variant_paths_[node_id].none();
          });
      if (is_universal) continue;
    }
    sequences_.push_back(std::move(node.key()));
    locations_.push_back(std::move(locations));
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
    ApplyPermutationInPlace(perm, sequences_);
    ApplyPermutationInPlace(perm, locations_);
  }
}

UniqueKmersOverlay::UniqueKmersOverlay(const Graph& graph, size_t k,
                                       std::vector<std::string> sequences,
                                       std::vector<std::vector<KmerLocation>> locations)
    : graph_(graph), k_(k),
      sequences_(std::move(sequences)),
      locations_(std::move(locations)) {}

void UniqueKmersOverlay::Save(std::ostream& out) const {
  boost::archive::binary_oarchive oa(out);
  oa << k_ << sequences_ << locations_;
}

void UniqueKmersOverlay::Save(const std::string& path) const {
  std::ofstream out(path, std::ios::binary);
  if (!out) throw std::runtime_error("Cannot open k-mer file for writing: " + path);
  Save(out);
}

void UniqueKmersOverlay::Load(UniqueKmersOverlay* target, const Graph& graph, std::istream& in) {
  boost::archive::binary_iarchive ia(in);
  size_t k;
  std::vector<std::string> sequences;
  std::vector<std::vector<KmerLocation>> locations;
  ia >> k >> sequences >> locations;
  new (target) UniqueKmersOverlay(graph, k, std::move(sequences), std::move(locations));
}

void UniqueKmersOverlay::Load(UniqueKmersOverlay* target, const Graph& graph,
                               const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("Cannot open k-mer file: " + path);
  Load(target, graph, in);
}

void UniqueKmersOverlay::SaveFasta(const std::string& fasta_path) const {
  // TODO: Write compressed FASTA
  std::ofstream fasta(fasta_path);
  for (size_t i = 0; i < sequences_.size(); ++i) {
    fasta << ">" << i << "\n";
    fasta << sequences_[i] << "\n";
  }
}

KmerClassify::KmerClassify(const std::string& db_path, double coverage, const KmerClassify::Params& params)
    : db_path_(db_path) {
  assert(params.absent_fraction < params.heterozygous_fraction &&
         params.heterozygous_fraction < params.homozygous_fraction);
  absent_coverage_ = params.absent_fraction * coverage;
  heterozygous_coverage_ = params.heterozygous_fraction * coverage;
  homozygous_coverage_ = params.homozygous_fraction * coverage;
}

// ---------------------------------------------------------------------------

KmerZygosity KmerClassify::ClassifyCount(uint32_t count) const {
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

void KmerClassify::ClassifySorted(const std::vector<std::string>& sequences, const std::function<void(size_t idx, KmerZygosity zyg)>& callback) const {
  CKMCFile db;
  if (!db.OpenForListing(db_path_)) {
    throw std::runtime_error(fmt::format("Cannot open KMC database for listing: {}", db_path_));
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

KmerCounts::KmerCounts(const std::string& db_path) {
  if (!kmc_file_.OpenForRA(db_path)) {
    throw std::runtime_error(fmt::format("Cannot open KMC database for random access: {}", db_path));
  }
}

KmerCounts::~KmerCounts() {
  kmc_file_.Close();
}

uint32_t KmerCounts::Count(const std::string& kmer) const {
  CKmerAPI kmer_api(kmc_file_.KmerLength());
  uint32_t count = 0;
  if (kmer_api.from_string(kmer)) {
    kmc_file_.CheckKmer(kmer_api, count);
  }
  return count;
}

}  // namespace npsv3
