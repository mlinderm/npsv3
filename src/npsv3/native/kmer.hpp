#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <kmc_api/kmc_file.h>

#include "graph.hpp"

namespace npsv3 {

class KmerCounts;

class UniqueKmersOverlay {
 public:
  struct KmerLocation {
    std::vector<handlegraph::handle_t> handles_;
    uint64_t starting_handle_offset_;

    bool operator==(const KmerLocation&) const;
  };

  UniqueKmersOverlay(const Graph& graph, size_t k, size_t max_edges, bool exclude_universal=true, bool canonicalize=false, const KmerCounts* ref_kmer_counts = nullptr);

  size_t size() const { return sequences_.size(); }

  const std::vector<std::string>& sequences() const { return sequences_; }
  const std::vector<std::vector<KmerLocation>>& locations() const { return locations_; }

  void SaveFasta(const std::string& fasta_path) const;

  /** @brief Serialize k-mer sequences and locations (as node IDs) to an output stream */
  void Save(std::ostream& out) const;
  /** @brief Serialize k-mer sequences and locations (as node IDs) to a binary file */
  void Save(const std::string& path) const;

  /**
   * @brief Deserialize into a pre-allocated UniqueKmersOverlay via placement new.
   *
   * Handles are reconstructed from stored node IDs using @p graph.
   * Use as the target of a nanobind __init__ overload so that the graph
   * reference lifetime is managed by keep_alive.
   */
  static void Load(UniqueKmersOverlay* target, const Graph& graph, std::istream& in);
  static void Load(UniqueKmersOverlay* target, const Graph& graph, const std::string& path);

 private:
  // Used by Load() to construct from pre-built sequences and locations.
  UniqueKmersOverlay(const Graph& graph, size_t k,
                     std::vector<std::string> sequences,
                     std::vector<std::vector<KmerLocation>> locations);

  const Graph& graph_;
  const size_t k_;

  std::vector<std::string> sequences_;

  /// All recorded locations (possible multiple) for each k-mer, in the same order as @p sequences_.
  std::vector<std::vector<KmerLocation>> locations_;
};


/// Zygosity classification for a k-mer in a sequencing sample
enum class KmerZygosity { ABSENT, HETEROZYGOUS, HOMOZYGOUS, FREQUENT };


namespace detail {

/**
 * @brief A CKMCFile whose random-access mode memory-maps *.kmc_suf read-only instead of
 * reading it into a private heap buffer to minimize aggregate memory usage with multiple
 * instances.
 */
class MMapKMCFile : public CKMCFile {
 public:
  MMapKMCFile() = default;
  ~MMapKMCFile();

  MMapKMCFile(const MMapKMCFile&) = delete;
  MMapKMCFile& operator=(const MMapKMCFile&) = delete;

  bool OpenForRA(const std::string& file_name);
  bool Close();

 private:
  void UnmapSufix();

  void* sufix_mmap_base_ = nullptr;
  size_t sufix_mmap_size_ = 0;
};

} // namespace detail

/**
 * @brief Report k-mer count from a pre-built KMC database using random access.
 */
class KmerCounts {
 public:
  explicit KmerCounts(const std::string& db_path);
  ~KmerCounts();

  uint32_t k() const { return kmc_file_.KmerLength(); }

  uint64_t Count(const std::string& kmer) const;

 private:
  mutable detail::MMapKMCFile kmc_file_;
};


/**
 * @brief Classifies k-mer sequences as ABSENT / HETEROZYGOUS / HOMOZYGOUS using
 *        a pre-built KMC database (Sirén et al. §2.1).
 */
class KmerClassify {
 public:
  typedef std::function<void(size_t idx, KmerZygosity zyg)> ClassificationCallback;

  /// Parameters controlling zygosity classification from k-mer counts.
  struct Params {
    double absent_fraction;
    double heterozygous_fraction;
    double homozygous_fraction;

    Params(double absent_fraction_a = 0.1, double heterozygous_fraction_a = 1.0 / std::log(4),
           double homozygous_fraction_a = 2.5)
        : absent_fraction(absent_fraction_a),
          heterozygous_fraction(heterozygous_fraction_a),
          homozygous_fraction(homozygous_fraction_a) {}
  };

  /**
   * @param db_path Path to a KMC database (without suffixes).
   * @param coverage K-mer coverage estimate (must be > 0).
   * @param params Coverage threshold parameters.
   * @throws std::runtime_error If the database cannot be opened.
   * @throws std::invalid_argument If params.coverage <= 0.
   */
  explicit KmerClassify(const std::string& db_path, double coverage, const Params& params = {});

  virtual ~KmerClassify();

  /**
   * @brief Classify k-mers in the globally sorted @p sequences against the k-mer DB, calling @p callback
   * with the index into @p sequences and zygosity.
   *
   * Dispatches per call between a linear merge pass over the whole DB (when @p sequences is comparable to
   * or larger than the DB) and per-k-mer random-access lookups (when @p sequences is much smaller than the DB).
   *
   * The callback may be called in any order. Not safe to call concurrently from multiple threads on the
   * same instance.
   */
  virtual void ClassifySorted(const std::vector<std::string>& sequences, const ClassificationCallback& callback) const;

 protected:
  /// Protected default constructor for subclasses that override ClassifySorted without a KMC database.
  KmerClassify() : absent_coverage_(0.0), heterozygous_coverage_(0.0), homozygous_coverage_(0.0) {}

 private:
  std::string db_path_;

  double absent_coverage_;
  double heterozygous_coverage_;
  double homozygous_coverage_;

  /// Opened for listing at construction (real subclass only) and re-opened before every scan-based
  /// ClassifySorted call. Re-opening (rather than CKMCFile::RestartListing()) is required for correctness:
  /// RestartListing() does not reliably reset the listing after a completed pass in this KMC build, so a
  /// repeated scan would silently return nearly all-ABSENT (see ClassifySortedScan). The scan path only runs
  /// when the DB is small, so the per-call re-open of the *.kmc_pre LUT is cheap relative to the scan.
  mutable CKMCFile db_;

  /// Random-access handle, opened lazily on the first ClassifySorted call whose `sequences` is smaller than
  /// the DB and reused thereafter. Left null by the protected default constructor.
  mutable std::unique_ptr<detail::MMapKMCFile> ra_db_;

  KmerZygosity ClassifyCount(uint32_t count) const;

  /// Linear merge pass over the whole DB via db_ (listing mode) -- ClassifySorted's original strategy,
  /// still the best choice when `sequences` is comparable to or larger than the DB.
  void ClassifySortedScan(const std::vector<std::string>& sequences, const ClassificationCallback& callback) const;
  /// Per-k-mer random-access lookup via ra_db_ (opened lazily on first use) -- ClassifySorted's strategy
  /// when `sequences` is much smaller than the DB.
  void ClassifySortedRandomAccess(const std::vector<std::string>& sequences, const ClassificationCallback& callback) const;
};

/**
 * @brief Classifies every k-mer with a single fixed zygosity, requiring no KMC database.
 *
 * Useful for tests and experiments that need a KmerClassify without building a real k-mer count database
 * on disk -- e.g. ConstantKmerClassify(KmerZygosity::ABSENT) makes every graph-unique k-mer ABSENT, which
 * (since the k-mer term is then a fixed function of the graph rather than of any query sample's reads)
 * isolates other scoring contributions such as the population prior.
 */
class ConstantKmerClassify : public KmerClassify {
 public:
  explicit ConstantKmerClassify(KmerZygosity zygosity = KmerZygosity::HOMOZYGOUS) : zygosity_(zygosity) {}

  void ClassifySorted(const std::vector<std::string>& sequences,
                      const ClassificationCallback& callback) const override {
    for (size_t i = 0; i < sequences.size(); ++i) {
      callback(i, zygosity_);
    }
  }

 private:
  KmerZygosity zygosity_;
};

}  // namespace npsv3
