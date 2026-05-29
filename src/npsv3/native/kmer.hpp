#pragma once

#include <functional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <kmc_api/kmc_file.h>

#include "graph.hpp"

namespace npsv3 {

class UniqueKmersOverlay {
 public:
  struct KmerLocation {
    std::vector<handlegraph::handle_t> handles_;
    uint64_t starting_handle_offset_;
  };

  UniqueKmersOverlay(const Graph& graph, size_t k, size_t max_edges, bool exclude_universal=true, bool canonicalize=false);

  size_t size() const { return sequences_.size(); }

  const std::vector<std::string>& sequences() const { return sequences_; }
  const std::vector<KmerLocation>& locations() const { return locations_; }


  void SaveFasta(const std::string& fasta_path) const;

 private:
  const Graph& graph_;
  const size_t k_;

  std::vector<std::string> sequences_;
  std::vector<KmerLocation> locations_;
};


/// Zygosity classification for a k-mer in a sequencing sample
enum class KmerZygosity { ABSENT, HETEROZYGOUS, HOMOZYGOUS, FREQUENT };


/**
 * @brief Classifies k-mer sequences as ABSENT / HETEROZYGOUS / HOMOZYGOUS using
 *        a pre-built KMC database (Sirén et al. §2.1).
 */
class KmerCounts {
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
   * @param coverage Haploid coverage estimate (must be > 0).
   * @param params Coverage and threshold parameters.
   * @throws std::runtime_error If the database cannot be opened.
   * @throws std::invalid_argument If params.coverage <= 0.
   */
  explicit KmerCounts(const std::string& db_path, double coverage, const Params& params = {});

  virtual ~KmerCounts() = default;


  /**
   * @brief Classify k-mers in the globally sorted @p sequences using a merge-like pass of the k-mer DB, 
   * calling @p callback with the index into @p sequences and zygosity.
   * 
   * The callback may be called in any order.
   */
  virtual void ClassifySorted(
      const std::vector<std::string>& sequences,
      const ClassificationCallback& callback) const;

 protected:
  /// Protected default constructor for subclasses that override ClassifyBatch without a KMC database.
  KmerCounts() : absent_coverage_(0.0), heterozygous_coverage_(0.0), homozygous_coverage_(0.0) {}

 private:
  std::string db_path_;
  
  double absent_coverage_;
  double heterozygous_coverage_;
  double homozygous_coverage_;

  KmerZygosity ClassifyCount(uint32_t count) const;
};

}  // namespace npsv3
