#pragma once

#include <stdexcept>
#include <string>

#include <kmc_api/kmc_file.h>

namespace npsv3 {

/// Zygosity classification for a k-mer in a sequencing sample
enum class KmerZygosity { ABSENT, HETEROZYGOUS, HOMOZYGOUS, FREQUENT };

/// Exception thrown when coverage cannot be auto-estimated.
struct CoverageEstimationError : std::runtime_error {
  explicit CoverageEstimationError(const std::string& msg) : std::runtime_error(msg) {}
};

/**
 * @brief Classifies k-mer sequences as ABSENT / HETEROZYGOUS / HOMOZYGOUS using
 *        a KMC k-mer count database (Sirén et al. §2.1).
 *
 * Opened in random-access mode for efficient single-k-mer queries. If
 * CoverageParams::coverage is 0, haploid coverage is auto-estimated from the
 * k-mer count histogram (requires a second listing pass at construction).
 *
 * K-mer queries are canonicalised automatically when the database was built in
 * canonical mode (the KMC default, i.e. without the -b flag).
 */
class KmerCounts {
 public:
  /// Parameters controlling zygosity classification from k-mer counts.
  struct Params {
    double coverage;  ///< Coverage estimate (or 0 to auto-estimate from histogram) as described in Sirén et al.
    double absent_fraction;
    double heterozygous_fraction;
    double homozygous_fraction;

    // Need an explicit constructor to indicate nested type is default constructible
    Params(double coverage_a = 0.0, double absent_fraction_a = 0.1, double heterozygous_fraction_a = 1.0 / std::log(4),
           double homozygous_fraction_a = 2.5)
        : coverage(coverage_a),
          absent_fraction(absent_fraction_a),
          heterozygous_fraction(heterozygous_fraction_a),
          homozygous_fraction(homozygous_fraction_a) {}
  };

  /**
   * @param db_path  Path to KMC database without extension (i.e. without .kmc_pre / .kmc_suf).
   * @param params   Coverage and threshold parameters. Set params.coverage to 0 to auto-estimate.
   * @throws std::runtime_error if the database cannot be opened.
   * @throws CoverageEstimationError if auto-estimation fails and params.coverage == 0.
   */
  explicit KmerCounts(const std::string& db_path, const Params& params = {});

  virtual ~KmerCounts();

  // Not copyable (CKMCFile manages FILE* resources).
  KmerCounts(const KmerCounts&) = delete;
  KmerCounts& operator=(const KmerCounts&) = delete;

  KmerCounts(KmerCounts&&) = default;
  KmerCounts& operator=(KmerCounts&&) = default;

  /**
   * @brief Classify a k-mer sequence.
   *
   * Canonicalises @p seq before lookup when the database is in canonical mode.
   * Sequences with invalid characters (not A/C/G/T) return ABSENT.
   */
  virtual KmerZygosity Classify(const std::string& seq) const;

  /// Estimated k-mer coverage (set at construction from params or auto-estimated).
  double coverage() const { return coverage_; }

 protected:
  /// Protected default constructor for subclasses that override Classify without a KMC database.
  KmerCounts() : coverage_(0.0), absent_coverage_(0.0), heterozygous_coverage_(0.0),
                 homozygous_coverage_(0.0), canonicalize_(false) {}

 private:
  mutable CKMCFile db_; ///< KMC random-access database handle (many KMC methods are logically const, but not marked as such)

  double coverage_;
  double absent_coverage_;
  double heterozygous_coverage_;
  double homozygous_coverage_;

  bool canonicalize_; ///< True if database only stores canonical k-mers (i.e., built without -b)

  /// Estimate k-mer coverage from the k-mer count histogram.
  static double EstimateCoverage(const std::string& db_path);
};

}  // namespace npsv3
