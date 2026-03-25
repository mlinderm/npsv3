#include "kmer.hpp"

#include <algorithm>
#include <map>
#include <stdexcept>

namespace npsv3 {

namespace {

/// Estimate haploid coverage from a KMC count histogram (Sirén et al. §2.1).
///
/// Algorithm:
///   1. Find the primary peak p = argmax(histogram[count >= 2]).
///   2. If p > weighted median of the non-singleton distribution → coverage = p
///      (unimodal distribution; the homozygous peak dominates).
///   3. Otherwise search for a secondary local maximum in [1.5·p, 2.5·p] whose
///      frequency is at least half the primary-peak frequency (a "good enough"
///      secondary peak that corresponds to the homozygous peak when the het peak
///      is primary). If found → coverage = secondary peak count.
///   4. If both steps fail, throw CoverageEstimationError.
double EstimateCoverageFromHistogram(const std::map<uint32_t, uint64_t>& histogram) {
  if (histogram.empty()) {
    throw CoverageEstimationError(
        "k-mer count histogram is empty; cannot auto-estimate coverage");
  }

  // Total non-singleton k-mers and the primary peak.
  uint64_t total = 0;
  uint32_t mode_count = 0;
  uint64_t mode_freq = 0;

  for (const auto& [cnt, freq] : histogram) {
    total += freq;
    if (freq > mode_freq) {
      mode_freq = freq;
      mode_count = cnt;
    }
  }

  // Weighted median: the count value at which cumulative weight reaches 50%.
  uint64_t cumulative = 0;
  uint32_t median_count = 0;
  for (const auto& [cnt, freq] : histogram) {  // std::map iterates in ascending key order
    cumulative += freq;
    if (2 * cumulative >= total) {
      median_count = cnt;
      break;
    }
  }

  // Step 2: unimodal check. "most common count exceeds the median" (Sirén et al. §2.1).
  // Use >= so that a perfectly unimodal distribution (mode == median) also succeeds.
  if (mode_count >= median_count) {
    return static_cast<double>(mode_count);
  }

  // Step 3: look for a secondary peak at approximately 2x the primary peak.
  const uint32_t search_lo = static_cast<uint32_t>(1.5 * mode_count);
  const uint32_t search_hi = static_cast<uint32_t>(2.5 * mode_count);

  uint32_t secondary_count = 0;
  uint64_t secondary_freq = 0;

  // Find the local maximum in the search range — a count whose frequency
  // exceeds both its immediate neighbours in the histogram.
  uint32_t prev_cnt = 0;
  uint64_t prev_freq = 0;
  for (auto it = histogram.lower_bound(search_lo); it != histogram.end() && it->first <= search_hi; ++it) {
    auto next_it = std::next(it);
    const uint64_t next_freq = (next_it != histogram.end() && next_it->first <= search_hi)
                                   ? next_it->second
                                   : 0;
    if (it->second >= prev_freq && it->second >= next_freq && it->second > secondary_freq) {
      secondary_freq = it->second;
      secondary_count = it->first;
    }
    prev_cnt = it->first;
    prev_freq = it->second;
  }

  // "Good enough" = secondary peak frequency is at least half the primary peak frequency.
  if (secondary_count > 0 && secondary_freq * 2 >= mode_freq) {
    return static_cast<double>(secondary_count);
  }

  throw CoverageEstimationError(
      "Cannot auto-estimate coverage from k-mer count distribution; "
      "please supply KmerCountsParams::coverage explicitly");
}

}  // namespace

// ---------------------------------------------------------------------------

double KmerCounts::EstimateCoverage(const std::string& db_path) {
  CKMCFile db;
  if (!db.OpenForListing(db_path)) {
    throw std::runtime_error("Cannot open KMC database for listing: " + db_path);
  }

  const uint32_t k = db.KmerLength();
  CKmerAPI kmer(k);
  uint32_t count = 0;

  // Build histogram, skipping singletons (dominated by sequencing errors).
  std::map<uint32_t, uint64_t> histogram;
  while (db.ReadNextKmer(kmer, count)) {
    if (count > 1) {
      histogram[count]++;
    }
  }
  db.Close();

  return EstimateCoverageFromHistogram(histogram);
}

// ---------------------------------------------------------------------------

KmerCounts::KmerCounts(const std::string& db_path, const KmerCounts::Params& params) {
  // Auto-estimate coverage before opening in RA mode (requires a listing pass).
  if (params.coverage <= 0.0) {
    coverage_ = EstimateCoverage(db_path);
  } else {
    coverage_ = params.coverage;
  }

  // Compute coverage thresholds from the coverage estimate and the specified cutoffs
  assert(params.absent_fraction < params.heterozygous_fraction && params.heterozygous_fraction < params.homozygous_fraction);
  absent_coverage_ = params.absent_fraction * coverage_;
  heterozygous_coverage_ = params.heterozygous_fraction * coverage_;
  homozygous_coverage_ = params.homozygous_fraction * coverage_;

  if (!db_.OpenForRA(db_path)) {
    throw std::runtime_error("Cannot open KMC database for random access: " + db_path);
  }

  // GetBothStrands() returns true when KMC was run WITHOUT the -b flag, i.e. all k-mers are stored in canonical form. 
  canonicalize_ = db_.GetBothStrands();
}

KmerCounts::~KmerCounts() {
  db_.Close();
}

KmerZygosity KmerCounts::Classify(const std::string& seq) const {
  const uint32_t k = db_.KmerLength();
  CKmerAPI kmer(k);

  if (!kmer.from_string(seq)) {
    // Treat k-mers with invalid characters as absent
    return KmerZygosity::ABSENT;
  }

  if (canonicalize_) {
    // KMC only stores the canonical (lexicographically smaller) k-mer. Canonicalize the query k-mer before lookup.
    CKmerAPI rev(kmer);
    rev.reverse();
    if (rev < kmer) {
      kmer = rev;
    }
  }

  uint32_t count = 0;
  auto found = db_.CheckKmer(kmer, count);  // returns false (count=0) if k-mer absent
  if (!found || count < absent_coverage_) {
    return KmerZygosity::ABSENT;
  } else if (count < heterozygous_coverage_) {
    return KmerZygosity::HETEROZYGOUS;
  } else if (count < homozygous_coverage_) {
    return KmerZygosity::HOMOZYGOUS;
  } else {
    return KmerZygosity::FREQUENT;
  }
}

}  // namespace npsv3
