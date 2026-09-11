#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace npsv3 {

/**
 * @brief Per-base ref/alt vote arrays produced by scanning one read against a KmerEvidenceIndex.
 *
 * Both arrays are in query (read) coordinates, one entry per base of the scanned sequence, and are
 * produced by accumulating the full per-allele support vector over the read and then reducing it the
 * way the BWA realignment path already does: `ref_votes[j] = support[0][j]`,
 * `alt_votes[j] = support[best_alt][j]` where `best_alt` is the non-reference allele with the most
 * supporting k-mer windows on this read.
 */
struct KmerEvidenceScan {
  std::vector<float> ref_votes;
  std::vector<float> alt_votes;

  /// Number of k-mer windows in this scan that hit a key flagged as "locally multi-occurrent" -- one that
  /// recurred more than once within some single haplotype at index-build time (plan constraint 2's
  /// multi-occurrence distortion). Only ever nonzero under the `all` haplotype-sequence build (constructor
  /// with `exclude_local_multi=false`): `local_unique` removes such keys from the table entirely (so scans
  /// against it simply miss them, indistinguishable from novel k-mers), and the `graph_unique` constructor
  /// never flags anything (uniqueness is already enforced globally by `UniqueKmersOverlay`). Used by the
  /// dual-run agreement harness's "multi-occurrence k-mer overlap" stratification.
  size_t multi_occurrence_hits = 0;
};

/**
 * @brief A per-region index mapping exact k-mers to the set of sampled alleles ("haplotypes") that
 * contain them, used to accumulate per-base allele-support evidence for a read in a single forward
 * pass (one hash lookup per read position) instead of one full realignment per (read, allele) pair.
 *
 * See `PERFORMANCE_NOTES.md` / the "Cheaper allele evidence for image generation" plan (Stage 1) for the
 * full design. In short:
 *  - Only exact matches are indexed; k-mers shared by every sampled allele carry no discriminating
 *    information and are pruned at build time, as are k-mers absent from every sampled allele.
 *  - Reads are scanned forward-only in reference orientation (constraint 7 of the plan) -- callers must
 *    not reverse-complement reads before calling Scan().
 *  - At most 16 alleles are supported (allele index is a bit position in a uint16_t mask).
 */
class KmerEvidenceIndex {
 public:
  /**
   * @brief Build an index directly from full haplotype sequences (the `all`/`local_unique`
   * `evidence_kmer_set` settings).
   *
   * @param haplotype_sequences One sequence per sampled allele, allele 0 must be the reference.
   * @param k K-mer size. Must satisfy 1 <= k <= 32 (so a k-mer packs into a uint64_t).
   * @param exclude_local_multi If true (the `local_unique` setting), a k-mer that occurs more than once
   *   within any single haplotype sequence is dropped entirely rather than retained with an
   *   under-informative multiplicity-inflated mask (see plan constraint 2). If false (the `all`
   *   setting), no such filtering is performed.
   *
   * @throws std::invalid_argument if `haplotype_sequences` is empty, has more than 16 entries, or `k`
   *   is out of range.
   */
  KmerEvidenceIndex(const std::vector<std::string>& haplotype_sequences, size_t k,
                     bool exclude_local_multi = false);

  /**
   * @brief Build an index from a pre-enumerated k-mer set and per-k-mer allele masks (the
   * `graph_unique` `evidence_kmer_set` setting, fed from `UniqueKmersOverlay`/`HaplotypeSamplerOverlay`
   * via `kmers_on_path`).
   *
   * @param kmer_sequences Every candidate k-mer sequence (each of length `k`); index-aligned with @p masks.
   * @param masks Per-k-mer allele bitset (bit `a` set if allele `a` contains that k-mer), index-aligned
   *   with @p kmer_sequences. Entries that are 0 (present on no sampled allele) or cover every sampled
   *   allele are dropped, matching the pruning applied by the other constructor.
   * @param k K-mer size; every entry of @p kmer_sequences must have this length.
   *
   * @throws std::invalid_argument if the inputs are empty, mismatched in length, or `k` is out of range.
   */
  KmerEvidenceIndex(const std::vector<std::string>& kmer_sequences, const std::vector<uint16_t>& masks,
                     size_t k, size_t num_alleles);

  size_t k() const { return k_; }
  size_t num_alleles() const { return num_alleles_; }

  /// Number of retained (indexed) k-mer keys -- the "retained-key count" measurement in plan constraint 4.
  /// (Distinct from the hash table's internal capacity, which is sized to keep the load factor low.)
  size_t size() const { return num_entries_; }

  /**
   * @brief Scan a forward, reference-oriented read sequence and return per-base ref/alt vote arrays.
   *
   * @param seq The read sequence (e.g. `pysam` `query_sequence`, already reference-oriented -- no
   *   reverse-complementing is performed here, see plan constraint 7). May contain non-ACGT bases (e.g.
   *   'N'); any k-mer window touching one is simply skipped, like a hash miss.
   * @param base_qualities If non-null, must be the same length as @p seq. Combined with @p min_baseq to
   *   gate low-quality windows (plan constraint 1); ignored when @p min_baseq is 0.
   * @param min_baseq If non-zero, any k-mer window containing a base with quality below this threshold
   *   is skipped, exactly as if it were a hash miss. Defaults to 0 (no gating), matching the plan's "do
   *   not weight or gate the votes" default.
   */
  KmerEvidenceScan Scan(const std::string& seq, const std::vector<uint8_t>* base_qualities = nullptr,
                        uint8_t min_baseq = 0) const;

 private:
  bool Lookup(uint64_t key, uint16_t* mask, bool* multi) const;
  void BuildTable(const std::vector<std::pair<uint64_t, uint16_t>>& entries,
                  const std::vector<bool>& multi_flags = {});

  size_t k_;
  size_t num_alleles_;

  // Flat open-addressing hash table (linear probing) over packed 2-bit-encoded keys. occupied_[i] guards
  // keys_[i]/masks_[i] rather than a sentinel key value, since every value of a k<=32 packed key is
  // otherwise a legal k-mer encoding.
  std::vector<uint64_t> keys_;
  std::vector<uint16_t> masks_;
  std::vector<bool> occupied_;
  // Parallel to keys_/masks_: true if this key was flagged as locally multi-occurrent at build time (see
  // KmerEvidenceScan::multi_occurrence_hits). Left all-false by the graph_unique constructor.
  std::vector<bool> multi_flag_;
  size_t num_entries_ = 0;

  // Reusable scan scratch space (avoids reallocating per-allele difference arrays on every Scan() call);
  // not thread-safe, matching the rest of this codebase's per-region/per-actor object lifetimes.
  mutable std::vector<float> scratch_diff_;
  mutable std::vector<double> scratch_total_;
};

}  // namespace npsv3
