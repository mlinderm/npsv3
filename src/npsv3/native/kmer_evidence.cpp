#include "kmer_evidence.hpp"

#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include <fmt/format.h>

namespace npsv3 {

namespace {

constexpr size_t kMaxAlleles = 16;  // Allele index is a bit position in a uint16_t mask.

inline int Base2Bit(char c) {
  switch (c) {
    case 'A':
    case 'a':
      return 0;
    case 'C':
    case 'c':
      return 1;
    case 'G':
    case 'g':
      return 2;
    case 'T':
    case 't':
      return 3;
    default:
      return -1;
  }
}

// 64-bit finalizer mix (splitmix64), used to distribute packed k-mer keys across the hash table.
inline uint64_t MixHash(uint64_t x) {
  x ^= x >> 33;
  x *= 0xff51afd7ed558ccdULL;
  x ^= x >> 33;
  x *= 0xc4ceb9fe1a85ec53ULL;
  x ^= x >> 33;
  return x;
}

inline size_t NextPow2(size_t n) {
  size_t p = 1;
  while (p < n) p <<= 1;
  return p;
}

void ValidateK(size_t k) {
  if (k < 1 || k > 32) {
    throw std::invalid_argument(fmt::format("Evidence k-mer size must be between 1 and 32, got {}", k));
  }
}

void ValidateNumAlleles(size_t num_alleles) {
  if (num_alleles < 1 || num_alleles > kMaxAlleles) {
    throw std::invalid_argument(
        fmt::format("KmerEvidenceIndex supports between 1 and {} alleles, got {}", kMaxAlleles, num_alleles));
  }
}

}  // namespace

KmerEvidenceIndex::KmerEvidenceIndex(const std::vector<std::string>& haplotype_sequences, size_t k,
                                      bool exclude_local_multi)
    : k_(k), num_alleles_(haplotype_sequences.size()) {
  ValidateK(k_);
  ValidateNumAlleles(num_alleles_);

  const uint64_t key_mask = (k_ >= 64) ? ~0ULL : ((1ULL << (2 * k_)) - 1);
  const uint16_t all_alleles_mask =
      (num_alleles_ >= kMaxAlleles) ? uint16_t{0xFFFF} : static_cast<uint16_t>((uint16_t{1} << num_alleles_) - 1);

  std::unordered_map<uint64_t, uint16_t> mask_map;
  std::unordered_set<uint64_t> multi_keys;  // keys occurring more than once within some single haplotype

  for (size_t a = 0; a < num_alleles_; ++a) {
    const std::string& hap = haplotype_sequences[a];

    std::unordered_map<uint64_t, uint32_t> local_counts;
    uint64_t key = 0;
    size_t valid_run = 0;
    for (char c : hap) {
      const int code = Base2Bit(c);
      if (code < 0) {
        valid_run = 0;
        key = 0;
        continue;
      }
      key = ((key << 2) | static_cast<uint64_t>(code)) & key_mask;
      if (++valid_run < k_) continue;

      mask_map[key] |= static_cast<uint16_t>(uint16_t{1} << a);
      // Tracked regardless of exclude_local_multi: local_unique uses this set to drop the key below, and
      // `all` retains it but still needs to know, for KmerEvidenceScan::multi_occurrence_hits, which
      // retained keys were locally multi-occurrent (the agreement harness's stratification axis).
      if (++local_counts[key] >= 2) {
        multi_keys.insert(key);
      }
    }
  }

  std::vector<std::pair<uint64_t, uint16_t>> entries;
  std::vector<bool> multi_flags;
  entries.reserve(mask_map.size());
  multi_flags.reserve(mask_map.size());
  for (const auto& [key, mask] : mask_map) {
    if (mask == 0 || mask == all_alleles_mask) continue;  // uninformative: on no / every sampled allele
    const bool is_multi = multi_keys.count(key) > 0;
    if (exclude_local_multi && is_multi) continue;
    entries.emplace_back(key, mask);
    multi_flags.push_back(is_multi);
  }
  BuildTable(entries, multi_flags);
}

KmerEvidenceIndex::KmerEvidenceIndex(const std::vector<std::string>& kmer_sequences,
                                      const std::vector<uint16_t>& masks, size_t k, size_t num_alleles)
    : k_(k), num_alleles_(num_alleles) {
  ValidateK(k_);
  ValidateNumAlleles(num_alleles_);
  if (kmer_sequences.size() != masks.size()) {
    throw std::invalid_argument(
        fmt::format("kmer_sequences ({}) and masks ({}) must have the same length", kmer_sequences.size(),
                    masks.size()));
  }

  const uint64_t key_mask = (k_ >= 64) ? ~0ULL : ((1ULL << (2 * k_)) - 1);
  const uint16_t all_alleles_mask =
      (num_alleles_ >= kMaxAlleles) ? uint16_t{0xFFFF} : static_cast<uint16_t>((uint16_t{1} << num_alleles_) - 1);

  std::unordered_map<uint64_t, uint16_t> mask_map;
  for (size_t i = 0; i < kmer_sequences.size(); ++i) {
    const std::string& seq = kmer_sequences[i];
    if (seq.size() != k_) {
      throw std::invalid_argument(
          fmt::format("kmer_sequences[{}] has length {}, expected k={}", i, seq.size(), k_));
    }
    uint64_t key = 0;
    bool valid = true;
    for (char c : seq) {
      const int code = Base2Bit(c);
      if (code < 0) {
        valid = false;
        break;
      }
      key = ((key << 2) | static_cast<uint64_t>(code)) & key_mask;
    }
    if (!valid) continue;  // Not representable (e.g. contains an N); skip rather than fail.
    mask_map[key] |= masks[i];
  }

  std::vector<std::pair<uint64_t, uint16_t>> entries;
  entries.reserve(mask_map.size());
  for (const auto& [key, mask] : mask_map) {
    if (mask == 0 || mask == all_alleles_mask) continue;
    entries.emplace_back(key, mask);
  }
  BuildTable(entries);
}

void KmerEvidenceIndex::BuildTable(const std::vector<std::pair<uint64_t, uint16_t>>& entries,
                                    const std::vector<bool>& multi_flags) {
  num_entries_ = entries.size();
  const size_t capacity = std::max<size_t>(8, NextPow2(entries.size() * 2));  // load factor <= 0.5
  keys_.assign(capacity, 0);
  masks_.assign(capacity, 0);
  occupied_.assign(capacity, false);
  multi_flag_.assign(capacity, false);

  const size_t mask_bits = capacity - 1;
  for (size_t i = 0; i < entries.size(); ++i) {
    const auto& [key, mask] = entries[i];
    size_t idx = MixHash(key) & mask_bits;
    while (occupied_[idx]) idx = (idx + 1) & mask_bits;
    keys_[idx] = key;
    masks_[idx] = mask;
    occupied_[idx] = true;
    multi_flag_[idx] = i < multi_flags.size() && multi_flags[i];
  }
}

bool KmerEvidenceIndex::Lookup(uint64_t key, uint16_t* mask, bool* multi) const {
  if (occupied_.empty()) return false;
  const size_t mask_bits = occupied_.size() - 1;
  size_t idx = MixHash(key) & mask_bits;
  for (size_t probes = 0; probes < occupied_.size(); ++probes) {
    if (!occupied_[idx]) return false;
    if (keys_[idx] == key) {
      *mask = masks_[idx];
      *multi = multi_flag_[idx];
      return true;
    }
    idx = (idx + 1) & mask_bits;
  }
  return false;
}

KmerEvidenceScan KmerEvidenceIndex::Scan(const std::string& seq, const std::vector<uint8_t>* base_qualities,
                                          uint8_t min_baseq) const {
  const size_t n = seq.size();
  KmerEvidenceScan result;
  result.ref_votes.assign(n, 0.0f);
  result.alt_votes.assign(n, 0.0f);
  if (n < k_ || num_entries_ == 0) return result;

  if (base_qualities && base_qualities->size() != n) {
    throw std::invalid_argument(
        fmt::format("base_qualities length {} does not match seq length {}", base_qualities->size(), n));
  }
  const bool gate_baseq = base_qualities != nullptr && min_baseq > 0;

  // Reusable per-allele difference arrays: num_alleles_ rows of (n + 1) columns, flattened.
  scratch_diff_.assign(num_alleles_ * (n + 1), 0.0f);
  scratch_total_.assign(num_alleles_, 0.0);
  auto diff_at = [&](size_t allele, size_t pos) -> float& { return scratch_diff_[allele * (n + 1) + pos]; };

  const uint64_t key_mask = (k_ >= 64) ? ~0ULL : ((1ULL << (2 * k_)) - 1);
  uint64_t key = 0;
  size_t valid_run = 0;
  long last_low_idx = -1;

  for (size_t i = 0; i < n; ++i) {
    if (gate_baseq && (*base_qualities)[i] < min_baseq) last_low_idx = static_cast<long>(i);

    const int code = Base2Bit(seq[i]);
    if (code < 0) {
      valid_run = 0;
      key = 0;
      continue;
    }
    key = ((key << 2) | static_cast<uint64_t>(code)) & key_mask;
    if (++valid_run < k_) continue;

    const size_t window_start = i + 1 - k_;
    if (gate_baseq && last_low_idx >= static_cast<long>(window_start)) continue;

    uint16_t mask;
    bool multi = false;
    if (!Lookup(key, &mask, &multi)) continue;  // No hit: novel/no-info, deferred channel (plan 1b).
    if (multi) ++result.multi_occurrence_hits;

    const size_t window_end = window_start + k_;  // == i + 1
    for (size_t a = 0; a < num_alleles_; ++a) {
      if (mask & (uint16_t(1) << a)) {
        diff_at(a, window_start) += 1.0f;
        diff_at(a, window_end) -= 1.0f;
        scratch_total_[a] += 1.0;
      }
    }
  }

  // Reduce the per-allele support vector to ref-vs-best-alt, mirroring the existing BWA path's
  // argmax collapse (plan 1a/1b). best_alt is chosen by total windows supporting that allele on this read.
  size_t best_alt = 1;
  double best_alt_total = -1.0;
  for (size_t a = 1; a < num_alleles_; ++a) {
    if (scratch_total_[a] > best_alt_total) {
      best_alt_total = scratch_total_[a];
      best_alt = a;
    }
  }
  const bool have_alt = num_alleles_ > best_alt;

  float acc_ref = 0.0f;
  float acc_alt = 0.0f;
  for (size_t j = 0; j < n; ++j) {
    acc_ref += diff_at(0, j);
    result.ref_votes[j] = acc_ref;
    if (have_alt) {
      acc_alt += diff_at(best_alt, j);
      result.alt_votes[j] = acc_alt;
    }
  }
  return result;
}

}  // namespace npsv3
