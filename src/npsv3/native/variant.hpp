#pragma once

#include <htslib/bgzf.h>
#include <htslib/hts.h>
#include <htslib/tbx.h>
#include <htslib/vcf.h>

#include <boost/hash2/digest.hpp>
#include <iosfwd>
#include <memory>
#include <optional>
#include <vector>

#include "range.hpp"

namespace npsv3 {

namespace detail {
struct htsFile_deleter {
  void operator()(htsFile* file) const { hts_close(file); }
};

struct bgzf_deleter {
  void operator()(BGZF* bgzf) const { bgzf_close(bgzf); }
};

struct bcf_hdr_deleter {
  void operator()(bcf_hdr_t* hdr) const { bcf_hdr_destroy(hdr); }
};

struct tbx_deleter {
  void operator()(tbx_t* tidx) const { tbx_destroy(tidx); }
};

struct hts_idx_deleter {
  void operator()(hts_idx_t* idx) const { hts_idx_destroy(idx); }
};

struct hts_iter_deleter {
  void operator()(hts_itr_t* iter) const { hts_itr_destroy(iter); }
};

struct bcf1_deleter {
  void operator()(bcf1_t* record) const { bcf_destroy(record); }
};

template <typename T>
struct BCFEncodingValues;

template <>
struct BCFEncodingValues<int8_t> {
  static constexpr int8_t kMissing = bcf_int8_missing;
  static constexpr int8_t kVectorEnd = bcf_int8_vector_end;
};

template <>
struct BCFEncodingValues<int16_t> {
  static constexpr int16_t kMissing = bcf_int16_missing;
  static constexpr int16_t kVectorEnd = bcf_int16_vector_end;
};

template <>
struct BCFEncodingValues<int32_t> {
  static constexpr int32_t kMissing = bcf_int32_missing;
  static constexpr int32_t kVectorEnd = bcf_int32_vector_end;
};

}  // namespace detail

class VariantFileHeader {
 public:
  typedef std::unique_ptr<bcf_hdr_t, detail::bcf_hdr_deleter> HeaderPtr;

  VariantFileHeader(bcf_hdr_t* hdr);

  bcf_hdr_t* bcf_hdr() const { return hdr_.get(); }

  bool HasGT() const { return gt_id_ >= 0; }
  int GTId() const { return gt_id_; }

  bool HasPS() const { return ps_id_ >= 0; }
  int PSId() const { return ps_id_; }

 private:
  HeaderPtr hdr_;
  int gt_id_ = -1;
  int ps_id_ = -1;
};

class Phase {
 public:
  static inline constexpr int32_t kMaxPhaseSet = 0x7FFFFFFF;

  enum Value : uint32_t {
    kUnphased = 0,
    kLocal = (uint32_t)1 << 31,
    kGlobal = (uint32_t)1 << 30,
    kImplicit = (uint32_t)1 << 29
  };

  Phase() : value_(kUnphased) {}
  Phase(int phase_set) : Phase(kLocal, phase_set) {}
  Phase(Value v, int phase_set = -1) : value_(v == kLocal ? static_cast<uint32_t>(kLocal) | static_cast<uint32_t>(phase_set) : static_cast<uint32_t>(v)) {;
    if (v == kLocal && (phase_set < 0 || phase_set > kMaxPhaseSet)) {
      throw std::out_of_range("Local phase set must be in [0, " + std::to_string(kMaxPhaseSet) + "]");
    }
  }

  bool operator==(const Phase& other) const { return value_ == other.value_; }
  bool operator!=(const Phase& other) const { return value_ != other.value_; }
  bool operator==(Value v) const { return (v == kLocal) ? value_ & static_cast<uint32_t>(kLocal) : value_ == static_cast<uint32_t>(v); }

  bool is_phased() const { return value_; }
  Value phasing() const { return (value_ & static_cast<uint32_t>(kLocal)) ? kLocal : static_cast<Value>(value_); }

 private:
  // Use MSB to encode local phase. If locally phased, store phase set in the lower 31 bits.
  uint32_t value_;
};

template <int kMaxPloidy = 3, typename AlleleIndex = int8_t>
class PackedGenotype {
 public:
  typedef std::array<AlleleIndex, kMaxPloidy> AlleleIndices;
  static inline constexpr AlleleIndex kMissingAllele = -1;
  static inline constexpr size_t kMaxAlleles = 0x3F;  // 6 bits for allele index

  PackedGenotype() : num_alleles_(0), phase_(Phase::kUnphased) {}

  template <typename BCFEncoding>
  PackedGenotype(const BCFEncoding* gt, int max_ploidy, int32_t phase_set)
      : num_alleles_(0), phase_(Phase::kUnphased) {
    if (max_ploidy > kMaxPloidy) {
      throw std::out_of_range("Genotype ploidy exceeds maximum supported ploidy");
    }
    bool phased = true;
    for (int i = 0; i < max_ploidy; i++) {
      auto allele = gt[i];
      if (allele == detail::BCFEncodingValues<BCFEncoding>::kVectorEnd) {
        break;  // No more alleles for this sample
      } else if (allele == detail::BCFEncodingValues<BCFEncoding>::kMissing) {
        allele_indices_[num_alleles_++] = kMissingAllele;
        phased = false;
      } else if (bcf_gt_is_missing(allele)) {
        allele_indices_[num_alleles_++] = kMissingAllele;
        phased = phased && bcf_gt_is_phased(allele);
      } else {
        auto idx = static_cast<AlleleIndex>(bcf_gt_allele(allele));
        if (idx < 0 || idx >= kMaxAlleles) {
          throw std::out_of_range("Allele index must be [0, " + std::to_string(kMaxAlleles) + "]");
        }
        allele_indices_[num_alleles_++] = idx;
        if (i > 0) {  // Phase only applies if more than one allele
          phased = phased && bcf_gt_is_phased(allele);
        }
      }
    }
    if (phased) {
      // phase_set of -1 indicates no PS, i.e., global phasing
      phase_ = (phase_set < 0) ? Phase::kGlobal : Phase(phase_set) /* Sets kLocal phasing */;
    } else if (num_alleles_ > 0 &&
               std::all_of(std::begin(allele_indices_), std::begin(allele_indices_) + num_alleles_,
                           [this](AlleleIndex index) { return index >= 0 && index == allele_indices_[0]; })) {
      phase_ = Phase::kImplicit;
    }
  }

  size_t num_alleles() const { return num_alleles_; }
  const AlleleIndices& allele_indices() const { return allele_indices_; }

  bool is_phased() const { return phase_.is_phased(); }
  Phase phase() const { return phase_; }

  bool AnyAllele(AlleleIndex allele_idx) const {
    return std::any_of(std::begin(allele_indices_), std::begin(allele_indices_) + num_alleles_,
                       [allele_idx](AlleleIndex index) { return index == allele_idx; });
  }
  bool AllAlleles(AlleleIndex allele_idx) const {
    return std::all_of(std::begin(allele_indices_), std::begin(allele_indices_) + num_alleles_,
                       [allele_idx](AlleleIndex index) { return index == allele_idx; });
  }
  size_t AlleleCount(AlleleIndex allele_idx) const {
    return std::count(std::begin(allele_indices_), std::begin(allele_indices_) + num_alleles_, allele_idx);
  }

  friend std::ostream& operator<<(std::ostream& os, const PackedGenotype& genotype) {
    for (size_t i = 0; i < genotype.num_alleles_; i++) {
      if (i > 0) os << "/";
      if (genotype.allele_indices_[i] == kMissingAllele)
        os << ".";
      else
        os << static_cast<int>(genotype.allele_indices_[i]);
    }
    return os;
  }

 private:
  size_t num_alleles_ : 6;
  AlleleIndices allele_indices_;
  Phase phase_;
};

class Variant {
 public:
  enum Flags : uint8_t {
    kHasStarAllele = 1 << 0,
    kIsOverlapping = 1 << 1
  };
  
  typedef std::shared_ptr<VariantFileHeader> HeaderPtr;
  typedef boost::hash2::digest<20> VariantId;
  typedef PackedGenotype<> Genotype;

  static std::unique_ptr<Variant> Create(const HeaderPtr& hdr, bcf1_t* record);

  ContigName contig() const;

  int num_alleles() const { return record_->n_allele; }
  int num_alts() const { return record_->n_allele - 1; }
  int AlleleIndex(const std::string& allele_sequence) const;

  VariantId variant_id() const;

  void add_flag(Flags flag) {
    flags_ = static_cast<Flags>(static_cast<std::underlying_type<Flags>::type>(flags_) |
                                static_cast<std::underlying_type<Flags>::type>(flag));
  }
  bool has_flag(Flags flag) const { return (flags_ & flag); }

  virtual Range ReferenceRegion() const;
  virtual std::optional<Range> AlleleReferenceRegion(int allele_idx) const = 0;

  virtual std::optional<std::string_view> AlleleSequence(int allele_idx) const = 0;

  std::vector<Genotype> Genotypes() const;

  friend std::ostream& operator<<(std::ostream&, const Variant&);

 protected:
  typedef std::unique_ptr<bcf1_t, detail::bcf1_deleter> RecordPtr;

  HeaderPtr hdr_;
  RecordPtr record_;
  Flags flags_ = static_cast<Flags>(0);
  Pos left_padding_ = 0;
  Pos right_padding_ = 0;

  Variant(const HeaderPtr& hdr, RecordPtr record);

  std::vector<Genotype> Genotypes(int gt_id, int ps_id) const;
};

class SequenceResolvedVariant : public Variant {
 public:
  SequenceResolvedVariant(const HeaderPtr& hdr, RecordPtr record);

  std::optional<Range> AlleleReferenceRegion(int allele_idx) const override;

  std::optional<std::string_view> AlleleSequence(int allele_idx) const override;

 private:
  std::vector<Pos> allele_left_padding_;
  std::vector<Pos> allele_right_padding_;
};

class VariantFileReader {
 public:
  typedef std::unique_ptr<VariantFileReader> VariantFileReaderPtr;
  typedef std::unique_ptr<Variant> VariantPtr;

  static VariantFileReaderPtr Open(const std::string& filename);

  // Iterate over the entire file
  virtual void SetRegion() = 0;
  virtual void SetRegion(const Range& region) = 0;

  virtual VariantPtr NextVariant(int max_unpack = BCF_UN_ALL);

  std::vector<std::string> Samples() const;

 protected:
  typedef std::unique_ptr<htsFile, detail::htsFile_deleter> FilePtr;
  typedef std::shared_ptr<VariantFileHeader> HeaderPtr;

  VariantFileReader(FilePtr&& file);

  FilePtr file_;
  HeaderPtr hdr_;
};

class VCFVariantFileReader : public VariantFileReader {
 public:
  VCFVariantFileReader(FilePtr&& file);

  void SetRegion() override;
  void SetRegion(const Range& region) override;

  VariantPtr NextVariant(int max_unpack = BCF_UN_ALL) override;

 protected:
  std::unique_ptr<tbx_t, detail::tbx_deleter> idx_;
  std::unique_ptr<hts_itr_t, detail::hts_iter_deleter> iter_;
};

class BCFVariantFileReader : public VariantFileReader {
 public:
  BCFVariantFileReader(FilePtr&& file);

  void SetRegion() override;
  void SetRegion(const Range& region) override;

 protected:
  std::unique_ptr<hts_idx_t, detail::hts_idx_deleter> idx_;
};
}  // namespace npsv3
