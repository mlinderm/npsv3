#pragma once

#include <iosfwd>
#include <memory>
#include <optional>
#include <vector>

#include <htslib/hts.h>
#include <htslib/bgzf.h>
#include <htslib/tbx.h>
#include <htslib/vcf.h>

#include <boost/hash2/digest.hpp>

#include "range.hpp"

namespace npsv3 {

namespace internal {
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
}  // namespace internal

class Variant {
 public:
  typedef boost::hash2::digest<20> VariantIdType;

  static std::unique_ptr<Variant> Create(const std::shared_ptr<bcf_hdr_t>& hdr, bcf1_t* record);

  ContigName Contig() const;

  int NumAllele() const { return record_->n_allele; }
  int NumAlt() const { return record_->n_allele - 1; }

  VariantIdType VariantId() const;

  virtual Range ReferenceRegion() const = 0;
  virtual std::optional<Range> AlleleReferenceRegion(int allele_idx) const = 0;

  virtual std::optional<std::string_view> AlleleSequence(int allele_idx) const = 0;

  friend std::ostream& operator<<(std::ostream&, const Variant&);

 protected:
  typedef std::unique_ptr<bcf1_t, internal::bcf1_deleter> RecordPtr;

  std::shared_ptr<bcf_hdr_t> hdr_;
  RecordPtr record_;

  Variant(const std::shared_ptr<bcf_hdr_t>& hdr, RecordPtr record);
};

class SequenceResolvedVariant : public Variant {
 public:
  SequenceResolvedVariant(const std::shared_ptr<bcf_hdr_t>& hdr, RecordPtr record);

  Range ReferenceRegion() const override;
  std::optional<Range> AlleleReferenceRegion(int allele_idx) const override;

  std::optional<std::string_view> AlleleSequence(int allele_idx) const override;

 private:
  std::vector<Pos> left_padding_;
  std::vector<Pos> right_padding_;
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

 protected:
  typedef std::unique_ptr<htsFile, internal::htsFile_deleter> FilePtr;

  VariantFileReader(FilePtr&& file);

  FilePtr file_;
  std::shared_ptr<bcf_hdr_t> hdr_;
};

class VCFVariantFileReader : public VariantFileReader {
public:
  VCFVariantFileReader(FilePtr&& file);

  void SetRegion() override;
  void SetRegion(const Range& region) override;

  VariantPtr NextVariant(int max_unpack = BCF_UN_ALL) override;

protected:
  std::unique_ptr<tbx_t, internal::tbx_deleter> idx_;
  std::unique_ptr<hts_itr_t, internal::hts_iter_deleter> iter_;
};

class BCFVariantFileReader : public VariantFileReader {
public:
  BCFVariantFileReader(FilePtr&& file);

  void SetRegion() override;
  void SetRegion(const Range& region) override;

protected:
  std::unique_ptr<hts_idx_t, internal::hts_idx_deleter> idx_;
};
}  // namespace npsv3