#include <algorithm>
#include <iostream>
#include <stdexcept>

#include <boost/hash2/sha1.hpp>
#include <boost/scope/scope_exit.hpp>
#include <htslib/kseq.h>
#include <htslib/kstring.h>

#include "variant.hpp"





namespace npsv3 {

std::unique_ptr<Variant> Variant::Create(const std::shared_ptr<bcf_hdr_t>& hdr, bcf1_t* record) {
  RecordPtr record_ptr(record);

  if (bcf_unpack(record_ptr.get(), BCF_UN_STR) < 0)  // Decode up to the ALTs (if not already done)
    throw std::runtime_error("Failed to unpack variant record");
  bool has_symbolic_allele = false;
  for (int i = 1; i < record_ptr->n_allele; i++) {
    auto allele = std::string_view(record_ptr->d.allele[i]);
    if (!allele.empty() && (allele.front() == '<' || allele.back() == '>')) {
      has_symbolic_allele = true;
      break;
    }
  }
  if (!has_symbolic_allele) {
    return std::make_unique<SequenceResolvedVariant>(hdr, std::move(record_ptr));
  } else {
    throw std::runtime_error("Symbolic alleles not yet supported");
  }
}

Variant::Variant(const std::shared_ptr<bcf_hdr_t>& hdr, RecordPtr record) : hdr_(hdr), record_(std::move(record)) {}

ContigName Variant::Contig() const {
  // TODO: Pre-create contig names for each rid in the header to avoid repeated construction
  auto contig = bcf_seqname(hdr_.get(), record_.get());
  if (!contig) {
    throw std::runtime_error("Failed to get contig name for variant record");
  }
  return ContigName(contig);
}

Variant::VariantIdType Variant::VariantId() const {
  // https://github.com/vgteam/vg/blob/da34f4e54b0e64d1b741da102217c97d5333fabc/src/utility.cpp#L505
  std::stringstream variant_stringer;
  variant_stringer << bcf_seqname(hdr_.get(), record_.get()) << '\n';
  variant_stringer << record_->pos << '\n';
  for (int i = 0; i < record_->n_allele; i++) {
    const char* allele = record_->d.allele[i];
    while (*allele != '\0') variant_stringer << toupper(*allele++);
    variant_stringer << '\n';
  }
  auto variant_string = variant_stringer.str();

  boost::hash2::sha1_160 hasher;
  hasher.update(variant_string.c_str(), variant_string.size());
  return hasher.result();
}

std::ostream& operator<<(std::ostream& os, const Variant& variant) {
  kstring_t line = {0, 0, nullptr};
  if (vcf_format(variant.hdr_.get(), variant.record_.get(), &line) < 0) {
    if (line.m) {
      free(line.s);
      // ks_free(&line);
    }
    throw std::runtime_error("vcf_format failed");
  }
  os << line.s;
  free(line.s);
  return os;
}

SequenceResolvedVariant::SequenceResolvedVariant(const std::shared_ptr<bcf_hdr_t>& hdr, RecordPtr record)
    : Variant(hdr, std::move(record)) {
  // Compute the left and right padding for each ALT allele (since it can be different for each allele)
  left_padding_.resize(record_->n_allele - 1, 0);
  right_padding_.resize(record_->n_allele - 1, 0);

  auto ref_allele = std::string_view(record_->d.allele[0]);
  for (int i = 1; i < record_->n_allele; i++) {
    auto alt_allele = std::string_view(record_->d.allele[i]);

    // Compute per-allele padding starting with the "right" side to effectively left-align the variant.
    // We don't remove "right" padding when any allele has just length 1 since that can collapse insertions.
    if (ref_allele.size() > 1 && alt_allele.size() > 1) {
      auto [right_padding_iter, _right_padding_iter] =
          std::mismatch(ref_allele.rbegin(), ref_allele.rend(), alt_allele.rbegin(), alt_allele.rend());
      right_padding_[i - 1] = std::distance(ref_allele.rbegin(), right_padding_iter);
    }

    auto [left_padding_iter, _left_padding_iter] =
        std::mismatch(ref_allele.begin(), ref_allele.end() - right_padding_[i - 1], alt_allele.begin(),
                      alt_allele.end() - right_padding_[i - 1]);
    left_padding_[i - 1] = std::distance(ref_allele.begin(), left_padding_iter);
  }
}

Range SequenceResolvedVariant::ReferenceRegion() const {
  auto pos = record_->pos;  // 0-based position
  return Range(Contig(), pos + *std::min_element(std::begin(left_padding_), std::end(left_padding_)),
               pos + record_->rlen - *std::min_element(std::begin(right_padding_), std::end(right_padding_)));
}

std::optional<Range> SequenceResolvedVariant::AlleleReferenceRegion(int allele_idx) const {
  if (allele_idx < 0 || allele_idx >= record_->n_allele) {
    throw std::out_of_range("Allele index out of range");
  }
  if (allele_idx == 0) {
    return std::make_optional<Range>(ReferenceRegion());
  }
  auto alt_allele = std::string_view(record_->d.allele[allele_idx]);
  if (alt_allele == "*") {
    return std::nullopt;  // Spanning deletion, no reference region
  }
  auto pos = record_->pos;  // 0-based position
  return std::make_optional<Range>(Contig(), pos + left_padding_[allele_idx - 1],
                                   pos + record_->rlen - right_padding_[allele_idx - 1]);
}

std::optional<std::string_view> SequenceResolvedVariant::AlleleSequence(int allele_idx) const {
  if (allele_idx <= 0 || allele_idx >= record_->n_allele) {
    throw std::out_of_range("Allele index out of range");
  }
  auto alt_allele = std::string_view(record_->d.allele[allele_idx]);
  if (alt_allele == "*") {
    return std::nullopt;  // Spanning deletion, no allele sequence
  }
  return std::string_view(std::begin(alt_allele) + left_padding_[allele_idx - 1],
                          alt_allele.size() - left_padding_[allele_idx - 1] - right_padding_[allele_idx - 1]);
}

std::unique_ptr<VariantFileReader> VariantFileReader::Open(const std::string& filename) {
  FilePtr file_ptr(hts_open(filename.c_str(), "r"));
  if (!file_ptr) {
    throw std::runtime_error("Failed to open VCF/BCF file: " + filename);
  }

  const htsFormat* format = hts_get_format(file_ptr.get());
  if (format->format == htsExactFormat::vcf && format->compression == htsCompression::bgzf) {
    return std::make_unique<VCFVariantFileReader>(std::move(file_ptr));
  } else if (format->format == htsExactFormat::bcf) {
    return std::make_unique<BCFVariantFileReader>(std::move(file_ptr));
  } else {
    throw std::runtime_error("Unsupported file format for file: " + filename);
  }
}

VariantFileReader::VariantFileReader(FilePtr&& file) : file_(std::move(file)) {
  hdr_.reset(bcf_hdr_read(file_.get()), internal::bcf_hdr_deleter());
}

std::unique_ptr<Variant> VariantFileReader::NextVariant(int max_unpack) {
  bcf1_t* record = bcf_init();
  record->max_unpack = max_unpack;
  int ret = bcf_read(file_.get(), hdr_.get(), record);
  if (ret < 0) {
    auto errcode = record->errcode;
    bcf_destroy(record);
    if (errcode) {
      throw std::runtime_error("Unable to parse next record");
    }
    if (ret == -1) {
      return nullptr;  // End of iteration
    } else {
      throw std::runtime_error("Unable to fetch or parse next record");
    }
  }
  return Variant::Create(hdr_, record);
}

VCFVariantFileReader::VCFVariantFileReader(FilePtr&& file) : VariantFileReader(std::move(file)) {
  idx_.reset(tbx_index_load(file_->fn));
  if (!idx_) {
    throw std::runtime_error("Failed to load VCF index (.tbi or .csi) for VCF file");
  }
  SetRegion(); // Default to reading the entire file
}


void VCFVariantFileReader::SetRegion() {
  iter_.reset(tbx_itr_queryi(idx_.get(), HTS_IDX_START, 0, 0));
  if (!iter_) {
    throw std::runtime_error("Failed to set region for VCF/BCF file");
  }
}

void VCFVariantFileReader::SetRegion(const Range& region) {
  iter_.reset(tbx_itr_queryi(idx_.get(), tbx_name2id(idx_.get(), region.Contig().c_str()), region.Start(), region.End()));
  if (!iter_) {
    throw std::runtime_error("Failed to set region for VCF/BCF file");
  }
}

VariantFileReader::VariantPtr VCFVariantFileReader::NextVariant(int max_unpack) {
  kstring_t str = KS_INITIALIZE;
  boost::scope::scope_exit free_str([&str]() { // Make sure to free kstring_t memory on all exit paths
    ks_free(&str);
  });
  
  int iter_ret = tbx_itr_next(file_.get(), idx_.get(), iter_.get(), &str);
  if (iter_ret == -1) {
    return nullptr;  // End of iteration
  } else if (iter_ret < 0) {
    throw std::runtime_error("Unable to fetch next record VCF file");
  }

  bcf1_t* record = bcf_init();
  record->max_unpack = max_unpack;
  int ret = vcf_parse(&str, hdr_.get(), record);
  if (ret < 0) {
    bcf_destroy(record);
    throw std::runtime_error("Failed to parse VCF record");
  }
  return Variant::Create(hdr_, record);
}

BCFVariantFileReader::BCFVariantFileReader(FilePtr&& file) : VariantFileReader(std::move(file)) {
  idx_.reset(bcf_index_load(file_->fn));
  if (!idx_) {
    throw std::runtime_error("Failed to load BCF index (.csi) for BCF file");
  }
}

void BCFVariantFileReader::SetRegion() {
  throw std::runtime_error("Not yet implemented");
}

void BCFVariantFileReader::SetRegion(const Range& region) {
  throw std::runtime_error("Not yet implemented");
}

}  // namespace npsv3