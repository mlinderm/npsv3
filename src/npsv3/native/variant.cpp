#include <algorithm>
#include <iostream>
#include <stdexcept>

#include <htslib/kseq.h>
#include <htslib/kstring.h>
#include <htslib/hfile.h>
#include <boost/hash2/sha1.hpp>
#include <boost/scope/scope_exit.hpp>
#include <fmt/format.h>
#include <spdlog/spdlog.h>

#include "variant.hpp"

namespace npsv3 {

VariantFileHeader::VariantFileHeader(bcf_hdr_t* hdr) : hdr_(hdr) {
  gt_id_ = bcf_hdr_id2int(hdr_.get(), BCF_DT_ID, "GT");
  if (gt_id_ >= 0 && !bcf_hdr_idinfo_exists(hdr_.get(), BCF_HL_FMT, gt_id_)) {
    gt_id_ = -1;
  }
  
  ps_id_ = bcf_hdr_id2int(hdr_.get(), BCF_DT_ID, "PS");
  if (ps_id_ >= 0 && !bcf_hdr_idinfo_exists(hdr_.get(), BCF_HL_FMT, ps_id_)) {
    ps_id_ = -1;
  }
  if (ps_id_ >= 0 && bcf_hdr_id2type(hdr_.get(), BCF_HL_FMT, ps_id_) != BCF_HT_INT) {
    spdlog::warn("PS format field is not of expected integer type; ignoring PS.");
    ps_id_ = -1;
  }

  ft_id_= bcf_hdr_id2int(hdr_.get(), BCF_DT_ID, "FT");
  if (ft_id_ >= 0 && !bcf_hdr_idinfo_exists(hdr_.get(), BCF_HL_FMT, ft_id_)) {
    ft_id_ = -1;
  }
  if (ft_id_ >= 0 && bcf_hdr_id2type(hdr_.get(), BCF_HL_FMT, ft_id_) != BCF_HT_STR) {
    spdlog::warn("FT format field is not of expected string type; ignoring FT.");
    ft_id_ = -1;
  }

  // PASS should always be defined (per https://github.com/samtools/htslib/blob/fe1721d876b1021ceb417cb2a0b246fa401b8c7f/vcf.c#L1425)
  assert(bcf_hdr_id2int(hdr_.get(), BCF_DT_ID, "PASS") == 0);
}

std::shared_ptr<VariantFileHeader> VariantFileHeader::Subset(const std::vector<std::string>& samples) const {
  // Convert sample names to char* array for htslib
  std::vector<char*> sample_ptrs;
  sample_ptrs.reserve(samples.size());
  for (const auto& sample : samples) {
    sample_ptrs.push_back(const_cast<char*>(sample.c_str()));
  }

  // Create subset header using htslib
  std::vector<int> imap; imap.reserve(samples.size());
  bcf_hdr_t* subset_hdr = bcf_hdr_subset(hdr_.get(), samples.size(), sample_ptrs.data(), imap.data());
  if (!subset_hdr) {
    throw std::runtime_error("Failed to create subset header");
  }

  return std::make_shared<VariantFileHeader>(subset_hdr);
}

Phase::Phase(Value v, int phase_set) {
  if (v == kLocal) {
    if (phase_set < 0 || phase_set > kMaxPhaseSet) {
      throw std::out_of_range("Local phase set must be in [0, " + std::to_string(kMaxPhaseSet) + "]");
    }
    value_ = static_cast<uint32_t>(kLocal) | static_cast<uint32_t>(phase_set);
  } else {
    value_ = static_cast<uint32_t>(v);
  }
}

std::unique_ptr<Variant> Variant::Create(const HeaderPtr& hdr, bcf1_t* record) {
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

Variant::Variant(const HeaderPtr& hdr, RecordPtr record) : hdr_(hdr), record_(std::move(record)) {
  if (bcf_unpack(record_.get(), BCF_UN_STR) < 0)  // Decode up to the ALTs (if not already done)
    throw std::runtime_error("Failed to unpack variant record");
}

ContigName Variant::contig() const {
  // TODO: Pre-create contig names for each rid in the header to avoid repeated construction
  auto contig = bcf_seqname(hdr_->bcf_hdr(), record_.get());
  if (!contig) {
    throw std::runtime_error("Failed to get contig name for variant record");
  }
  return ContigName(contig);
}

Variant::VariantId Variant::variant_id() const {
  // https://github.com/vgteam/vg/blob/da34f4e54b0e64d1b741da102217c97d5333fabc/src/utility.cpp#L505
  std::stringstream variant_stringer;
  variant_stringer << bcf_seqname(hdr_->bcf_hdr(), record_.get()) << '\n';
  variant_stringer << record_->pos + 1 << '\n'; // VG uses 1-based inclusive position for hash
  for (int i = 0; i < record_->n_allele; i++) {
    const char* allele = record_->d.allele[i];
    if (!allele || *allele == '*') // VG ignores "overlap" alleles
      continue;
    while (*allele != '\0') variant_stringer << static_cast<char>(toupper(*allele++));
    variant_stringer << '\n';
  }
  auto variant_string = variant_stringer.str();
  boost::hash2::sha1_160 hasher;
  hasher.update(variant_string.c_str(), variant_string.size());
  return hasher.result();
}

int Variant::AlleleIndex(const std::string& allele_sequence) const {
  for (int i = 1; i < record_->n_allele; i++) {
    if (allele_sequence == record_->d.allele[i]) {
      return i;
    }
  }
  return -1;
}

Range Variant::ReferenceRegion() const {
  auto pos = record_->pos;  // 0-based position
  return Range(contig(), pos + left_padding_, pos + record_->rlen - right_padding_);
}

Variant::RecordPtr Variant::SubsetSamplesRecord(const std::vector<int>& original_idxs) const {
  RecordPtr subset_record_ptr(bcf_dup(record_.get()));
  if (!subset_record_ptr) {
    throw std::runtime_error("Failed to duplicate variant record for subsetting");
  }

  if (bcf_subset(hdr_->bcf_hdr(), subset_record_ptr.get(), original_idxs.size(), const_cast<int*>(original_idxs.data())) < 0) {
    throw std::runtime_error("Failed to subset variant record");
  }

  return subset_record_ptr;
}

std::vector<Variant::Genotype> Variant::Genotypes(int gt_id, int ps_id) const {
  if (bcf_unpack(record_.get(), BCF_UN_ALL) < 0)
    throw std::runtime_error("Failed to unpack variant record for genotype extraction");

  int gt_fmt_idx = record_->n_fmt, ps_fmt_idx = record_->n_fmt;
  for (int i = 0; i < record_->n_fmt; i++) {
    int id = record_->d.fmt[i].id;
    if (id == gt_id) { gt_fmt_idx = i; if (ps_id < 0) break; }
    // Since "The first sub-field must always be the genotype (GT) if it is present.", PS must be past it
    // and so we can break early once we find PS.
    else if (id == ps_id) { ps_fmt_idx = i; break; }
  }
  if (gt_fmt_idx == record_->n_fmt) {
    throw std::runtime_error("GT format field not present in variant record");
  }
  
  int num_samples = bcf_hdr_nsamples(hdr_->bcf_hdr());
  
  // Extract phase sets if PS is defined
  std::vector<int32_t> phase_sets(num_samples, -1);
  if (ps_fmt_idx < record_->n_fmt) {
    bcf_fmt_t* ps_fmt = &record_->d.fmt[ps_fmt_idx];
    if (ps_fmt->n == 1) {
      #define BRANCH_CASE(TYPE) { \
        for (int i = 0; i < num_samples; i++) { \
          TYPE ps = *reinterpret_cast<TYPE*>(ps_fmt->p + i * ps_fmt->size); \
          if (ps == detail::BCFEncodingValues<TYPE>::kVectorEnd) { break; } \
          else if (ps == detail::BCFEncodingValues<TYPE>::kMissing) { continue; } \
          else { phase_sets[i] = ps; } \
        } \
      }
      
      switch (ps_fmt->type) {
        case BCF_BT_INT8: BRANCH_CASE(int8_t); break;
        case BCF_BT_INT16: BRANCH_CASE(int16_t); break;
        case BCF_BT_INT32: BRANCH_CASE(int32_t); break;
        default:
          throw std::runtime_error("Unsupported PS format field type");
      }

      #undef BRANCH_CASE
    } else {
      spdlog::warn("PS format field has {} number of values per sample, expecting 1; ignoring PS.", ps_fmt->n);
    }
  }
  
  std::vector<Variant::Genotype> genotypes;
  genotypes.reserve(num_samples);

  bcf_fmt_t* fmt = &record_->d.fmt[gt_fmt_idx];
  #define BRANCH_CASE(TYPE) { \
    for (int i = 0; i < num_samples; i++) { \
      TYPE* gt = reinterpret_cast<TYPE*>(fmt->p + i * fmt->size); \
      genotypes.emplace_back(gt, fmt->n, phase_sets[i]); \
    } \
  }

  switch (fmt->type) {
    case BCF_BT_INT8: BRANCH_CASE(int8_t); break;
    case BCF_BT_INT16: BRANCH_CASE(int16_t); break;
    case BCF_BT_INT32: BRANCH_CASE(int32_t); break;
    default:
      throw std::runtime_error("Unsupported GT format field type");
  }

  #undef BRANCH_CASE

  return genotypes;
}

std::vector<Variant::Genotype> Variant::Genotypes() const {
  if (!hdr_->HasGT()) {
    throw std::runtime_error("GT format field not defined in variant file");
  }
  return Genotypes(hdr_->GTId(), hdr_->PSId());
}

bool Variant::HasPassingGenotype(int gt_id, int ft_id) const {
  if (bcf_unpack(record_.get(), BCF_UN_ALL) < 0)
    throw std::runtime_error("Failed to unpack variant record for genotype FT checking");

  int gt_fmt_idx = record_->n_fmt, ft_fmt_idx = record_->n_fmt;
  for (int i = 0; i < record_->n_fmt; i++) {
    int id = record_->d.fmt[i].id;
    if (id == gt_id) { gt_fmt_idx = i; if (ft_id < 0) break; }
    // Since "The first sub-field must always be the genotype (GT) if it is present.", FT must be past it
    // and so we can break early once we find FT.
    else if (id == ft_id) { ft_fmt_idx = i; break; }
  }
  if (gt_fmt_idx == record_->n_fmt) {
    throw std::runtime_error("GT format field not present in variant record");
  }
  if (ft_fmt_idx == record_->n_fmt) {
    return true;  // No FT field defined, all genotypes are passing
  }
  
  int num_samples = bcf_hdr_nsamples(hdr_->bcf_hdr());
  bcf_fmt_t *ft_fmt = &record_->d.fmt[ft_fmt_idx], *gt_fmt = &record_->d.fmt[gt_fmt_idx];

  for (int i = 0; i < num_samples; i++) {
    const char* ft = reinterpret_cast<const char*>(ft_fmt->p) + i * ft_fmt->n;
    if (ft_fmt->n >= 4 && std::string_view(ft, 4) == "PASS") {
      return true;  // Found an explicitly passing genotype
    }
    if (ft_fmt->n >= 1 && ft[0] != '.') {
      continue;  // Found an explicitly failing genotype
    }
    assert(ft_fmt->n >= 1 && ft[0] == '.');
    // If FT is '.', check to see if there is a valid genotype (i.e. non-missing). If so, we have found a passing genotype
    // and return true.
    #define BRANCH_CASE(TYPE) { \
      for (int g=0; g < gt_fmt->n; g++) { \
        auto allele = reinterpret_cast<TYPE*>(gt_fmt->p + i * gt_fmt->size)[g]; \
        if (allele == detail::BCFEncodingValues<TYPE>::kVectorEnd) { break; } \
        else if (allele== detail::BCFEncodingValues<TYPE>::kMissing) { continue; } \
        else if (bcf_gt_is_missing(allele)) { continue; } \
        else { return true; } \
      } \
    }
    switch (gt_fmt->type) {
      case BCF_BT_INT8: BRANCH_CASE(int8_t); break;
      case BCF_BT_INT16: BRANCH_CASE(int16_t); break;
      case BCF_BT_INT32: BRANCH_CASE(int32_t); break;
      default:
        throw std::runtime_error("Unsupported GT format field type");
    }
    #undef BRANCH_CASE
  }

  return false;  // No passing genotypes found
}

bool Variant::HasPassingGenotype() const {
  if (!hdr_->HasGT()) {
    throw std::runtime_error("GT format field not defined in variant file");
  }
  if (!hdr_->HasFT()) {
    return true;  // No FT field defined, all genotypes are passing
  }
  return HasPassingGenotype(hdr_->GTId(), hdr_->FTId());
}

bool Variant::IsFiltered() const {
  if (!(record_->unpacked & BCF_UN_FLT)) {
    bcf_unpack(record_.get(), BCF_UN_FLT);
  }
  for (int i = 0; i < record_->d.n_flt; i++) {
    if (record_->d.flt[i] != hdr_->PASSId()) {
      return true;  // Has non-PASSing filter
    }
  }
  return false;
}

void Variant::SetFilterToPass() {
  static const int pass_id = hdr_->PASSId();
  if (bcf_update_filter(hdr_->bcf_hdr(), record_.get(), const_cast<int*>(&pass_id), 1) < 0) {
    throw std::runtime_error("Failed to set FILTER");
  }
}

std::ostream& operator<<(std::ostream& os, const Variant& variant) {
  kstring_t line = {0, 0, nullptr};
  if (vcf_format(variant.hdr_->bcf_hdr(), variant.record_.get(), &line) < 0) {
    ks_free(&line);
    throw std::runtime_error("vcf_format failed");
  }
  os << line.s;
  ks_free(&line);
  return os;
}

SequenceResolvedVariant::SequenceResolvedVariant(const HeaderPtr& hdr, RecordPtr record)
    : Variant(hdr, std::move(record)) {
  // Compute the left and right padding for each ALT allele (since it can be different for each allele)
  allele_left_padding_.resize(record_->n_allele - 1, 0);
  allele_right_padding_.resize(record_->n_allele - 1, 0);

  Pos left_padding = std::numeric_limits<Pos>::max();
  Pos right_padding = std::numeric_limits<Pos>::max();

  auto ref_allele = std::string_view(record_->d.allele[0]);
  for (int i = 1; i < record_->n_allele; i++) {
    auto alt_allele = std::string_view(record_->d.allele[i]);
    if (alt_allele == "*") {
      flags_ = static_cast<Flags>(flags_ | kHasStarAllele);
      continue;
    }

    // Compute per-allele padding starting with the "right" side to effectively left-align the variant.
    // We don't remove "right" padding when any allele has just length 1 since that can collapse insertions.
    if (ref_allele.size() > 1 && alt_allele.size() > 1) {
      auto [right_padding_iter, _right_padding_iter] =
          std::mismatch(ref_allele.rbegin(), ref_allele.rend(), alt_allele.rbegin(), alt_allele.rend());
      allele_right_padding_[i - 1] = std::distance(ref_allele.rbegin(), right_padding_iter);
    }
    right_padding = std::min(right_padding, allele_right_padding_[i - 1]);

    auto [left_padding_iter, _left_padding_iter] =
        std::mismatch(ref_allele.begin(), ref_allele.end() - allele_right_padding_[i - 1], alt_allele.begin(),
                      alt_allele.end() - allele_right_padding_[i - 1]);
    allele_left_padding_[i - 1] = std::distance(ref_allele.begin(), left_padding_iter);
    left_padding = std::min(left_padding, allele_left_padding_[i - 1]);
  }
  
  // Update overall variant padding if defined. This should exclude '*' alleles.
  if (left_padding < std::numeric_limits<Pos>::max()) {
    left_padding_ = left_padding;
  }
  if (right_padding < std::numeric_limits<Pos>::max()) {
    right_padding_ = right_padding;
  }

}

std::optional<Range> SequenceResolvedVariant::AlleleReferenceRegion(int allele_idx) const {
  if (allele_idx < 0 || allele_idx >= record_->n_allele) {
    throw std::out_of_range("Allele index out of range");
  }
  if (allele_idx == 0) {
    return std::make_optional<Range>(ReferenceRegion());
  }
  if (record_->d.allele[allele_idx][0] == '*' && record_->d.allele[allele_idx][1] == '\0') {
    return std::nullopt;  // Spanning deletion, no length change
  }
  auto pos = record_->pos;  // 0-based position
  return std::make_optional<Range>(contig(), pos + allele_left_padding_[allele_idx - 1],
                                   pos + record_->rlen - allele_right_padding_[allele_idx - 1]);
}

std::optional<int> SequenceResolvedVariant::AlleleLengthChange(int allele_idx) const {
  if (allele_idx < 0 || allele_idx >= record_->n_allele) {
    throw std::out_of_range("Allele index out of range");
  }
  if (allele_idx == 0) {
    return std::nullopt;  // Reference allele has no length change
  }
  if (record_->d.allele[allele_idx][0] == '*' && record_->d.allele[allele_idx][1] == '\0') {
    return std::nullopt;  // Spanning deletion, no length change
  }
  auto ref_allele = std::string_view(record_->d.allele[0]);
  auto alt_allele = std::string_view(record_->d.allele[allele_idx]);
  return std::make_optional<int>(alt_allele.size() - ref_allele.size());
}

std::optional<std::string_view> SequenceResolvedVariant::AlleleSequence(int allele_idx) const {
  if (allele_idx <= 0 || allele_idx >= record_->n_allele) {
    throw std::out_of_range("Allele index out of range");
  }
  auto alt_allele = std::string_view(record_->d.allele[allele_idx]);
  if (alt_allele == "*") {
    return std::nullopt;  // Spanning deletion, no allele sequence
  }
  return std::string_view(
      std::begin(alt_allele) + allele_left_padding_[allele_idx - 1],
      alt_allele.size() - allele_left_padding_[allele_idx - 1] - allele_right_padding_[allele_idx - 1]);
}

std::unique_ptr<VariantFileReader> VariantFileReader::Open(const std::string& filename) {
  FilePtr file_ptr(hts_open(filename.c_str(), "r"));
  if (!file_ptr) {
    throw std::runtime_error("Failed to open VCF/BCF file for reading: " + filename);
  }

  const htsFormat* format = hts_get_format(file_ptr.get());
  if (format->format == htsExactFormat::vcf) {
    return std::make_unique<VCFVariantFileReader>(std::move(file_ptr));
  } else if (format->format == htsExactFormat::bcf) {
    return std::make_unique<BCFVariantFileReader>(std::move(file_ptr));
  } else {
    throw std::runtime_error("Unknown file format for reading VCF/BCF file: " + filename);
  }
}

VariantFileReader::VariantFileReader(FilePtr&& file) : file_(std::move(file)) {
  hdr_ = std::make_shared<VariantFileHeader>(bcf_hdr_read(file_.get()));
  variant_offset_ = file_->is_bgzf ? bgzf_utell(file_->fp.bgzf) : htell(file_->fp.hfile);
}

htsExactFormat VariantFileReader::format() const {
  auto * hts_fmt = hts_get_format(file_.get());
  assert(hts_fmt);
  return hts_fmt->format;
}

htsCompression VariantFileReader::compression() const {
  auto * hts_fmt = hts_get_format(file_.get());
  assert(hts_fmt);
  return hts_fmt->compression;
}

std::unique_ptr<Variant> VariantFileReader::NextVariant(int max_unpack) {
  bcf1_t* record = bcf_init();
  record->max_unpack = max_unpack;
  int ret = bcf_read(file_.get(), hdr_->bcf_hdr(), record);
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

std::vector<std::string> VariantFileReader::Samples() const {
  std::vector<std::string> samples;
  for (int i = 0; i < bcf_hdr_nsamples(hdr_->bcf_hdr()); i++) {
    samples.emplace_back(hdr_->bcf_hdr()->samples[i]);
  }
  return samples;
}

void VariantFileReader::Close() {
  file_.reset();
}

VCFVariantFileReader::VCFVariantFileReader(FilePtr&& file) : VariantFileReader(std::move(file)) {
  idx_.reset(tbx_index_load3(file_->fn, NULL, HTS_IDX_SAVE_REMOTE|HTS_IDX_SILENT_FAIL));
  SetRegion();  // Default to reading the entire file
}

void VCFVariantFileReader::SetRegion() {
  if (idx_) {
    iter_.reset(tbx_itr_queryi(idx_.get(), HTS_IDX_START, 0, 0));
    if (!iter_) {
      throw std::runtime_error("Failed to set region for VCF/BCF file");
    }
  }
}

void VCFVariantFileReader::SetRegion(const Range& region) {
  if (!idx_) {
    throw std::runtime_error("Cannot set region on VCF/BCF file without index");
  }
  iter_.reset(tbx_itr_queryi(idx_.get(), tbx_name2id(idx_.get(), region.contig().c_str()), region.start(), region.end()));
  if (!iter_) {
    throw std::runtime_error("Failed to set region for VCF/BCF file");
  }
}

VariantFileReader::VariantPtr VCFVariantFileReader::NextVariant(int max_unpack) {
  kstring_t str = KS_INITIALIZE;
  boost::scope::scope_exit free_str([&str]() {  // Make sure to free kstring_t memory on all exit paths
    ks_free(&str);
  });

  int iter_ret = iter_ ? tbx_itr_next(file_.get(), idx_.get(), iter_.get(), &str) : hts_getline(file_.get(), KS_SEP_LINE, &str);
  if (iter_ret == -1) {
    return nullptr;  // End of iteration
  } else if (iter_ret < 0) {
    throw std::runtime_error("Unable to fetch next record VCF file");
  }

  bcf1_t* record = bcf_init();
  record->max_unpack = max_unpack;
  int ret = vcf_parse(&str, hdr_->bcf_hdr(), record);
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

void BCFVariantFileReader::SetRegion() { throw std::runtime_error("Not yet implemented"); }

void BCFVariantFileReader::SetRegion(const Range& region) { throw std::runtime_error("Not yet implemented"); }

VariantFileWriter::VariantFileWriterPtr VariantFileWriter::Open(const std::string& filename, const HeaderPtr& header, const char* format) {
  const char* filename_c_str = filename.c_str();
  char mode[] = {'w', '\0', '\0'};
  if (vcf_open_mode(mode + 1, filename_c_str, format) < 0) {
    throw std::runtime_error("Unknown file format for writing VCF/BCF file: " + filename);
  }

  FilePtr file_ptr(hts_open(filename_c_str, mode));
  if (!file_ptr) {
    throw std::runtime_error("Failed to open VCF/BCF file for writing: " + filename);
  }
  if (bcf_hdr_write(file_ptr.get(), header->bcf_hdr()) < 0) {
    throw std::runtime_error("Failed to write VCF/BCF header to file: " + filename);
  }
  return std::unique_ptr<VariantFileWriter>(new VariantFileWriter(std::move(file_ptr), header));
}

void VariantFileWriter::Close() {
  file_.reset();
}

void VariantFileWriter::Write(const Variant& variant) {
  if (bcf_write(file_.get(), hdr_->bcf_hdr(), variant.record_.get()) < 0) {
    throw std::runtime_error("Failed to write variant record to VCF/BCF file");
  }
}


}  // namespace npsv3