#pragma once

#include <htslib/faidx.h>

#include <memory>

#include "range.hpp"

namespace npsv3 {

namespace internal {
struct faidx_deleter {
  void operator()(faidx_t* fai) const { fai_destroy(fai); }
};

struct free_deleter {
  void operator()(char* seq) const { free(seq); }
};
}  // namespace internal

class FastaSequence {
 public:
  FastaSequence() : seq_len_(0) {}
  FastaSequence(char* seq, size_t seq_len) : seq_(seq), seq_len_(seq_len) {}
  FastaSequence(FastaSequence&& other) : seq_(std::move(other.seq_)), seq_len_(other.seq_len_) {}
  FastaSequence(const FastaSequence&) = delete;

  FastaSequence& operator=(const FastaSequence&) = delete;
  FastaSequence& operator=(FastaSequence&& other) {
    seq_ = std::move(other.seq_);
    seq_len_ = other.seq_len_;
    return *this;
  }

  std::string substr(size_t pos = 0, size_t count = std::string::npos) const {
    return std::string(seq_.get() + pos, count);
  }

 private:
  std::unique_ptr<char, internal::free_deleter> seq_;
  size_t seq_len_;
};

class FastaReader {
 public:
  FastaReader(const std::string& fasta_path);
  FastaSequence FetchSequence(const Range& region);

 private:
  typedef std::unique_ptr<faidx_t, internal::faidx_deleter> FaidxPtr;

  FaidxPtr file_;
};
}  // namespace npsv3
