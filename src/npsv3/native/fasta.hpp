#pragma once

#include <memory>
#include <htslib/faidx.h>

#include "range.hpp"

namespace npsv3 {

namespace internal {
struct faidx_deleter {
  void operator()(faidx_t* fai) const { fai_destroy(fai); }
};
}  // namespace internal

class FastaReader {
 public:
  FastaReader(const std::string& fasta_path);
  std::string FetchSequence(const Range& region);

 private:
  typedef std::unique_ptr<faidx_t, internal::faidx_deleter> FaidxPtr;

  FaidxPtr file_;
};
}  // namespace npsv3
