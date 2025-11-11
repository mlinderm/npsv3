#include "fasta.hpp"

namespace npsv3 {
FastaReader::FastaReader(const std::string& fasta_path) {
  // Don't create an index if it doesn't exist
  faidx_t* fai = fai_load3(fasta_path.c_str(), NULL, NULL, 0);
  if (!fai) {
    throw std::runtime_error("Failed to load FASTA index for file: " + fasta_path);
  }
  file_ = FaidxPtr(fai);
}

FastaSequence FastaReader::FetchSequence(const Range& region) {
  int64_t seq_len = 0;
  // faidx_fetch_seq* uses an inclusive end coordinate
  char* seq = faidx_fetch_seq64(file_.get(), region.contig().c_str(), region.start(), region.end()-1, &seq_len);
  if (!seq) {
    throw std::runtime_error("Failed to fetch sequence for region: " +
                             region.contig().get() + ":" +
                             std::to_string(region.start()) + "-" +
                             std::to_string(region.end()));
  }
  return FastaSequence(seq, seq_len);
}
} // namespace npsv3