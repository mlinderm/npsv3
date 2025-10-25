#pragma once
#include <cstdint>
#include <iosfwd>

#include "utility.hpp"

namespace npsv3 {

struct ContigNameTag {};
typedef util::FlyweightStringNoTrack<ContigNameTag> ContigName;

typedef uint64_t Pos;

class Range {
 public:
  Range(const ContigName& contig, Pos start, Pos end) : contig_(contig), start_(start), end_(end) {
    assert(start <= end);
  }

  const ContigName& Contig() const { return contig_; }
  Pos Start() const { return start_; }
  Pos End() const { return end_; }
  
  Pos Length() const { return end_ - start_; }

  bool operator<(Pos point) const { return start_ < point; }
  bool operator<=(const Range& other) const { return contig_ == other.contig_ && start_ >= other.start_ && end_ <= other.end_; }

  friend std::ostream& operator<<(std::ostream&, const Range&);
 protected:
  ContigName contig_;
  Pos start_;
  Pos end_;
};

}  // namespace npsv3