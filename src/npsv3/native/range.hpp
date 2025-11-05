#pragma once
#include <fmt/base.h>

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

  const ContigName& contig() const { return contig_; }
  Pos start() const { return start_; }
  Pos end() const { return end_; }

  Pos length() const { return end_ - start_; }

  bool Overlaps(const Range& other) const {
    return contig_ == other.contig_ && start_ < other.end_ && other.start_ < end_;
  }
  void UnionWith(const Range& other) {
    if (contig_ != other.contig_)
      throw std::runtime_error("Cannot union ranges on different contigs");
    if (other.start_ < start_) start_ = other.start_;
    if (other.end_ > end_) end_ = other.end_;
  }

  bool operator==(const Range& other) const {
    return contig_ == other.contig_ && start_ == other.start_ && end_ == other.end_;
  }
  bool operator<(Pos point) const { return start_ < point; }
  bool operator<=(const Range& other) const {
    return contig_ == other.contig_ && start_ >= other.start_ && end_ <= other.end_;
  }

  friend std::ostream& operator<<(std::ostream&, const Range&);

 protected:
  ContigName contig_;
  Pos start_;
  Pos end_;
};

}  // namespace npsv3

// Full specialization to enable formatting of ContigName
// https://stackoverflow.com/a/67215894
template <>
struct fmt::formatter<npsv3::ContigName> : formatter<string_view> {
  auto format(npsv3::ContigName c, format_context& ctx) const -> format_context::iterator;
};