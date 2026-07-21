#pragma once
#include <fmt/base.h>

#include <cstdint>
#include <iosfwd>
#include <vector>

#include "utility.hpp"

namespace npsv3 {

struct ContigNameTag {};
using ContigName = util::FlyweightStringNoTrack<ContigNameTag>;

using Pos = uint64_t;

class Range {
 public:
  Range() : contig_(), start_(0), end_(0) {}
  Range(const ContigName& contig, Pos start, Pos end) : contig_(contig), start_(start), end_(end) {
    assert(start <= end);
  }

  const ContigName& contig() const { return contig_; }
  Pos start() const { return start_; }
  Pos end() const { return end_; }

  Pos length() const { return end_ - start_; }

  Range Expand(Pos left_flank, Pos right_flank) const;
  Range Expand(Pos flank) const {
    return Expand(flank, flank);
  }
  Range Union(const Range& other) const;
  void UnionWith(const Range& other);
  Range Intersection(const Range& other) const;

  Range Center() const {
    Pos mid = start_ + length() / 2;
    return Range(contig_, mid, mid);
  }

  std::vector<Range> Window(Pos size) const;

  bool Contains(Pos point) const { return start_ <= point && point < end_; }

  bool Overlaps(const Range& other) const {
    return contig_ == other.contig_ && start_ < other.end_ && other.start_ < end_;
  }
  bool operator==(const Range& other) const {
    return contig_ == other.contig_ && start_ == other.start_ && end_ == other.end_;
  }
  bool operator<(Pos point) const { return start_ < point; }
  
  /// Return true if this Range is a proper subset of @p other (matches Python Range.__lt__)
  bool operator<(const Range& other) const {
    return contig_ == other.contig_ &&
           ((start_ >= other.start_ && end_ < other.end_) || (start_ > other.start_ && end_ <= other.end_));
  }
  bool operator<=(const Range& other) const {
    return contig_ == other.contig_ && start_ >= other.start_ && end_ <= other.end_;
  }

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