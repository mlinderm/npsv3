#include "range.hpp"

#include <fmt/format.h>

#include <ostream>

namespace npsv3 {
Range Range::Expand(Pos left_flank, Pos right_flank) const {
  Pos new_start = (left_flank > start_) ? 0 : start_ - left_flank;
  // TODO: Have some type of contig map that enforces maximum contig lengths?
  Pos new_end = end_ + right_flank;
  return Range(contig_, new_start, new_end);
}

Range Range::Union(const Range& other) const {
  if (contig_ != other.contig_) throw std::runtime_error("Cannot union ranges on different contigs");
  Pos new_start = (other.start_ < start_) ? other.start_ : start_;
  Pos new_end = (other.end_ > end_) ? other.end_ : end_;
  return Range(contig_, new_start, new_end);
}

void Range::UnionWith(const Range& other) {
  if (contig_ != other.contig_) throw std::runtime_error("Cannot union ranges on different contigs");
  if (other.start_ < start_) start_ = other.start_;
  if (other.end_ > end_) end_ = other.end_;
}

}  // namespace npsv3

auto fmt::formatter<npsv3::ContigName>::format(npsv3::ContigName c, format_context& ctx) const
    -> format_context::iterator {
  return formatter<string_view>::format(c.get(), ctx);
}