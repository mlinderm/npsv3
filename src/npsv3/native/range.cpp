#include "range.hpp"

#include <fmt/format.h>

#include <ostream>

namespace npsv3 {
std::ostream& operator<<(std::ostream& os, const Range& range) {
  os << range.contig() << "[" << range.start() << ", " << range.end() << ")";
  return os;
}
}  // namespace npsv3

auto fmt::formatter<npsv3::ContigName>::format(npsv3::ContigName c, format_context& ctx) const
    -> format_context::iterator {
  return formatter<string_view>::format(c.get(), ctx);
}