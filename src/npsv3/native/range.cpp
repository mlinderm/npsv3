#include <ostream>

#include "range.hpp"


namespace npsv3 {
std::ostream& operator<<(std::ostream& os, const Range& range) {
    os << range.Contig() << "[" << range.Start() << ", " << range.End() << ")";
    return os;
}
} // namespace npsv3