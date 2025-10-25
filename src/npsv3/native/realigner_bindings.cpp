#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/tuple.h>

#include "realigner.hpp"

namespace nb = nanobind;

NB_MODULE(_native_realign, m) {
  m.doc() = "NPSV3 native realigner tools";

  // Realignment
  nb::class_<npsv3::FragmentRealigner>(m, "FragmentRealigner")
      .def(nb::init<const std::string&, double, double, nb::kwargs>())
      .def("realign_read_pair", &npsv3::FragmentRealigner::RealignReadPair);

  m.def("test_score_alignment", &npsv3::test::TestScoreAlignment, "Test interface for scoring alignment");
  m.def("test_realign_read_pair", &npsv3::test::TestRealignReadPair, "Test interface for realigning reads");
}