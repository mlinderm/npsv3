#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "realigner.hpp"
//#include "graph.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_native, m) {
  m.doc() = "NPSV3 native tools";

  // Realignment
  py::class_<npsv3::FragmentRealigner>(m, "FragmentRealigner")
      .def(py::init<const std::string&, double, double, py::kwargs>())
      .def("realign_read_pair", &npsv3::FragmentRealigner::RealignReadPair);

  m.def("test_score_alignment", &npsv3::test::TestScoreAlignment, "Test interface for scoring alignment");
  m.def("test_realign_read_pair", &npsv3::test::TestRealignReadPair, "Test interface for realigning reads");


  // Graph operations
  //m.def("test_kmers", &npsv3::test::TestKmers, "Test interface for kmers");
}