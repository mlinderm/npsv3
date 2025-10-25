#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "graph.hpp"

namespace nb = nanobind;

NB_MODULE(_native_graph, m) {
  m.doc() = "NPSV3 native graph tools";

  // Graph operations
  m.def("test_create_graph", &npsv3::test::TestCreateGraph, "Test interface for creating a graph");
  //m.def("test_kmers", &npsv3::test::TestKmers, "Test interface for kmers");
}