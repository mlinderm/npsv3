#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "graph.hpp"

namespace nb = nanobind;

NB_MODULE(_native_graph, m) {
  m.doc() = "NPSV3 native graph tools";

  nb::class_<npsv3::Range>(m, "Range")
      // https://nanobind.readthedocs.io/en/latest/api_core.html#_CPPv4IDpEN8nanobind4initE
    .def("__init__", [](npsv3::Range* r, const char* contig, npsv3::Pos start, npsv3::Pos end) {
      new (r) npsv3::Range(contig, start, end);
    })
    .def_prop_ro("contig", [](npsv3::Range* r) { return r->contig().c_str(); })
    .def_prop_ro("start", &npsv3::Range::start)
    .def_prop_ro("end", &npsv3::Range::end)
    .def_prop_ro("length", &npsv3::Range::length);

  nb::class_<npsv3::Graph>(m, "Graph")
    .def(nb::init<const std::string&, const std::string&, const npsv3::Range&>())
    .def("node_count", &npsv3::Graph::get_node_count)
    .def("has_path", &npsv3::Graph::has_path)
    .def("path_nodes", nb::overload_cast<const std::string&>(&npsv3::Graph::PathNodes, nb::const_))
    .def("samples_including", &npsv3::Graph::SamplesIncluding);
  

  // Graph operations
  //m.def("test_create_graph", &npsv3::test::TestCreateGraph, "Test interface for creating a graph");
  //m.def("test_kmers", &npsv3::test::TestKmers, "Test interface for kmers");
}