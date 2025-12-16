#include <sstream>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/optional.h>
#include <nanobind/operators.h>

#include "graph.hpp"

namespace nb = nanobind;
using namespace nb::literals;

class VariantFileReaderIterator {
  public:
    VariantFileReaderIterator(npsv3::VariantFileReader& reader) : reader_(reader) {}
    std::unique_ptr<npsv3::Variant> next() {
      auto variant = reader_.NextVariant();
      if (!variant) {
        throw nb::stop_iteration();
      }
      return variant;
    }
  private:
    npsv3::VariantFileReader& reader_;
};

NB_MODULE(_native_graph, m) {
  m.doc() = "NPSV3 native graph tools";

  nb::class_<npsv3::Range>(m, "Range")
      // https://nanobind.readthedocs.io/en/latest/api_core.html#_CPPv4IDpEN8nanobind4initE
    .def("__init__", [](npsv3::Range* r, const char* contig, npsv3::Pos start, npsv3::Pos end) {
      new (r) npsv3::Range(contig, start, end);
    })
    .def_prop_ro("contig", [](const npsv3::Range& r) { return r.contig().get(); })
    .def_prop_ro("start", &npsv3::Range::start)
    .def_prop_ro("end", &npsv3::Range::end)
    .def_prop_ro("length", &npsv3::Range::length)
    .def("expand", nb::overload_cast<npsv3::Pos, npsv3::Pos>(&npsv3::Range::Expand, nb::const_))
    .def("expand", nb::overload_cast<npsv3::Pos>(&npsv3::Range::Expand, nb::const_))
    .def("union_with", &npsv3::Range::UnionWith)
    .def("overlaps", &npsv3::Range::Overlaps)
    .def(nb::self == nb::self)
    .def("__str__", [](const npsv3::Range& r) {
      std::ostringstream oss;
      oss << r;
      return oss.str();
    });

  nb::class_<npsv3::Variant>(m, "Variant")
    .def_prop_ro("num_alleles", &npsv3::Variant::num_alleles)
    .def_prop_ro("variant_id", [](const npsv3::Variant& v) { return to_string(v.variant_id()); })
    .def("reference_region", &npsv3::Variant::ReferenceRegion)
    .def("allele_reference_region", &npsv3::Variant::AlleleReferenceRegion)
    .def("allele_length_change", &npsv3::Variant::AlleleLengthChange)
    .def("is_filtered", &npsv3::Variant::IsFiltered)
    .def("set_filter_pass", &npsv3::Variant::SetFilterToPass)
    .def("has_passing_genotype", nb::overload_cast<>(&npsv3::Variant::HasPassingGenotype, nb::const_))
    .def("__str__", [](const npsv3::Variant& v) {
      std::ostringstream oss;
      oss << v;
      return oss.str();
    });

  nb::class_<npsv3::VariantFileHeader>(m, "VariantFileHeader");

  nb::class_<VariantFileReaderIterator>(m, "VariantFileReaderIterator")
    .def("__iter__", [](nb::handle h) { return h; })
    .def("__next__", &VariantFileReaderIterator::next);

  nb::class_<npsv3::VariantFileReader>(m, "VariantFileReader")
    .def_static("open", &npsv3::VariantFileReader::Open)
    .def("fetch", [](npsv3::VariantFileReader& reader) {
      reader.SetRegion();
      return VariantFileReaderIterator(reader);
    }, nb::keep_alive<0, 1>()) // Keep reader alive while variant is alive
    .def("fetch", [](npsv3::VariantFileReader& reader, const npsv3::Range& region) {
      reader.SetRegion(region);
      return VariantFileReaderIterator(reader);
    }, nb::keep_alive<0, 1>()) // Keep reader alive while variant is alive
    .def("samples", &npsv3::VariantFileReader::Samples)
    .def("header", &npsv3::VariantFileReader::header);

  nb::class_<npsv3::VariantFileWriter>(m, "VariantFileWriter")
    // We seem to need this lambda to handle the optional string argument properly
    .def_static("open", [](const std::string& filename, const std::shared_ptr<npsv3::VariantFileHeader>& header, const std::optional<std::string>& format) {
      const char* format_cstr = format ? format->c_str() : nullptr;
      return npsv3::VariantFileWriter::Open(filename, header, format_cstr);
    }, "filename"_a, "header"_a, "format"_a = nb::none())
    .def("write", &npsv3::VariantFileWriter::Write);

  nb::class_<npsv3::Graph>(m, "Graph")
    .def(nb::init<const std::string&, const std::string&, const npsv3::Range&>())
    .def("node_count", &npsv3::Graph::get_node_count)
    .def("has_path", &npsv3::Graph::has_path)
    .def("path_nodes", nb::overload_cast<const std::string&>(&npsv3::Graph::PathNodes, nb::const_))
    .def("samples_including", &npsv3::Graph::SamplesIncluding)
    .def("dump", [](npsv3::Graph& graph) {
      graph.ToGFA(std::cout);
    });
}