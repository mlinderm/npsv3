#include <sstream>
#include <streambuf>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/array.h>
#include <nanobind/operators.h>
#include <fmt/format.h>

#include "graph.hpp"
#include "kmer.hpp"
#include "haplotype.hpp"

namespace nb = nanobind;
using namespace nb::literals;

// Read-only streambuf that wraps an existing byte buffer without copying.
// The caller must ensure the buffer outlives any istream using this buf.
struct MemReadBuf : std::streambuf {
  MemReadBuf(const char* begin, size_t size) {
    char* p = const_cast<char*>(begin);
    setg(p, p, p + size);
  }
};

// // Serializes `obj` via its `Save(ostream&)` method and returns the result as a numpy uint8
// // array. Unlike `nb::bytes` (which always copies into a new, separately-allocated Python
// // object), the array is a zero-copy view over the serialized buffer: a capsule ties the
// // buffer's lifetime to the array so no C++-to-Python copy is required.
// template <typename T>
// static nb::ndarray<nb::numpy, uint8_t, nb::ndim<1>> SaveAsNdarray(const T& obj) {
//   std::ostringstream oss(std::ios::binary);
//   obj.Save(oss);
//   auto buf = std::make_unique<std::string>(std::move(oss).str());
//   size_t size = buf->size();
//   auto* data = reinterpret_cast<uint8_t*>(buf->data());
//   nb::capsule owner(buf.release(), [](void* p) noexcept {
//     delete static_cast<std::string*>(p);
//   });
//   return nb::ndarray<nb::numpy, uint8_t, nb::ndim<1>>(data, {size}, owner);
// }

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

  nb::enum_<npsv3::KmerZygosity>(m, "KmerZygosity")
    .value("ABSENT", npsv3::KmerZygosity::ABSENT)
    .value("HETEROZYGOUS", npsv3::KmerZygosity::HETEROZYGOUS)
    .value("HOMOZYGOUS", npsv3::KmerZygosity::HOMOZYGOUS)
    .value("FREQUENT", npsv3::KmerZygosity::FREQUENT);

  nb::class_<npsv3::UniqueKmersOverlay>(m, "UniqueKmersOverlay")
    // graph must outlive the overlay, so we use keep_alive<1, 2> to tie their lifetimes together
    .def("__init__", [](npsv3::UniqueKmersOverlay* self, const npsv3::Graph& graph, size_t k, size_t max_edges, bool exclude_universal, bool canonicalize, const npsv3::KmerCounts* ref_kmer_counts) {
      new (self) npsv3::UniqueKmersOverlay(graph, k, max_edges, exclude_universal, canonicalize, ref_kmer_counts);
    }, nb::keep_alive<1, 2>(), "graph"_a, "k"_a, "max_edges"_a = 1000, "exclude_universal"_a = true, "canonicalize"_a = false, "ref_kmer_counts"_a = nullptr)
    // Deserialisation overload: UniqueKmersOverlay(graph, path) loads from a binary file
    .def("__init__", [](npsv3::UniqueKmersOverlay* self, const npsv3::Graph& graph, const std::string& path) {
      npsv3::UniqueKmersOverlay::Load(self, graph, path);
    }, nb::keep_alive<1, 2>(), "graph"_a, "path"_a)
    // Deserialisation overload: UniqueKmersOverlay(graph, data) loads from bytes without copying
    .def("__init__", [](npsv3::UniqueKmersOverlay* self, const npsv3::Graph& graph, nb::bytes data) {
      MemReadBuf buf(data.c_str(), data.size());
      std::istream is(&buf);
      npsv3::UniqueKmersOverlay::Load(self, graph, is);
    }, nb::keep_alive<1, 2>(), "graph"_a, "data"_a)
    .def("__len__", &npsv3::UniqueKmersOverlay::size)
    .def_prop_ro("sequences", &npsv3::UniqueKmersOverlay::sequences)
    .def("save_fasta", &npsv3::UniqueKmersOverlay::SaveFasta, "fasta_path"_a)
    .def("save", nb::overload_cast<const std::string&>(&npsv3::UniqueKmersOverlay::Save, nb::const_), "path"_a)
    .def("save_bytes", [](const npsv3::UniqueKmersOverlay& overlay) {
      std::ostringstream oss(std::ios::binary);
      overlay.Save(oss);
      auto s = std::move(oss).str();
      return nb::bytes(s.data(), s.size());
    })
    // .def("save_ndarray", [](const npsv3::UniqueKmersOverlay& overlay) {
    //   return SaveAsNdarray(overlay);
    // })
    ;

  nb::class_<npsv3::KmerCounts>(m, "KmerCounts")
    .def("__init__", [](npsv3::KmerCounts* self, const std::string& db_path) {
      new (self) npsv3::KmerCounts(db_path);
    }, "db_path"_a)
    .def("count", [](const npsv3::KmerCounts& self, const std::string& kmer) {
      return self.Count(kmer);
    }, "kmer"_a);

  nb::class_<npsv3::KmerClassify>(m, "KmerClassify")
    .def("__init__", [](npsv3::KmerClassify* self, const std::string& db_path, double coverage) {
      new (self) npsv3::KmerClassify(db_path, coverage);
    }, "db_path"_a, "coverage"_a);

  nb::class_<npsv3::HaplotypeSamplerOverlay::Diplotype>(m, "Diplotype")
    .def_prop_ro("haplotypes", [](const npsv3::HaplotypeSamplerOverlay::Diplotype& d) {
      return nb::make_tuple(d.h1, d.h2);
    })
    .def_ro("score", &npsv3::HaplotypeSamplerOverlay::Diplotype::score);

  nb::class_<npsv3::HaplotypeSamplerOverlay>(m, "HaplotypeSamplerOverlay")
    // graph must outlive the overlay, so we use keep_alive<1, 2> to tie their lifetimes together
    .def("__init__", [](npsv3::HaplotypeSamplerOverlay* self, const npsv3::Graph& graph, const npsv3::UniqueKmersOverlay& unique_kmers) {
      new (self) npsv3::HaplotypeSamplerOverlay(graph, unique_kmers);
    }, nb::keep_alive<1, 2>(), "graph"_a, "unique_kmers"_a)
    .def("__init__", [](npsv3::HaplotypeSamplerOverlay* self, const npsv3::Graph& graph, const npsv3::UniqueKmersOverlay& unique_kmers, const std::string& inference_vcf, const npsv3::Range& region, size_t min_size) {
      new (self) npsv3::HaplotypeSamplerOverlay(graph, unique_kmers, inference_vcf, region, min_size);
    }, nb::keep_alive<1, 2>(), "graph"_a, "unique_kmers"_a, "inference_vcf"_a, "region"_a, "min_size"_a = 50)
    .def("initialize_scores", &npsv3::HaplotypeSamplerOverlay::InitializeScores, "counts"_a)
    .def("sample_haplotypes", &npsv3::HaplotypeSamplerOverlay::SampleHaplotypes, "n"_a)
    .def("sample_diplotypes", &npsv3::HaplotypeSamplerOverlay::SampleDiplotypes, "candidates"_a, "n"_a)
    .def("decode_haplotype", nb::overload_cast<const npsv3::HaplotypeSamplerOverlay::Haplotype&>(&npsv3::HaplotypeSamplerOverlay::DecodeHaplotype, nb::const_), "haplotype"_a)
    .def("num_kmers", &npsv3::HaplotypeSamplerOverlay::NumKmers);

  nb::class_<npsv3::Range>(m, "Range")
    // https://nanobind.readthedocs.io/en/latest/api_core.html#_CPPv4IDpEN8nanobind4initE
    .def("__init__", [](npsv3::Range* r, const char* contig, npsv3::Pos start, npsv3::Pos end) {
      new (r) npsv3::Range(contig, start, end);
    })
    // Construct from a 1-indexed fully closed region string "contig:start-end"
    .def("__init__", [](npsv3::Range* r, const char* region) {
      hts_pos_t beg, end;
      const char* colon = hts_parse_reg64(region, &beg, &end);
      if (colon == nullptr) {
        throw std::invalid_argument(fmt::format("Invalid region string: {}", region));
      }
      npsv3::ContigName contig(region, colon);
      new (r) npsv3::Range(contig, static_cast<npsv3::Pos>(beg), static_cast<npsv3::Pos>(end));
    }, "region"_a)
    .def_prop_ro("contig", [](const npsv3::Range& r) { return r.contig().get(); })
    .def_prop_ro("start", &npsv3::Range::start)
    .def_prop_ro("end", &npsv3::Range::end)
    .def_prop_ro("length", &npsv3::Range::length)
    .def("expand", nb::overload_cast<npsv3::Pos, npsv3::Pos>(&npsv3::Range::Expand, nb::const_))
    .def("expand", nb::overload_cast<npsv3::Pos>(&npsv3::Range::Expand, nb::const_))
    .def("union_with", &npsv3::Range::UnionWith)
    .def("overlaps", &npsv3::Range::Overlaps)
    .def(nb::self == nb::self)
    .def_prop_ro("slug", [](const npsv3::Range& r) {
      return fmt::format("{}_{}_{}", r.contig(), r.start(), r.end());
    })
    .def("__str__", [](const npsv3::Range& r) {
      // Convert to 1-based closed interval for display, which is more conventional for genomic coordinates
      return fmt::format("{}:{}-{}", r.contig(), r.start()+1, r.end());
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
    .def("subset_samples", &npsv3::Variant::SubsetSamples)
    .def("genotype", [](const npsv3::Variant& v, int sample_idx) {
      auto genotypes = v.Genotypes();
      if (sample_idx < 0 || static_cast<size_t>(sample_idx) >= genotypes.size())
        throw std::out_of_range("Sample index out of range");
      const auto& gt = genotypes[sample_idx];
      const auto& idx = gt.allele_indices();
      static_assert(npsv3::Variant::Genotype::kMaxPloidy == 3,
          "genotype() binding switch must be updated to cover all cases");
      switch (gt.num_alleles()) {
        case 1: return nb::make_tuple(idx[0]);
        case 2: return nb::make_tuple(idx[0], idx[1]);
        case 3: return nb::make_tuple(idx[0], idx[1], idx[2]);
        default: return nb::make_tuple();
      }
    }, "sample_idx"_a)
    .def("__str__", [](const npsv3::Variant& v) {
      std::ostringstream oss;
      oss << v;
      return oss.str();
    });

  nb::class_<npsv3::VariantFileHeader>(m, "VariantFileHeader")
    .def("subset", &npsv3::VariantFileHeader::Subset);

  nb::class_<VariantFileReaderIterator>(m, "VariantFileReaderIterator")
    .def("__iter__", [](nb::handle h) { return h; })
    .def("__next__", &VariantFileReaderIterator::next);

  nb::class_<npsv3::VariantFileReader>(m, "VariantFileReader")
    .def_static("open", &npsv3::VariantFileReader::Open)
    .def("__enter__", [](npsv3::VariantFileReader& reader) { return &reader; })
    .def("__exit__", [](npsv3::VariantFileReader& reader, [[maybe_unused]] nb::handle exc_type, [[maybe_unused]] nb::handle exc_value, [[maybe_unused]] nb::handle traceback) {
      reader.Close();
      return false; // Don't suppress exceptions
    }, "exc_type"_a = nb::none(), "exc_value"_a = nb::none(), "traceback"_a = nb::none())
    .def("fetch", [](npsv3::VariantFileReader& reader) {
      reader.SetRegion();
      return VariantFileReaderIterator(reader);
    }, nb::keep_alive<0, 1>()) // Keep reader alive while iterator is alive
    .def("fetch", [](npsv3::VariantFileReader& reader, const std::optional<npsv3::Range>& region) {
      if (region) {
        reader.SetRegion(*region);
      } else {
        reader.SetRegion();
      }
      return VariantFileReaderIterator(reader);
    }, nb::keep_alive<0, 1>(), "region"_a = nb::none()) // Keep reader alive while iterator is alive
    .def("samples", &npsv3::VariantFileReader::Samples)
    .def("header", &npsv3::VariantFileReader::header)
    .def("close", &npsv3::VariantFileReader::Close);

  nb::class_<npsv3::VariantFileWriter>(m, "VariantFileWriter")
    // We seem to need this lambda to handle the optional string argument properly
    .def_static("open", [](const std::string& filename, const std::shared_ptr<npsv3::VariantFileHeader>& header, const std::optional<std::string>& format) {
      const char* format_cstr = format ? format->c_str() : nullptr;
      return npsv3::VariantFileWriter::Open(filename, header, format_cstr);
    }, "filename"_a, "header"_a, "format"_a = nb::none())
    .def("write", &npsv3::VariantFileWriter::Write)
    .def("close", &npsv3::VariantFileWriter::Close);

  nb::class_<npsv3::Graph>(m, "Graph")
    .def(nb::init<const std::string&, const std::string&, const npsv3::Range&>())
    .def("save", nb::overload_cast<const std::string&>(&npsv3::Graph::Save, nb::const_), "path"_a)
    .def("save_bytes", [](const npsv3::Graph& graph) {
      std::ostringstream oss(std::ios::binary);
      graph.Save(oss);
      auto s = std::move(oss).str();
      return nb::bytes(s.data(), s.size());
    })
    // .def("save_ndarray", [](const npsv3::Graph& graph) {
    //   return SaveAsNdarray(graph);
    // })
    .def_static("load", [](const std::string& path) { return npsv3::Graph::Load(path); }, "path"_a)
    .def_static("load_bytes", [](nb::bytes data) {
      MemReadBuf buf(data.c_str(), data.size());
      std::istream is(&buf);
      return npsv3::Graph::Load(is);
    }, "data"_a)
    .def("node_count", &npsv3::Graph::get_node_count)
    .def("has_path", &npsv3::Graph::has_path)
    .def("path_nodes", nb::overload_cast<const std::string&>(&npsv3::Graph::PathNodes, nb::const_))
    .def("samples_including", &npsv3::Graph::SamplesIncluding)
    .def("dump", [](npsv3::Graph& graph) {
      graph.ToGFA(std::cout);
    })
    .def("path_sequence", [](const npsv3::Graph& graph, const npsv3::Graph::NodeIdSeq& nodes) {
      // Returns concatenated sequence of the given node IDs (null '*' nodes are skipped).
      // Converts node IDs to handles before calling PathSequence.
      npsv3::Graph::HandleSeq handles;
      handles.reserve(nodes.size());
      for (const auto& nid : nodes) {
        handles.push_back(graph.get_handle(nid));
      }
      return graph.PathSequence(handles.begin(), handles.end());
    }, "nodes"_a)
    .def("haplotype_paths", &npsv3::Graph::HaplotypePaths, "prefix"_a)
    ;
}
