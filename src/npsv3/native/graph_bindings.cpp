#include <algorithm>
#include <cstdint>
#include <cstring>
#include <functional>
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
#include <nanobind/stl/string_view.h>
#include <nanobind/operators.h>
#include <fmt/format.h>
#include <boost/container_hash/hash.hpp>

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

// Output streambuf that discards everything written, only counting the total number of bytes. Used as a cheap
// first pass to learn an object's exact serialized size before allocating the serialization buffer.
class CountingStreambuf : public std::streambuf {
 public:
  size_t count() const { return count_; }

 protected:
  int_type overflow(int_type ch) override {
    if (traits_type::eq_int_type(ch, traits_type::eof())) return traits_type::eof();
    ++count_;
    return ch;
  }

  std::streamsize xsputn(const char*, std::streamsize n) override {
    if (n > 0) count_ += static_cast<size_t>(n);
    return n;
  }

 private:
  size_t count_ = 0;
};

// Output streambuf that writes directly into an `nb::bytearray's` own backing memory, sized to an already-known
// exact final length so there are no reallocations. Using `bytearray` enables zero-copies during serialization at
// the C++/Python boundary. The exact size is required because any reallocations can substantially increase peak
// memory usage (both the original and new larger region need to be allocated at the same time).
class BytearrayStreambuf : public std::streambuf {
 public:
  explicit BytearrayStreambuf(size_t exact_size) : capacity_(exact_size) { buffer_.resize(exact_size); }

  // Return the backing bytearray, already at its exact final size (Finish() only exists for symmetry with
  // building on this class elsewhere; there's nothing left to shrink).
  nb::bytearray Finish() { return std::move(buffer_); }

 protected:
  int_type overflow(int_type ch) override {
    if (traits_type::eq_int_type(ch, traits_type::eof())) return traits_type::eof();
    EnsureCapacity(write_pos_ + 1);
    static_cast<char*>(buffer_.data())[write_pos_] = traits_type::to_char_type(ch);
    ++write_pos_;
    return ch;
  }

  std::streamsize xsputn(const char* s, std::streamsize n) override {
    if (n <= 0) return 0;
    const size_t count = static_cast<size_t>(n);
    EnsureCapacity(write_pos_ + count);
    std::memcpy(static_cast<char*>(buffer_.data()) + write_pos_, s, count);
    write_pos_ += count;
    return n;
  }

 private:
  void EnsureCapacity(size_t min_capacity) {
    if (min_capacity <= capacity_) {
      return;
    }
    // Should be unreachable: the counting pass and this real pass both call the same const Save(ostream&) over
    // the same unchanged object, so they must agree on length.
    throw std::runtime_error("BytearrayStreambuf: write exceeded pre-counted exact size (Save() nondeterministic?)");
  }

  nb::bytearray buffer_;
  size_t capacity_;
  size_t write_pos_ = 0;
};

// Serializes `obj` via its `Save(ostream&)` method directly into an exactly-sized Python-owned bytearray. Backs the
// save_bytes() binding on Graph, UniqueKmersOverlay, and HaplotypePriorOverlay.
template <typename T>
static nb::bytearray SaveAsBytearray(const T& obj) {
  CountingStreambuf counter;
  {
    std::ostream count_os(&counter);
    obj.Save(count_os);
  }

  BytearrayStreambuf buf(counter.count());
  std::ostream os(&buf);
  obj.Save(os);
  os.flush();
  return buf.Finish();
}


// Parse a 1-indexed fully closed region string "contig:start-end" into a Range, as accepted by
// samtools/htslib (e.g. Range("chr1:1000-2000") or Range.parse_literal("chr1:1000-2000")).
static npsv3::Range ParseRegionString(const char* region) {
  hts_pos_t beg, end;
  const char* colon = hts_parse_reg64(region, &beg, &end);
  if (colon == nullptr) {
    throw std::invalid_argument(fmt::format("Invalid region string: {}", region));
  }
  npsv3::ContigName contig(region, colon);
  return npsv3::Range(contig, static_cast<npsv3::Pos>(beg), static_cast<npsv3::Pos>(end));
}

// Convert a Genotype's packed allele indices into a Python tuple; shared by the Genotype.alleles
// property binding below.
static nb::tuple GenotypeAlleles(const npsv3::Variant::Genotype& gt) {
  const auto& idx = gt.allele_indices();
  static_assert(npsv3::Variant::Genotype::kMaxPloidy == 3,
      "GenotypeAlleles must be updated to cover all cases");
  switch (gt.num_alleles()) {
    case 1: return nb::make_tuple(idx[0]);
    case 2: return nb::make_tuple(idx[0], idx[1]);
    case 3: return nb::make_tuple(idx[0], idx[1], idx[2]);
    default: return nb::make_tuple();
  }
}

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
    // Deserialisation overloads: UniqueKmersOverlay(graph, data) loads from bytes/bytearray without copying
    // (bytes and bytearray are separate, non-inheriting types at the C level and so require distinct overloads).
    .def("__init__", [](npsv3::UniqueKmersOverlay* self, const npsv3::Graph& graph, nb::bytes data) {
      MemReadBuf buf(data.c_str(), data.size());
      std::istream is(&buf);
      npsv3::UniqueKmersOverlay::Load(self, graph, is);
    }, nb::keep_alive<1, 2>(), "graph"_a, "data"_a)
    .def("__init__", [](npsv3::UniqueKmersOverlay* self, const npsv3::Graph& graph, nb::bytearray data) {
      MemReadBuf buf(static_cast<const char*>(data.data()), data.size());
      std::istream is(&buf);
      npsv3::UniqueKmersOverlay::Load(self, graph, is);
    }, nb::keep_alive<1, 2>(), "graph"_a, "data"_a)
    .def("__len__", &npsv3::UniqueKmersOverlay::size)
    .def_prop_ro("sequences", &npsv3::UniqueKmersOverlay::sequences)
    .def("locations_as_node_ids", [](const npsv3::UniqueKmersOverlay& self, const npsv3::Graph& graph) {
      // DIAGNOSTIC: temporary, returns for each k-mer index a list of (node_id_list, offset) locations.
      std::vector<std::vector<std::pair<std::vector<odgi::nid_t>, uint64_t>>> result;
      for (const auto& kmer_locations : self.locations()) {
        std::vector<std::pair<std::vector<odgi::nid_t>, uint64_t>> locs;
        for (const auto& loc : kmer_locations) {
          std::vector<odgi::nid_t> node_ids;
          for (const auto& h : loc.handles_) node_ids.push_back(graph.get_id(h));
          locs.emplace_back(std::move(node_ids), loc.starting_handle_offset_);
        }
        result.push_back(std::move(locs));
      }
      return result;
    }, "graph"_a) // DIAGNOSTIC: temporary
    .def("save_fasta", &npsv3::UniqueKmersOverlay::SaveFasta, "fasta_path"_a)
    .def("save", nb::overload_cast<const std::string&>(&npsv3::UniqueKmersOverlay::Save, nb::const_), "path"_a)
    .def("save_bytes", [](const npsv3::UniqueKmersOverlay& overlay) {
      return SaveAsBytearray(overlay);
    })
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

  // Fixed-zygosity, DB-free KmerClassify usable anywhere a KmerClassify is expected (e.g.
  // HaplotypeSamplerOverlay.initialize_scores). For example, ConstantKmerClassify(KmerZygosity.ABSENT) makes every
  // graph-unique k-mer ABSENT.
  nb::class_<npsv3::ConstantKmerClassify, npsv3::KmerClassify>(m, "ConstantKmerClassify")
    .def("__init__", [](npsv3::ConstantKmerClassify* self, npsv3::KmerZygosity zygosity) {
      new (self) npsv3::ConstantKmerClassify(zygosity);
    }, "zygosity"_a = npsv3::KmerZygosity::HOMOZYGOUS);

  // Opaque wrapper for HaplotypePriorOverlay::PanelState. Callers only round-trip this through HaplotypePriorOverlay's
  // own methods, never construct or inspect it directly, so no constructor or fields are exposed to Python.
  nb::class_<npsv3::HaplotypePriorOverlay::PanelState>(m, "PanelState");

  // Overlay managing the population-prior "likely path state" including both full haplotypes and node-keyed
  // transitions.
  nb::class_<npsv3::HaplotypePriorOverlay>(m, "HaplotypePriorOverlay")
    // graph must outlive the overlay, so we use keep_alive<1, 2> to tie their lifetimes together. excluded_samples
    // (e.g. the sample being genotyped) is consumed entirely during construction, so it needs no keep_alive.
    .def("__init__", [](npsv3::HaplotypePriorOverlay* self, const npsv3::Graph& graph,
                         const std::vector<std::string>& excluded_samples) {
      new (self) npsv3::HaplotypePriorOverlay(graph, excluded_samples);
    }, nb::keep_alive<1, 2>(), "graph"_a, "excluded_samples"_a = std::vector<std::string>{})
    // Deserialisation overload: HaplotypePriorOverlay(graph, path) loads from a binary file
    .def("__init__", [](npsv3::HaplotypePriorOverlay* self, const npsv3::Graph& graph, const std::string& path) {
      npsv3::HaplotypePriorOverlay::Load(self, graph, path);
    }, nb::keep_alive<1, 2>(), "graph"_a, "path"_a)
    // Deserialisation overloads: HaplotypePriorOverlay(graph, data) loads from bytes/bytearray without copying
    .def("__init__", [](npsv3::HaplotypePriorOverlay* self, const npsv3::Graph& graph, nb::bytes data) {
      MemReadBuf buf(data.c_str(), data.size());
      std::istream is(&buf);
      npsv3::HaplotypePriorOverlay::Load(self, graph, is);
    }, nb::keep_alive<1, 2>(), "graph"_a, "data"_a)
    .def("__init__", [](npsv3::HaplotypePriorOverlay* self, const npsv3::Graph& graph, nb::bytearray data) {
      MemReadBuf buf(static_cast<const char*>(data.data()), data.size());
      std::istream is(&buf);
      npsv3::HaplotypePriorOverlay::Load(self, graph, is);
    }, nb::keep_alive<1, 2>(), "graph"_a, "data"_a)
    .def("find", &npsv3::HaplotypePriorOverlay::Find, "node"_a)
    .def("extend", &npsv3::HaplotypePriorOverlay::Extend, "state"_a, "node"_a)
    .def("empty", &npsv3::HaplotypePriorOverlay::Empty, "state"_a)
    .def("width", &npsv3::HaplotypePriorOverlay::Width, "state"_a)
    .def("score_to_go", &npsv3::HaplotypePriorOverlay::ScoreToGo, "state"_a)
    .def_prop_ro("node_totals", &npsv3::HaplotypePriorOverlay::node_totals)
    .def("transition_probability", &npsv3::HaplotypePriorOverlay::TransitionProbability, "from_node"_a, "to_node"_a, "alpha"_a)
    .def("log_transition_probability", &npsv3::HaplotypePriorOverlay::LogTransitionProbability, "from_node"_a, "to_node"_a, "alpha"_a)
    .def("save", nb::overload_cast<const std::string&>(&npsv3::HaplotypePriorOverlay::Save, nb::const_), "path"_a)
    .def("save_bytes", [](const npsv3::HaplotypePriorOverlay& overlay) {
      return SaveAsBytearray(overlay);
    });

  nb::class_<npsv3::HaplotypeSamplerOverlay::Diplotype>(m, "Diplotype")
    .def_prop_ro("haplotypes", [](const npsv3::HaplotypeSamplerOverlay::Diplotype& d) {
      return nb::make_tuple(d.h1, d.h2);
    })
    .def_ro("score", &npsv3::HaplotypeSamplerOverlay::Diplotype::score);

  auto sampler = nb::class_<npsv3::HaplotypeSamplerOverlay>(m, "HaplotypeSamplerOverlay");

  // Scoring/prior hyperparameters. The C++ member initializers stay the single source of truth for
  // the defaults via the compile-time constexpr below, rather than re-stated as literals.
  using Params = npsv3::HaplotypeSamplerOverlay::Params;

  constexpr Params kDefaultParams{};
  nb::class_<Params>(sampler, "Params")
    .def(nb::init<>())
    .def("__init__", [](Params* self, double homozygous_score, double absent_score, double heterozygous_score,
                         double homozygous_discount, double het_adjustment, double haplotype_prior_weight,
                         double panel_fallback_penalty, double transition_prior_alpha, size_t population_state_pool_cap) {
      new (self) Params();
      self->homozygous_score = homozygous_score;
      self->absent_score = absent_score;
      self->heterozygous_score = heterozygous_score;
      self->homozygous_discount = homozygous_discount;
      self->het_adjustment = het_adjustment;
      self->haplotype_prior_weight = haplotype_prior_weight;
      self->panel_fallback_penalty = panel_fallback_penalty;
      self->transition_prior_alpha = transition_prior_alpha;
      self->population_state_pool_cap = population_state_pool_cap;
    }, "homozygous_score"_a = kDefaultParams.homozygous_score,
       "absent_score"_a = kDefaultParams.absent_score,
       "heterozygous_score"_a = kDefaultParams.heterozygous_score,
       "homozygous_discount"_a = kDefaultParams.homozygous_discount,
       "het_adjustment"_a = kDefaultParams.het_adjustment,
       "haplotype_prior_weight"_a = kDefaultParams.haplotype_prior_weight,
       "panel_fallback_penalty"_a = kDefaultParams.panel_fallback_penalty,
       "transition_prior_alpha"_a = kDefaultParams.transition_prior_alpha,
       "population_state_pool_cap"_a = kDefaultParams.population_state_pool_cap)
    .def_rw("homozygous_score", &Params::homozygous_score)
    .def_rw("absent_score", &Params::absent_score)
    .def_rw("heterozygous_score", &Params::heterozygous_score)
    .def_rw("homozygous_discount", &Params::homozygous_discount)
    .def_rw("het_adjustment", &Params::het_adjustment)
    .def_rw("haplotype_prior_weight", &Params::haplotype_prior_weight)
    .def_rw("panel_fallback_penalty", &Params::panel_fallback_penalty)
    .def_rw("transition_prior_alpha", &Params::transition_prior_alpha)
    .def_rw("population_state_pool_cap", &Params::population_state_pool_cap);

  sampler
    // graph must outlive the overlay, so we use keep_alive<1, 2> to tie their lifetimes together. The
    // population-prior overlay is a non-owning constructor argument held for the sampler's whole lifetime, so
    // it gets its own keep_alive on whichever argument position it lands at in each overload (same rule as
    // graph). `params` is copied into the sampler, so its default shared instance needs no keep_alive.
    .def("__init__", [](npsv3::HaplotypeSamplerOverlay* self, const npsv3::Graph& graph, const npsv3::UniqueKmersOverlay& unique_kmers,
                         const npsv3::HaplotypePriorOverlay* prior, const Params* params) {
      new (self) npsv3::HaplotypeSamplerOverlay(graph, unique_kmers, prior, params ? *params : Params{});
    }, nb::keep_alive<1, 2>(), nb::keep_alive<1, 4>(),
       "graph"_a, "unique_kmers"_a, "prior"_a = nullptr, "params"_a = nullptr)
    .def("__init__", [](npsv3::HaplotypeSamplerOverlay* self, const npsv3::Graph& graph, const npsv3::UniqueKmersOverlay& unique_kmers,
                         const std::string& inference_vcf, const npsv3::Range& region, size_t min_size,
                         const npsv3::HaplotypePriorOverlay* prior, const Params* params) {
      new (self) npsv3::HaplotypeSamplerOverlay(graph, unique_kmers, inference_vcf, region, min_size, prior, params ? *params : Params{});
    }, nb::keep_alive<1, 2>(), nb::keep_alive<1, 7>(),
       "graph"_a, "unique_kmers"_a, "inference_vcf"_a, "region"_a, "min_size"_a = 50,
       "prior"_a = nullptr, "params"_a = nullptr)
    .def("initialize_scores", &npsv3::HaplotypeSamplerOverlay::InitializeScores, "counts"_a)
    .def("sample_haplotypes", &npsv3::HaplotypeSamplerOverlay::SampleHaplotypes, "n"_a)
    .def("find_best_paths", &npsv3::HaplotypeSamplerOverlay::FindBestPaths, "n"_a)
    .def("sample_diplotypes", &npsv3::HaplotypeSamplerOverlay::SampleDiplotypes, "candidates"_a, "n"_a)
    .def("score", &npsv3::HaplotypeSamplerOverlay::Score, "haplotype"_a)
    .def("decode_haplotype", nb::overload_cast<const npsv3::HaplotypeSamplerOverlay::Haplotype&>(&npsv3::HaplotypeSamplerOverlay::DecodeHaplotype, nb::const_), "haplotype"_a)
    .def("num_kmers", &npsv3::HaplotypeSamplerOverlay::NumKmers)
    .def("kmers_on_path", [](const npsv3::HaplotypeSamplerOverlay& self, const npsv3::HaplotypeSamplerOverlay::Haplotype& path) {
      // Returns the set-bit (present) k-mer indices for a path -- useful for score debugging.
      auto bits = self.KmersOnPath(path);
      std::vector<size_t> result;
      for (size_t idx = bits.find_first(); idx != npsv3::HaplotypeSamplerOverlay::KmerIdSet::npos; idx = bits.find_next(idx)) {
        result.push_back(idx);
      }
      return result;
    }, "path"_a)
    .def("kmer_score_at", &npsv3::HaplotypeSamplerOverlay::KmerScoreAt, "idx"_a)
    .def("kmer_sequence_at", &npsv3::HaplotypeSamplerOverlay::KmerSequenceAt, "idx"_a);

  nb::class_<npsv3::Range>(m, "Range")
    // https://nanobind.readthedocs.io/en/latest/api_core.html#_CPPv4IDpEN8nanobind4initE
    .def("__init__", [](npsv3::Range* r, const char* contig, npsv3::Pos start, npsv3::Pos end) {
      new (r) npsv3::Range(contig, start, end);
    })
    // Construct from a 1-indexed fully closed region string "contig:start-end"
    .def("__init__", [](npsv3::Range* r, const char* region) {
      new (r) npsv3::Range(ParseRegionString(region));
    }, "region"_a)
    // Alias for the region-string constructor, for parity with the old pysam-backed Range.parse_literal
    .def_static("parse_literal", [](const std::string& region) {
      return ParseRegionString(region.c_str());
    }, "region"_a)
    // Parse a slug of the form "contig_start_end" (as produced by the `slug` property) from the end to
    // allow contigs with underscores in their names. The slug is a 0-indexed half-open interval.
    .def_static("parse_slug", [](const std::string& slug) {
      size_t end = slug.rfind('_');
      if (end != std::string::npos) {
        size_t start = slug.rfind('_', end-1);
        if (start != std::string::npos) {
          return npsv3::Range(
            slug.substr(0, start),
            static_cast<npsv3::Pos>(std::stoull(slug.substr(start + 1, end - (start + 1)))),
            static_cast<npsv3::Pos>(std::stoull(slug.substr(end + 1)))
          );
        }
      } 
      throw std::invalid_argument(fmt::format("Invalid Range slug: {}", slug));
    }, "slug"_a)
    .def_prop_ro("contig", [](const npsv3::Range& r) { return r.contig().get(); })
    .def_prop_ro("start", &npsv3::Range::start)
    .def_prop_ro("end", &npsv3::Range::end)
    .def_prop_ro("length", &npsv3::Range::length)
    .def("__len__", &npsv3::Range::length)
    .def_prop_ro("center", &npsv3::Range::Center)
    .def_prop_ro("pysam_fetch", [](const npsv3::Range& r) {
      nb::dict d;

      d["contig"] = r.contig().get();
      d["start"] = r.start();
      d["stop"] = r.end();

      return d;
    })
    .def("expand", nb::overload_cast<npsv3::Pos, npsv3::Pos>(&npsv3::Range::Expand, nb::const_))
    .def("expand", nb::overload_cast<npsv3::Pos>(&npsv3::Range::Expand, nb::const_))
    .def("union", &npsv3::Range::Union)
    .def("union_with", &npsv3::Range::UnionWith)
    .def("intersection", &npsv3::Range::Intersection)
    .def("overlaps", &npsv3::Range::Overlaps)
    .def("contains", &npsv3::Range::Contains, "point"_a)
    .def("window", &npsv3::Range::Window, "size"_a)
    .def("get_overlap", [](const npsv3::Range& r, nb::object has_region) -> int64_t {
      if (nb::isinstance<npsv3::Range>(has_region)) {
        const auto& other = nb::cast<const npsv3::Range&>(has_region);
        if (r.contig() != other.contig()) { 
          return 0;
        }
        auto overlap_start = std::max<int64_t>(r.start(), other.start());
        auto overlap_end = std::min<int64_t>(r.end(), other.end());
        return std::max<int64_t>(0, overlap_end - overlap_start);
      }
      // Otherwise assume a pysam.AlignedSegment-like object
      if (!nb::hasattr(has_region, "reference_name")) {
        throw std::invalid_argument("Unsupported type for Range.get_overlap");
      }
      nb::object ref_name = has_region.attr("reference_name");
      if (ref_name.is_none() || nb::cast<std::string>(ref_name) != r.contig().get()) {
        return 0;
      }
      return nb::cast<int64_t>(has_region.attr("get_overlap")(r.start(), r.end()));
    }, "has_region"_a)
    .def(nb::self == nb::self)
    .def(nb::self <= nb::self)
    .def(nb::self < nb::self)
    .def("__hash__", [](const npsv3::Range& r) {
      size_t seed = 0;
      boost::hash_combine(seed, r.contig());
      boost::hash_combine(seed, r.start());
      boost::hash_combine(seed, r.end());
      return seed;
    })
    .def_prop_ro("slug", [](const npsv3::Range& r) {
      return fmt::format("{}_{}_{}", r.contig(), r.start(), r.end());
    })
    .def("__str__", [](const npsv3::Range& r) {
      // Convert to 1-based closed interval for display
      return fmt::format("{}:{}-{}", r.contig(), r.start()+1, r.end());
    });

  nb::class_<npsv3::Variant::SampleGenotype>(m, "Genotype")
    .def_prop_ro("alleles", [](const npsv3::Variant::SampleGenotype& sg) { return GenotypeAlleles(sg.genotype()); })
    .def_prop_ro("is_filtered", &npsv3::Variant::SampleGenotype::is_filtered);

  nb::class_<npsv3::Variant>(m, "Variant")
    .def_prop_ro("contig", [](const npsv3::Variant& v) { return v.contig().get(); })
    .def_prop_ro("start", &npsv3::Variant::start)
    .def_prop_ro("end", &npsv3::Variant::end)
    .def_prop_ro("num_alleles", &npsv3::Variant::num_alleles)
    .def_prop_ro("num_alts", &npsv3::Variant::num_alts)
    .def_prop_ro("ref_length", &npsv3::Variant::ref_length)
    // variant_id is recomputed (SHA1) on every access; not cached on the C++ side yet
    .def_prop_ro("variant_id", [](const npsv3::Variant& v) { return to_string(v.variant_id()); })
    .def_prop_ro("has_star_allele", [](const npsv3::Variant& v) { return v.has_flag(npsv3::Variant::kHasStarAllele); })
    .def("reference_region", &npsv3::Variant::ReferenceRegion)
    .def("record_reference_region", &npsv3::Variant::RecordReferenceRegion)
    .def("allele_reference_region", &npsv3::Variant::AlleleReferenceRegion)
    .def("allele_length_change", &npsv3::Variant::AlleleLengthChange)
    .def("allele_length", &npsv3::Variant::AlleleLength, "allele_idx"_a)
    // No-arg form returns the per-ALT list (SVLEN-aware, see LengthChanges()); the single-allele
    // overload below indexes into it, e.g. length_change(2) == length_change()[1]
    .def("length_change", &npsv3::Variant::LengthChanges)
    .def("length_change", [](const npsv3::Variant& v, int allele_idx) {
      if (allele_idx < 1 || allele_idx > v.num_alts()) {
        throw std::out_of_range("Allele index out of range");
      }
      return v.LengthChanges()[allele_idx - 1];
    }, "allele_idx"_a)
    .def("allele", [](const npsv3::Variant& v, int allele_idx) {
      return std::string(v.AlleleRawSequence(allele_idx));
    }, "allele_idx"_a)
    .def("allele_sequence", [](const npsv3::Variant& v, int allele_idx) -> std::optional<std::string> {
      auto seq = v.AlleleSequence(allele_idx);
      if (!seq) return std::nullopt;
      return std::string(*seq);
    }, "allele_idx"_a)
    .def("info_int", &npsv3::Variant::InfoInt, "key"_a)
    .def("is_filtered", &npsv3::Variant::IsFiltered)
    .def("set_filter_pass", &npsv3::Variant::SetFilterToPass)
    .def("has_passing_genotype", nb::overload_cast<>(&npsv3::Variant::HasPassingGenotype, nb::const_))
    .def("subset_samples", &npsv3::Variant::SubsetSamples)
    .def("genotype", &npsv3::Variant::genotype, "sample_idx"_a)
    .def("__str__", [](const npsv3::Variant& v) {
      std::ostringstream oss;
      oss << v;
      return oss.str();
    });

  nb::class_<npsv3::VariantFileHeader>(m, "VariantFileHeader")
    .def("subset", &npsv3::VariantFileHeader::Subset);

  nb::class_<VariantFileReaderIterator>(m, "VariantFileReaderIterator")
    .def("__iter__", [](nb::handle h) { return h; },
         nb::sig("def __iter__(self) -> VariantFileReaderIterator"))
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
      return SaveAsBytearray(graph);
    })
    .def_static("load", [](const std::string& path) { return npsv3::Graph::Load(path); }, "path"_a)
    .def_static("load_bytes", [](nb::bytes data) {
      MemReadBuf buf(data.c_str(), data.size());
      std::istream is(&buf);
      return npsv3::Graph::Load(is);
    }, "data"_a)
    .def_static("load_bytes", [](nb::bytearray data) {
      MemReadBuf buf(static_cast<const char*>(data.data()), data.size());
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
    .def_prop_ro("region", &npsv3::Graph::region)
    ;
}
