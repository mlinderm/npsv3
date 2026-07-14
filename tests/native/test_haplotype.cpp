#include <cstdlib>
#include <filesystem>

#include <fmt/std.h>
#include <fmt/ranges.h>

#include "graph.hpp"
#include "haplotype.hpp"
#include "kmer.hpp"

#include "test_helpers.hpp"

using namespace npsv3;
using npsv3::test::GraphConstructionTest;
namespace fs = std::filesystem;

/// Classifies every k-mer with a fixed zygosity (no KMC database required).
class ConstantKmerClassify : public KmerClassify {
 public:
  explicit ConstantKmerClassify(KmerZygosity z = KmerZygosity::HOMOZYGOUS) : z_(z) {}

  void ClassifySorted(const std::vector<std::string>& sequences,
                      const ClassificationCallback& callback) const override {
    for (size_t i = 0; i < sequences.size(); ++i) {
      callback(i, z_);
    }
  }

 private:
  KmerZygosity z_;
};

void CheckKmerSubset(const HaplotypeSamplerOverlay::PathKmerMap& path_kmers,
                     const HaplotypeSamplerOverlay::KmerNodeIdSeq& kmer_path,
                     const HaplotypeSamplerOverlay::KmerIdSet& kmer_superset) {
  for (size_t start = 0; start < kmer_path.size(); ++start) {
    for (size_t end = start + 1; end <= kmer_path.size(); ++end) {
      HaplotypeSamplerOverlay::KmerNodeIdSeq kmer_sub_path(kmer_path.begin() + start, kmer_path.begin() + end);
      // The path k-mers should be superset of all sub-paths, including individual nodes
      auto sub_it = path_kmers.find(kmer_sub_path);
      if (sub_it != path_kmers.end() && !sub_it->second.kmer_set_.is_subset_of(kmer_superset)) {
        std::cerr << "Difference: " << sub_it->second.kmer_set_ - kmer_superset << std::endl;
        throw std::runtime_error(
            fmt::format("Sub-path {} k-mers are not a subset of the path k-mers.", fmt::join(kmer_sub_path, ",")));
      }
    }
  }
}

TEST_F(GraphConstructionTest, HaplotypeSamplerPathKmerDataStructure) {
  test::TestVCFFile vcf(R"VCF(##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##contig=<ID=chr1,length=248956422,md5=2648ae1bacce4ec4b6cf337dcae37816>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	Sample1
chr1	52277185	.	TT	C	.	PASS	.	GT	0/1
chr1	52277191	.	TCTATTGTTAGTAAAATAC	T	.	PASS	.	GT	0/1
)VCF");

  auto region = Range("chr1", 52277181, 52277219);
  Graph graph(HG38FastaPath_, vcf.file_path_, region);
  graph.ToGFA(std::cout);

  const size_t k = 7, max_edges = 5;
  UniqueKmersOverlay unique_kmers(graph, k, max_edges);
  HaplotypeSamplerOverlay sampler(graph, unique_kmers);

  // Paths should contain their explicit k-mers, as well as k-mers from all sub-paths, including individual nodes
  const auto& kmers_locations = unique_kmers.locations();
  const auto& path_kmers = sampler.PathKmers();

  for (size_t i = 0; i < kmers_locations.size(); ++i) {
    const auto& kmer_locations = kmers_locations[i];
    for (const auto& [handles, offset] : kmer_locations) {
      HaplotypeSamplerOverlay::KmerNodeIdSeq kmer_path(handles.size());
      std::transform(handles.begin(), handles.end(), kmer_path.begin(),
                     [&](const handlegraph::handle_t& handle) { return graph.get_id(handle); });
      auto it = path_kmers.find(kmer_path);
      ASSERT_NE(it, path_kmers.end()) << "k-mer[" << i << "]'s path not found in path_kmers";
      EXPECT_TRUE(it->second.kmer_set_.test(i)) << "k-mer[" << i << "] should be present in its path's k-mer set";

      CheckKmerSubset(path_kmers, kmer_path, it->second.kmer_set_);
    }
  }

  // Verify that all paths contain k-mers from all sub-paths, including individual nodes
  graph.for_each_path_handle([&](const handlegraph::path_handle_t& path) {
    auto path_nodes = graph.PathNodes(path);
    CheckKmerSubset(path_kmers, path_nodes, sampler.KmersOnPath(path_nodes));
  });
}

TEST_F(GraphConstructionTest, HaplotypeSamplerGreedilySelectsConsensusPath) {
  test::TestVCFFile vcf(R"VCF(##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##contig=<ID=chr1,length=248956422,md5=2648ae1bacce4ec4b6cf337dcae37816>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	Sample1
chr1	52277191	.	TCTATTGTTAGTAAAATAC	T	.	PASS	.	GT	0/1
)VCF");

  auto region = Range("chr1", 52277181, 52277219);
  Graph graph(HG38FastaPath_, vcf.file_path_, region);

  const size_t k = 7, max_edges = 5;
  UniqueKmersOverlay unique_kmers(graph, k, max_edges);
  HaplotypeSamplerOverlay sampler(graph, unique_kmers);
  
  ConstantKmerClassify counts;
  sampler.InitializeScores(counts);

  EXPECT_GT(sampler.NumKmers(), 0u);

  {
    auto paths = sampler.FindBestPaths(4 /* finite limit larger than the expected number of paths */);
    ASSERT_EQ(paths.size(), 2u)
        << "For a single bi-allelic variant, there should be exactly 2 distinct paths (ref and alt)";

    const auto & best_path = paths[0];
    ASSERT_FALSE(best_path.empty());
    EXPECT_EQ(best_path.front(), graph.min_node_id()); // Path must span the whole graph.
    EXPECT_EQ(best_path.back(),  graph.max_node_id());

    // Node IDs are in topological (ascending) order.
    EXPECT_TRUE(std::is_sorted(best_path.begin(), best_path.end()));

    // With all k-mers scored as HOMOZYGOUS, the path that maximizes k-mer coverage is the longer reference path.
    auto ref_path = graph.PathNodes("chr1");
    EXPECT_EQ(best_path, ref_path);
    EXPECT_NE(paths[1], ref_path);
  }

  {
    auto haplotypes = sampler.SampleHaplotypes(1);
    ASSERT_EQ(haplotypes.size(), 1u) << "Sampler should return exactly n paths";

    const auto & best_haplotype = haplotypes[0];
    EXPECT_EQ(best_haplotype, graph.PathNodes("chr1"));
  }
};

TEST_F(GraphConstructionTest, HaplotypeSamplerHandlesAllAbsentKmers) {
  test::TestVCFFile vcf(R"VCF(##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##contig=<ID=chr1,length=248956422,md5=2648ae1bacce4ec4b6cf337dcae37816>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	Sample1
chr1	52277191	.	TCTATTGTTAGTAAAATAC	T	.	PASS	.	GT	0/1
)VCF");

  auto region = Range("chr1", 52277181, 52277219);
  Graph graph(HG38FastaPath_, vcf.file_path_, region);

  const size_t k = 7, max_edges = 5;
  HaplotypeSamplerOverlay sampler(graph, UniqueKmersOverlay(graph, k, max_edges));

  ConstantKmerClassify counts(KmerZygosity::ABSENT);
  sampler.InitializeScores(counts);
  
  auto haplotypes = sampler.SampleHaplotypes(2);
  EXPECT_EQ(haplotypes.size(), 2u) << "Sampler should return exactly n paths";
};

TEST_F(GraphConstructionTest, HaplotypeSamplerInferenceFilteringRetainsInferenceAlleles) {
  test::TestVCFFile vcf(R"VCF(##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##contig=<ID=chr1,length=248956422,md5=2648ae1bacce4ec4b6cf337dcae37816>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	Sample1
chr1	52277185	.	TT	C	.	PASS	.	GT	0/1
chr1	52277191	.	TCTATTGTTAGTAAAATAC	T	.	PASS	.	GT	0/1
)VCF");

  auto region = Range("chr1", 52277181, 52277219);
  Graph graph(HG38FastaPath_, vcf.file_path_, region);
  graph.ToGFA(std::cout);

  const size_t k = 7, max_edges = 5;
  UniqueKmersOverlay unique_kmers(graph, k, max_edges);
  
  HaplotypeSamplerOverlay unfiltered(graph, unique_kmers);
  HaplotypeSamplerOverlay filtered(graph, unique_kmers, vcf.file_path_, region, /*min_size=*/5);
  
  ConstantKmerClassify counts;
  unfiltered.InitializeScores(counts);
  filtered.InitializeScores(counts);

  auto unfiltered_paths = unfiltered.FindBestPaths(10); // limit larger than the expected number of paths
  auto filtered_paths = filtered.FindBestPaths(10);

  ASSERT_EQ(unfiltered_paths.size(), 4u); // Two bi-allelic variants yield 4 possible paths
  ASSERT_EQ(filtered_paths.size(), 2u); // Single bi-allelic variant included in inference nodes

  // With all k-mers scored as HOMOZYGOUS, the path that maximizes k-mer coverage is the longer reference path.
  auto ref_path = graph.PathNodes("chr1");
  EXPECT_EQ(unfiltered_paths[0], ref_path);
  for (size_t i = 1; i < unfiltered_paths.size(); i++) {
    EXPECT_NE(unfiltered_paths[i], ref_path);
  }
  EXPECT_EQ(filtered_paths[0], ref_path);
  EXPECT_NE(filtered_paths[1], ref_path);

  auto small_variant_nodes = graph.PathNodes("_alt_3d867efa809c4987b04ebad884982f7788b364c8_1");
  for (size_t i = 0; i < filtered_paths.size(); i++) {
    EXPECT_FALSE(std::includes(filtered_paths[i].begin(), filtered_paths[i].end(), small_variant_nodes.begin(),
                               small_variant_nodes.end()));
  }
}

TEST_F(GraphConstructionTest, HaplotypeSamplerDecodeHaplotypeReportsCoveredAlleles) {
  test::TestVCFFile vcf(R"VCF(##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##contig=<ID=chr1,length=248956422,md5=2648ae1bacce4ec4b6cf337dcae37816>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	Sample1
chr1	52277185	.	TT	C	.	PASS	.	GT	0/1
chr1	52277191	.	TCTATTGTTAGTAAAATAC	T	.	PASS	.	GT	0/1
)VCF");

  auto region = Range("chr1", 52277181, 52277219);
  Graph graph(HG38FastaPath_, vcf.file_path_, region);
  graph.ToGFA(std::cout);
  
  const size_t k = 7, max_edges = 5;
  UniqueKmersOverlay unique_kmers(graph, k, max_edges);
  // ConstantKmerClassify counts;

  // Without inference-VCF filtering, decoding is unmasked: every variant the path traverses is reported.
  HaplotypeSamplerOverlay unfiltered(graph, unique_kmers);
  EXPECT_EQ(unfiltered.DecodeHaplotype(graph.PathNodes("chr1")), (std::vector<std::pair<std::string, size_t>>{
    {"3d867efa809c4987b04ebad884982f7788b364c8", 0}, {"70e6adb077463a6f66e692cdb11f2ca4540ff066", 0}
  }));

  // With inference-VCF filtering (min_size=5), only the large variant's alleles are ever reported: the
  // small variant's nodes fall outside the inference node mask entirely.
  HaplotypeSamplerOverlay filtered(graph, unique_kmers, vcf.file_path_, region, /*min_size=*/5);
  EXPECT_EQ(filtered.DecodeHaplotype(graph.PathNodes("chr1")), (std::vector<std::pair<std::string, size_t>>{
    {"70e6adb077463a6f66e692cdb11f2ca4540ff066", 0}
  }));
}

TEST_F(GraphConstructionTest, HaplotypeSamplerSamplesDiplotypesWithHomozygousRefWhenAllHomozygousKmers) {
  test::TestVCFFile vcf(R"VCF(##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##contig=<ID=chr1,length=248956422,md5=2648ae1bacce4ec4b6cf337dcae37816>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	Sample1
chr1	52277191	.	TCTATTGTTAGTAAAATAC	T	.	PASS	.	GT	0/1
)VCF");

  auto region = Range("chr1", 52277181, 52277219);
  Graph graph(HG38FastaPath_, vcf.file_path_, region);

  const size_t k = 7, max_edges = 5;

  HaplotypeSamplerOverlay sampler(graph, UniqueKmersOverlay(graph, k, max_edges));
  ConstantKmerClassify counts(KmerZygosity::HOMOZYGOUS);
  sampler.InitializeScores(counts);

  auto haplotypes = sampler.SampleHaplotypes(4 /* finite limit larger than the expected number of paths */);
  ASSERT_GE(haplotypes.size(), 2u);
  
  auto ref_path = graph.PathNodes("chr1");
  ASSERT_EQ(haplotypes[0], ref_path) << "First haplotype should be the reference path";
  EXPECT_NE(haplotypes[1], ref_path) << "Second haplotype should not be the reference path";

  auto diplotypes = sampler.SampleDiplotypes(haplotypes, 8 /* finite limit larger than the expected number of diplotypes */);
  ASSERT_EQ(diplotypes.size(), 3u);
  EXPECT_TRUE(diplotypes[0].h1 == 0u && diplotypes[0].h2 == 0u) << "First diplotype should 0,0 haplotypes";
  EXPECT_TRUE((diplotypes[1].h1 == 0) != (diplotypes[1].h2 == 0u)) << "Second diplotype should 0,1 haplotypes";
  EXPECT_TRUE((diplotypes[2].h1 != 0) && (diplotypes[2].h2 != 0u)) << "Third diplotype should 1,1 haplotypes";
 }

TEST_F(GraphConstructionTest, HaplotypeSamplerSamplesHaplotypesWillSelectAltPathWithAllHomozygousKmers) {
  test::TestVCFFile vcf(R"VCF(##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##contig=<ID=chr1,length=248956422,md5=2648ae1bacce4ec4b6cf337dcae37816>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	Sample1
chr1	52277185	.	T	CC	.	PASS	.	GT	0/1
chr1	52277191	.	TCTATTGTTAGTAAAATAC	T	.	PASS	.	GT	0/1
)VCF");
  auto region = Range("chr1", 52277181, 52277219);
  Graph graph(HG38FastaPath_, vcf.file_path_, region);
  graph.ToGFA(std::cout);
  const size_t k = 7, max_edges = 5;
  UniqueKmersOverlay unique_kmers(graph, k, max_edges);

  HaplotypeSamplerOverlay sampler(graph, unique_kmers, vcf.file_path_, region, /*min_size=*/5);

  ConstantKmerClassify counts;
  sampler.InitializeScores(counts);

  auto paths = sampler.FindBestPaths(10 /* finite limit larger than the expected number of paths */);
  ASSERT_EQ(paths.size(), 2u)
      << "For a single bi-allelic variant, there should be exactly 2 distinct paths (ref and alt)";

  EXPECT_NE(paths[0], graph.PathNodes("chr1"))
      << "Since alt path is longer, it should be selected for all HOMOZYGOUS k-mers";
  auto small_variant_nodes = graph.PathNodes("_alt_d1644fdd4bf6df8bf9cff96b532c07a5856cd17f_1");
  EXPECT_TRUE(std::includes(paths[0].begin(), paths[0].end(), small_variant_nodes.begin(), small_variant_nodes.end()))
      << "Selected path should include the alt variant nodes";
}

class KmersOnPathTest : public ::testing::Test {
public:
  static constexpr const char* kFastaContent = R"FASTA(>chr1
AAAAAAAAACAAAAAAAAA)FASTA";
  static constexpr const char* kVCFContent = R"VCF(##fileformat=VCFv4.2\n"
##FILTER=<ID=PASS,Description="All filters passed">
##contig=<ID=chr1>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	Sample1
chr1	10	.	C	G	.	PASS	.	GT	0/1
)VCF";

  KmersOnPathTest() : fasta_(kFastaContent), vcf_(kVCFContent), graph_(fasta_.file_path_, vcf_.file_path_, Range("chr1", 5, 15)) {}

protected:
  void SetUp() override {
    ASSERT_EQ(graph_.get_node_count(), 4u) << "Expected 4 nodes: left-flank, ref, alt, right-flank";

    auto ref_handles = graph_.PathHandles("chr1");
    ASSERT_EQ(ref_handles.size(), 3u) << "Reference path must span 3 nodes";
    h_prefix_ = ref_handles[0];
    h_ref_    = ref_handles[1];
    h_suffix_ = ref_handles[2];

    auto alt_handles = graph_.PathHandles("_alt_6821904d16079e2c0f8817ee65046a61b01bf571_1");
    ASSERT_EQ(alt_handles.size(), 1u) << "Alternative path must span 1 node";
    h_alt_ = alt_handles[0];
  }

  UniqueKmersOverlay::KmerLocation MakeLoc(std::vector<handlegraph::handle_t> handles, size_t offset = 0) {
    return UniqueKmersOverlay::KmerLocation({ std::move(handles), offset });
  }

  test::TestFastaFile fasta_;
  test::TestVCFFile vcf_;
  Graph graph_;
  handlegraph::handle_t h_prefix_, h_ref_, h_alt_, h_suffix_;
};

TEST_F(KmersOnPathTest, HaplotypeSamplerLongestPrefixMatchedOnBothBranches) {
  std::vector<std::string> sequences = {"A", "B", "C", "D", "E", "F"};
  std::vector<std::vector<UniqueKmersOverlay::KmerLocation>> locations = {
    { MakeLoc({h_prefix_}) },         // k-mer 0 (A): prefix only
    { MakeLoc({h_ref_}) },            // k-mer 1 (B): ref only
    { MakeLoc({h_alt_}) },            // k-mer 2 (C): alt only
    { MakeLoc({h_prefix_, h_ref_}) }, // k-mer 3 (D): prefix→ref 2-node edge
    { MakeLoc({h_prefix_, h_alt_}) }, // k-mer 4 (E): prefix→alt 2-node edge
    { MakeLoc({h_suffix_}) },         // k-mer 5 (F): suffix only
  };
  // After construction path_kmers_ contains (with absorbed sub-path k-mers):
  //   [prefix]     → {A}
  //   [prefix,ref] → {D,A,B} (sub-paths [prefix] and [ref] absorbed)
  //   [prefix,alt] → {E,A,C} (sub-paths [prefix] and [alt] absorbed)
  //   [ref]        → {B}
  //   [alt]        → {C}
  //   [suffix]     → {F}
  HaplotypeSamplerOverlay sampler(graph_, sequences, locations, {});

  // Reference path [prefix,ref,suffix]
  Graph::NodeIdSeq ref_nodes = {
    graph_.get_id(h_prefix_), graph_.get_id(h_ref_), graph_.get_id(h_suffix_)
  };
  auto on_ref = sampler.KmersOnPath(ref_nodes);
  EXPECT_EQ(on_ref, HaplotypeSamplerOverlay::KmerIdSet(std::string("101011"))) << "Reference path should have k-mers {A,B,D,F}";

  // Alternative path [prefix,alt,suffix]
  Graph::NodeIdSeq alt_nodes = {
    graph_.get_id(h_prefix_), graph_.get_id(h_alt_), graph_.get_id(h_suffix_)
  };
  auto on_alt = sampler.KmersOnPath(alt_nodes);
  EXPECT_EQ(on_alt, HaplotypeSamplerOverlay::KmerIdSet(std::string("110101"))) << "Alternative path should have k-mers {A,C,E,F}";
}

// Verify that KmersOnPath correctly finds the lognest matching prefix in the 
// case of partial prefix matches.
//
// The alt path [prefix,alt,suffix] has no exact or exhausted-key match at 'prefix'. 
// The predecessor to the upper_bound is [prefix,ref,suffix], which diverges at
// index 1. KmersOnPath must trim to [prefix] and find A.
TEST_F(KmersOnPathTest, HaplotypeSamplerFallsBackToCommonPrefix) {
  std::vector<std::string> sequences = {"A", "G"};
  std::vector<std::vector<UniqueKmersOverlay::KmerLocation>> locations = {
    {MakeLoc({h_prefix_})},                        // k-mer 0 (A): prefix only
    {MakeLoc({h_prefix_, h_ref_, h_suffix_})},     // k-mer 1 (G): 3-node ref path
  };
  // After construction path_kmers_ contains (with absorbed sub-path k-mers):
  //   [prefix]              → {A}
  //   [prefix,ref,suffix]   → {G,A}
  HaplotypeSamplerOverlay sampler(graph_, sequences, locations, {});

  // Reference path: query [prefix,ref,suffix]
  Graph::NodeIdSeq ref_nodes = {
    graph_.get_id(h_prefix_), graph_.get_id(h_ref_), graph_.get_id(h_suffix_)
  };
  auto on_ref = sampler.KmersOnPath(ref_nodes);
  EXPECT_EQ(on_ref, HaplotypeSamplerOverlay::KmerIdSet(std::string("11"))) << "Reference path should have k-mers {A,G}";
 
  // Alternative path [prefix,alt,suffix] should only have { A }
  Graph::NodeIdSeq alt_nodes = {
    graph_.get_id(h_prefix_), graph_.get_id(h_alt_), graph_.get_id(h_suffix_)
  };
  auto on_alt = sampler.KmersOnPath(alt_nodes);
  EXPECT_EQ(on_alt, HaplotypeSamplerOverlay::KmerIdSet(std::string("01"))) << "Alternative path should have k-mers {A}";

}

// Regression test: KmersOnPath must not assume that the common prefix shared between the query
// path and its lexicographic-predecessor key in path_kmers_ is itself a path_kmers_ entry.
TEST_F(KmersOnPathTest, HaplotypeSamplerFallsBackToCommonPrefixNotInMap) {
  std::vector<std::string> sequences = {"E"};
  std::vector<std::vector<UniqueKmersOverlay::KmerLocation>> locations = {
    {MakeLoc({h_prefix_, h_alt_})},  // k-mer 0 (E): prefix->alt 2-node edge only
  };
  // After construction path_kmers_ contains only:
  //   [prefix,alt] → {E}
  HaplotypeSamplerOverlay sampler(graph_, sequences, locations, {});

  // Reference path [prefix,ref,suffix] diverges from the stored [prefix,alt] key at index 1, and
  // [prefix] alone is not a path_kmers_ entry, so it shares no k-mers with the alt-only k-mer.
  Graph::NodeIdSeq ref_nodes = {
    graph_.get_id(h_prefix_), graph_.get_id(h_ref_), graph_.get_id(h_suffix_)
  };
  auto on_ref = sampler.KmersOnPath(ref_nodes);
  EXPECT_EQ(on_ref, HaplotypeSamplerOverlay::KmerIdSet(std::string("0"))) << "Reference path should have no k-mers";
}

// Regression test: a shorter, genuinely-matching key must not be shadowed by a longer,
// off-path key that happens to be closer (lexicographically) to the query.
TEST_F(KmersOnPathTest, HaplotypeSamplerFindsShorterPrefixMaskedByOffPathEdgeKey) {
  std::vector<std::string> sequences = {"A", "E"};
  std::vector<std::vector<UniqueKmersOverlay::KmerLocation>> locations = {
    {MakeLoc({h_prefix_})},          // k-mer 0 (A): prefix only
    {MakeLoc({h_prefix_, h_alt_})},  // k-mer 1 (E): prefix->alt 2-node edge
  };
  // After construction path_kmers_ contains:
  //   [prefix]       → {A}
  //   [prefix,alt]   → {E,A} (A absorbed into the edge key)
  HaplotypeSamplerOverlay sampler(graph_, sequences, locations, {});

  // Reference path [prefix,ref,suffix] genuinely passes through [prefix], so it should be
  // credited with A, but not E (which is confined to the alt branch).
  Graph::NodeIdSeq ref_nodes = {
    graph_.get_id(h_prefix_), graph_.get_id(h_ref_), graph_.get_id(h_suffix_)
  };
  auto on_ref = sampler.KmersOnPath(ref_nodes);
  EXPECT_EQ(on_ref, HaplotypeSamplerOverlay::KmerIdSet(std::string("01"))) << "Reference path should have k-mer {A} only";
}
