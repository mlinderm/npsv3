#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <numeric>
#include <random>

#include <fmt/std.h>
#include <fmt/ranges.h>

#include "graph.hpp"
#include "haplotype.hpp"
#include "kmer.hpp"

#include "test_helpers.hpp"

using namespace npsv3;
using npsv3::test::GraphConstructionTest;
namespace fs = std::filesystem;

/// Classifies each k-mer with an independently-random zygosity from a fixed-seed RNG (no KMC database
/// required). Used to stress-test beam-search correctness at real, large-scale k-mer counts where
/// hand-listing every zygosity (as IndexedKmerClassify requires) is impractical.
class RandomKmerClassify : public KmerClassify {
 public:
  explicit RandomKmerClassify(unsigned seed) : rng_(seed) {}

  void ClassifySorted(const std::vector<std::string>& sequences,
                      const ClassificationCallback& callback) const override {
    static const KmerZygosity kZygosities[] = {KmerZygosity::ABSENT, KmerZygosity::HETEROZYGOUS, KmerZygosity::HOMOZYGOUS};
    std::uniform_int_distribution<int> dist(0, 2);
    for (size_t i = 0; i < sequences.size(); ++i) {
      callback(i, kZygosities[dist(rng_)]);
    }
  }

 private:
  mutable std::mt19937 rng_;
};

/// Classifies each k-mer according to a fixed, per-index zygosity (no KMC database required).
class IndexedKmerClassify : public KmerClassify {
 public:
  explicit IndexedKmerClassify(std::vector<KmerZygosity> zygosities) : zygosities_(std::move(zygosities)) {}

  void ClassifySorted(const std::vector<std::string>& sequences,
                      const ClassificationCallback& callback) const override {
    for (size_t i = 0; i < sequences.size(); ++i) {
      callback(i, zygosities_.at(i));
    }
  }

 private:
  std::vector<KmerZygosity> zygosities_;
};

TEST_F(GraphConstructionTest, HaplotypeSamplerFindsEveryKmerLocation) {
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

  // Every k-mer's own recorded location must be discoverable by walking that exact node sequence
  // through the automaton (via KmersOnPath), on real graph-derived data.
  const auto& kmers_locations = unique_kmers.locations();
  for (size_t i = 0; i < kmers_locations.size(); ++i) {
    for (const auto& [handles, offset] : kmers_locations[i]) {
      HaplotypeSamplerOverlay::KmerNodeIdSeq kmer_path(handles.size());
      std::transform(handles.begin(), handles.end(), kmer_path.begin(),
                     [&](const handlegraph::handle_t& handle) { return graph.get_id(handle); });
      EXPECT_TRUE(sampler.KmersOnPath(kmer_path).test(i))
          << "k-mer[" << i << "]'s own location " << fmt::format("{}", fmt::join(kmer_path, ",")) << " not found";
    }
  }
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
  // The automaton (goto/fail/output) finds every key that's a substring of the path, including via
  // failure-link outputs for keys not starting at the walk's current position:
  //   [prefix]     -> {A}
  //   [prefix,ref] -> {D}, plus {B} via [ref]'s failure-link output
  //   [prefix,alt] -> {E}, plus {C} via [alt]'s failure-link output
  //   [suffix]     -> {F}
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
  // The automaton has keys [prefix] -> {A} and [prefix,ref,suffix] -> {G} (plus {A} via [prefix]'s
  // failure-link output once [prefix,ref,suffix] is reached).
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

// Regression test: KmersOnPath must not credit a k-mer whose only recorded location diverges from
// the query path partway through (the automaton's goto for [prefix] on the ref path's next node
// must correctly fail out of the [prefix,alt] branch rather than matching it).
TEST_F(KmersOnPathTest, HaplotypeSamplerFallsBackToCommonPrefixNotInMap) {
  std::vector<std::string> sequences = {"E"};
  std::vector<std::vector<UniqueKmersOverlay::KmerLocation>> locations = {
    {MakeLoc({h_prefix_, h_alt_})},  // k-mer 0 (E): prefix->alt 2-node edge only
  };
  // The automaton has a single key [prefix,alt] -> {E}; [prefix] alone is not itself a key.
  HaplotypeSamplerOverlay sampler(graph_, sequences, locations, {});

  // Reference path [prefix,ref,suffix] diverges from the only key [prefix,alt] at index 1, and
  // [prefix] alone is not a key either, so it shares no k-mers with the alt-only k-mer.
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
  // The automaton has keys [prefix] -> {A} and [prefix,alt] -> {E} (plus {A} via [prefix]'s
  // failure-link output once [prefix,alt] is reached).
  HaplotypeSamplerOverlay sampler(graph_, sequences, locations, {});

  // Reference path [prefix,ref,suffix] genuinely passes through [prefix], so it should be
  // credited with A, but not E (which is confined to the alt branch).
  Graph::NodeIdSeq ref_nodes = {
    graph_.get_id(h_prefix_), graph_.get_id(h_ref_), graph_.get_id(h_suffix_)
  };
  auto on_ref = sampler.KmersOnPath(ref_nodes);
  EXPECT_EQ(on_ref, HaplotypeSamplerOverlay::KmerIdSet(std::string("01"))) << "Reference path should have k-mer {A} only";
}

// Regression test: PropagateBestPathState (via FindBestPaths) must not double-count a node's own
// k-mers when a later explicit multi-node edge starting at that node has already absorbed them.
//
// k-mer "P" lives on [prefix] alone, and k-mer "E" lives on the explicit edge [prefix,alt], which
// (per the absorption rule exercised above) also absorbs P into its kmer_set_. "Q" lives on [ref]
// alone and is unrelated to any explicit edge. P's score should not be double-counted on the alt path.
TEST_F(KmersOnPathTest, HaplotypeSamplerFindBestPathsDoesNotDoubleCountAbsorbedNodeKmer) {
  std::vector<std::string> sequences = {"P", "Q", "E"};
  std::vector<std::vector<UniqueKmersOverlay::KmerLocation>> locations = {
    { MakeLoc({h_prefix_}) },         // k-mer 0 (P): prefix only
    { MakeLoc({h_ref_}) },            // k-mer 1 (Q): ref only
    { MakeLoc({h_prefix_, h_alt_}) }, // k-mer 2 (E): prefix->alt 2-node edge (absorbs P)
  };

  HaplotypeSamplerOverlay::Params params;
  params.heterozygous_score = 1.0; // Score used for P and E
  params.homozygous_score = 1.2;   // Score used for Q

  HaplotypeSamplerOverlay sampler(graph_, sequences, locations, nullptr, params);
  IndexedKmerClassify counts({KmerZygosity::HETEROZYGOUS, KmerZygosity::HOMOZYGOUS, KmerZygosity::HETEROZYGOUS});
  sampler.InitializeScores(counts);

  Graph::NodeIdSeq ref_nodes = {
    graph_.get_id(h_prefix_), graph_.get_id(h_ref_), graph_.get_id(h_suffix_)
  };
  Graph::NodeIdSeq alt_nodes = {
    graph_.get_id(h_prefix_), graph_.get_id(h_alt_), graph_.get_id(h_suffix_)
  };

  // Ground truth: the ref path is genuinely the better haplotype (P present, Q present, E absent
  // beats P present, E present, Q absent, since homozygous_score > heterozygous_score).
  const double true_ref_score = sampler.Score(ref_nodes);
  const double true_alt_score = sampler.Score(alt_nodes);
  EXPECT_DOUBLE_EQ(true_ref_score, 1.2);  //  1.0 (P) + 1.2 (Q) - 1.0 (E absent)
  EXPECT_DOUBLE_EQ(true_alt_score, 0.8);  //  1.0 (P) + 1.0 (E) - 1.2 (Q absent)

  auto paths = sampler.FindBestPaths(4 /* limit larger than the expected 2 distinct paths */);
  ASSERT_EQ(paths.size(), 2u);

  EXPECT_DOUBLE_EQ(sampler.Score(paths[0]), true_ref_score);
  EXPECT_EQ(paths[0], ref_nodes);
}

// Regression test for failure to detect when traversing "plain edges" is the same as a multi-node k-mer.
// k-mer "P" lives only on the 3-node explicit edge [prefix,ref,suffix]: prefix->ref and ref->suffix
// are both ordinary graph edges in this fixture with no 2-node explicit-edge counterpart of their
// own, so this key can only ever be found by walking node-by-node through the automaton -- never via
// a single atomic 2-node hop check. k-mer "Q" lives on [alt] alone (an ordinary single-node credit).
//
// P and Q are scored so that, correctly credited:
//   true Score(ref path) = -2  (P present, Q absent)
//   true Score(alt path) = +2  (P absent, Q present)   <- alt is the genuinely better haplotype
TEST_F(KmersOnPathTest, HaplotypeSamplerFindBestPathsCreditsMultiNodeKmerViaPlainEdgeChain) {
  std::vector<std::string> sequences = {"P", "Q"};
  std::vector<std::vector<UniqueKmersOverlay::KmerLocation>> locations = {
    { MakeLoc({h_prefix_, h_ref_, h_suffix_}) }, // k-mer 0 (P): 3-node explicit edge, no shorter sub-key
    { MakeLoc({h_alt_}) },                       // k-mer 1 (Q): alt only
  };

  HaplotypeSamplerOverlay::Params params;
  params.absent_score = -3.0;       // Score used for P (classified ABSENT)
  params.heterozygous_score = -1.0; // Score used for Q (classified HETEROZYGOUS)

  HaplotypeSamplerOverlay sampler(graph_, sequences, locations, nullptr, params);
  IndexedKmerClassify counts({KmerZygosity::ABSENT, KmerZygosity::HETEROZYGOUS});
  sampler.InitializeScores(counts);

  Graph::NodeIdSeq ref_nodes = {
    graph_.get_id(h_prefix_), graph_.get_id(h_ref_), graph_.get_id(h_suffix_)
  };
  Graph::NodeIdSeq alt_nodes = {
    graph_.get_id(h_prefix_), graph_.get_id(h_alt_), graph_.get_id(h_suffix_)
  };

  // Ground truth: the alt path is genuinely the better haplotype.
  const double true_ref_score = sampler.Score(ref_nodes);
  const double true_alt_score = sampler.Score(alt_nodes);
  EXPECT_DOUBLE_EQ(true_ref_score, -2.0); // -3.0 (P present) + 1.0 (Q absent)
  EXPECT_DOUBLE_EQ(true_alt_score, 2.0);  //  3.0 (P absent) - 1.0 (Q present)

  // Fixed behavior: FindBestPaths' internal DP now credits P for the plain-edge-chain route through
  // ref via the automaton, so it correctly ranks the alt path first.
  auto paths = sampler.FindBestPaths(4 /* limit larger than the expected number of distinct paths */);
  ASSERT_EQ(paths.size(), 2u);

  EXPECT_DOUBLE_EQ(sampler.Score(paths[0]), true_alt_score);
  EXPECT_EQ(paths[0], alt_nodes);
}

// Regression test covering *overlapping* multi-node key shape: two explicit keys that share a node span, e.g.
// [prefix,ref,mid] and [ref,mid,suffix] both live near node "mid". A DP that can only take one atomic explicit-edge
// jump per transition can credit at most one of the two overlapping keys; the Aho-Corasick automaton's failure-link
// outputs must find *both*, exactly as KmersOnPath (which independently rescans every position) does.
class OverlappingKeyTest : public ::testing::Test {
 public:
  static constexpr const char* kFastaContent = R"FASTA(>chr1
AAAAAAAAACAAGAAAAAAAA)FASTA";
  static constexpr const char* kVCFContent = R"VCF(##fileformat=VCFv4.2\n"
##FILTER=<ID=PASS,Description="All filters passed">
##contig=<ID=chr1>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	Sample1
chr1	10	.	C	G	.	PASS	.	GT	0/1
chr1	12	.	A	T	.	PASS	.	GT	0/1
)VCF";

  OverlappingKeyTest() : fasta_(kFastaContent), vcf_(kVCFContent), graph_(fasta_.file_path_, vcf_.file_path_, Range("chr1", 5, 17)) {}

protected:
  void SetUp() override {
    // Expect: left-flank(prefix), ref1|alt1, mid, ref2|alt2, right-flank(suffix) -- 7 nodes total.
    ASSERT_EQ(graph_.get_node_count(), 7u);

    auto ref_handles = graph_.PathHandles("chr1");
    ASSERT_EQ(ref_handles.size(), 5u);
    h_prefix_ = ref_handles[0];
    h_ref1_   = ref_handles[1];
    h_mid_    = ref_handles[2];
    h_ref2_   = ref_handles[3];
    h_suffix_ = ref_handles[4];
  }

  UniqueKmersOverlay::KmerLocation MakeLoc(std::vector<handlegraph::handle_t> handles, size_t offset = 0) {
    return UniqueKmersOverlay::KmerLocation({ std::move(handles), offset });
  }

  test::TestFastaFile fasta_;
  test::TestVCFFile vcf_;
  Graph graph_;
  handlegraph::handle_t h_prefix_, h_ref1_, h_mid_, h_ref2_, h_suffix_;
};

TEST_F(OverlappingKeyTest, HaplotypeSamplerCreditsBothOverlappingMultiNodeKeys) {
  std::vector<std::string> sequences = {"P", "R"};
  std::vector<std::vector<UniqueKmersOverlay::KmerLocation>> locations = {
    { MakeLoc({h_prefix_, h_ref1_, h_mid_}) }, // k-mer 0 (P): 3-node key [prefix,ref1,mid]
    { MakeLoc({h_ref1_, h_mid_, h_ref2_}) },   // k-mer 1 (R): 3-node key [ref1,mid,ref2], overlaps P at [ref1,mid]
  };

  HaplotypeSamplerOverlay::Params params;
  params.absent_score = 1.0; // Score used for both P and R (classified ABSENT: present -> penalty avoided is +1 each)

  HaplotypeSamplerOverlay sampler(graph_, sequences, locations, nullptr, params);
  IndexedKmerClassify counts({KmerZygosity::ABSENT, KmerZygosity::ABSENT});
  sampler.InitializeScores(counts);

  Graph::NodeIdSeq ref_nodes = {
    graph_.get_id(h_prefix_), graph_.get_id(h_ref1_), graph_.get_id(h_mid_), graph_.get_id(h_ref2_), graph_.get_id(h_suffix_)
  };

  // Ground truth: both overlapping keys' k-mers must be found (via failure-link outputs), not just one.
  const double true_ref_score = sampler.Score(ref_nodes);
  EXPECT_DOUBLE_EQ(true_ref_score, 2.0); // both P and R present: 1.0 + 1.0

  auto on_ref = sampler.KmersOnPath(ref_nodes);
  EXPECT_TRUE(on_ref.test(0)) << "P should be found on the reference path";
  EXPECT_TRUE(on_ref.test(1)) << "R should be found on the reference path";

  // The DP (walking node-by-node through the automaton) must independently find and credit both
  // overlapping keys too, matching Score() exactly -- not just one of the two.
  auto paths = sampler.FindBestPaths(1);
  ASSERT_EQ(paths.size(), 1u);
  EXPECT_EQ(paths[0], ref_nodes);
  EXPECT_DOUBLE_EQ(sampler.Score(paths[0]), true_ref_score);
}

// Verification that an aggressively narrow beam (width 1) still finds the true global optimum.
class BeamMissStressTest : public ::testing::Test {
public:
  static constexpr const char* kFastaContent = R"FASTA(>chr1
AAAAAAAAACAAAACAAAACAAAAAAAAAA)FASTA";
  static constexpr const char* kVCFContent = R"VCF(##fileformat=VCFv4.2\n"
##FILTER=<ID=PASS,Description="All filters passed">
##contig=<ID=chr1>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	Sample1
chr1	10	.	C	G	.	PASS	.	GT	0/1
chr1	15	.	C	G	.	PASS	.	GT	0/1
chr1	20	.	C	G	.	PASS	.	GT	0/1
)VCF";

  BeamMissStressTest() : fasta_(kFastaContent), vcf_(kVCFContent), graph_(fasta_.file_path_, vcf_.file_path_, Range("chr1", 5, 25)) {}

protected:
  void SetUp() override {
    auto ref_handles = graph_.PathHandles("chr1");
    ASSERT_EQ(ref_handles.size(), 7u) << "Expect prefix, {ref1,mid1,ref2}x..., suffix along the reference path";
    h_prefix_ = ref_handles[0];
    h_ref_[0] = ref_handles[1];
    h_mid_[0] = ref_handles[2];
    h_ref_[1] = ref_handles[3];
    h_mid_[1] = ref_handles[4];
    h_ref_[2] = ref_handles[5];
    h_suffix_ = ref_handles[6];

    // Each SNP's alt allele is the *other* successor of the node preceding it (prefix for SNP0, mid0 for
    // SNP1, mid1 for SNP2) -- the one that isn't the ref allele already on the chr1 path.
    handlegraph::handle_t predecessor[3] = {h_prefix_, h_mid_[0], h_mid_[1]};
    for (int v = 0; v < 3; ++v) {
      handlegraph::handle_t found = h_ref_[v];
      graph_.follow_edges(predecessor[v], false, [&](const handlegraph::handle_t& next) {
        if (graph_.get_id(next) != graph_.get_id(h_ref_[v])) found = next;
        return true;
      });
      ASSERT_NE(graph_.get_id(found), graph_.get_id(h_ref_[v])) << "Failed to find alt allele for SNP " << v;
      h_alt_[v] = found;
    }
  }

  UniqueKmersOverlay::KmerLocation MakeLoc(std::vector<handlegraph::handle_t> handles, size_t offset = 0) {
    return UniqueKmersOverlay::KmerLocation({ std::move(handles), offset });
  }

  test::TestFastaFile fasta_;
  test::TestVCFFile vcf_;
  Graph graph_;
  handlegraph::handle_t h_prefix_, h_suffix_;
  handlegraph::handle_t h_ref_[3], h_alt_[3], h_mid_[3];
};

TEST_F(BeamMissStressTest, HaplotypeSamplerFindsGlobalOptimumUnderAggressiveTrimming) {
  // k-mer i (2*v) is the ref allele's own credit at SNP v; k-mer i+1 is the alt allele's. Scores are
  // deliberately non-monotonic across the three SNPs (ref wins SNP0, alt wins SNP1, ref wins SNP2 by a
  // wide margin) so that no single "always prefer ref" or "always prefer alt" greedy rule happens to
  // match the true optimum by coincidence.
  std::vector<std::string> sequences = {"r0", "a0", "r1", "a1", "r2", "a2"};
  std::vector<std::vector<UniqueKmersOverlay::KmerLocation>> locations = {
    { MakeLoc({h_ref_[0]}) }, { MakeLoc({h_alt_[0]}) },
    { MakeLoc({h_ref_[1]}) }, { MakeLoc({h_alt_[1]}) },
    { MakeLoc({h_ref_[2]}) }, { MakeLoc({h_alt_[2]}) },
  };

  HaplotypeSamplerOverlay sampler(graph_, sequences, locations);
  IndexedKmerClassify counts({
    KmerZygosity::HETEROZYGOUS, KmerZygosity::ABSENT,      // SNP0: ref (het, score 0) beats alt (absent, score -0.8)
    KmerZygosity::ABSENT, KmerZygosity::HETEROZYGOUS,      // SNP1: alt (het, 0) beats ref (absent, -0.8)
    KmerZygosity::ABSENT, KmerZygosity::HOMOZYGOUS,        // SNP2: alt (homozygous, 1.0) beats ref (absent, -0.8) by the widest margin
  });
  sampler.InitializeScores(counts);

  auto node_id = [&](handlegraph::handle_t h) { return graph_.get_id(h); };
  auto build_path = [&](int b0, int b1, int b2) {
    handlegraph::handle_t choice[3] = {b0 ? h_alt_[0] : h_ref_[0], b1 ? h_alt_[1] : h_ref_[1], b2 ? h_alt_[2] : h_ref_[2]};
    Graph::NodeIdSeq path = {node_id(h_prefix_)};
    path.push_back(node_id(choice[0]));
    path.push_back(node_id(h_mid_[0]));
    path.push_back(node_id(choice[1]));
    path.push_back(node_id(h_mid_[1]));
    path.push_back(node_id(choice[2]));
    path.push_back(node_id(h_suffix_));
    return path;
  };

  // Brute-force ground truth over all 8 combinations.
  double best_score = -std::numeric_limits<double>::infinity();
  Graph::NodeIdSeq best_path;
  for (int b0 = 0; b0 < 2; ++b0) {
    for (int b1 = 0; b1 < 2; ++b1) {
      for (int b2 = 0; b2 < 2; ++b2) {
        auto path = build_path(b0, b1, b2);
        double score = sampler.Score(path);
        if (score > best_score) {
          best_score = score;
          best_path = path;
        }
      }
    }
  }
  // The optimum should require choosing alt at SNP1 and SNP2 (biggest margins), ref at SNP0 -- not a
  // uniform "always ref" or "always alt" choice.
  EXPECT_EQ(best_path, build_path(0, 1, 1));

  // FindBestPaths(1) forces n=1, the narrowest possible beam, at every single node throughout the DP --
  // it should still return exactly the brute-forced optimum (see OPTIMIZATION_PROPOSALS.md's "Empirical
  // validation of Proposal 2": same-pool score comparisons are exact regardless of beam width).
  auto paths = sampler.FindBestPaths(1);
  ASSERT_EQ(paths.size(), 1u);
  EXPECT_EQ(paths[0], best_path);
  EXPECT_DOUBLE_EQ(sampler.Score(paths[0]), best_score);
}

// Generalizes BeamMissStressTest to an arbitrary-length chain of biallelic SNPs, then across many
// random adversarial per-SNP score assignments (including exact ties, which stress the beam's
// tie-breaking as directly as any margin does), brute-forces the true global optimum (2^K
// combinations) and compares it against FindBestPaths(n).
class SNPChainFixture {
 public:
  explicit SNPChainFixture(size_t num_snps)
      : fasta_(BuildFasta(num_snps)),
        vcf_(BuildVCF(num_snps)),
        graph_(fasta_.file_path_, vcf_.file_path_, Range("chr1", 5, 5 * num_snps + 10)),
        num_snps_(num_snps) {
    auto ref_handles = graph_.PathHandles("chr1");
    // prefix, ref0, mid0, ref1, mid1, ..., ref_{K-1}, suffix
    EXPECT_EQ(ref_handles.size(), 2 * num_snps + 1);

    h_prefix_ = ref_handles.front();
    h_suffix_ = ref_handles.back();
    h_ref_.resize(num_snps);
    h_alt_.resize(num_snps);
    h_mid_.resize(num_snps > 0 ? num_snps - 1 : 0);

    size_t idx = 1;
    for (size_t i = 0; i < num_snps; ++i) {
      h_ref_[i] = ref_handles[idx++];
      if (i + 1 < num_snps) h_mid_[i] = ref_handles[idx++];
    }

    // Each SNP's alt allele is the *other* successor of the node preceding it.
    for (size_t i = 0; i < num_snps; ++i) {
      handlegraph::handle_t predecessor = (i == 0) ? h_prefix_ : h_mid_[i - 1];
      handlegraph::handle_t found = h_ref_[i];
      graph_.follow_edges(predecessor, false, [&](const handlegraph::handle_t& next) {
        if (graph_.get_id(next) != graph_.get_id(h_ref_[i])) found = next;
        return true;
      });
      h_alt_[i] = found;
    }
  }

  /// Fresh sampler over this fixture's graph: one k-mer per ref/alt allele (k-mer "r{i}"/"a{i}" for SNP i).
  std::unique_ptr<HaplotypeSamplerOverlay> MakeSampler() const {
    std::vector<std::string> sequences;
    std::vector<std::vector<UniqueKmersOverlay::KmerLocation>> locations;
    for (size_t i = 0; i < num_snps_; ++i) {
      sequences.push_back("r" + std::to_string(i));
      locations.push_back({MakeLoc({h_ref_[i]})});
      sequences.push_back("a" + std::to_string(i));
      locations.push_back({MakeLoc({h_alt_[i]})});
    }
    return std::make_unique<HaplotypeSamplerOverlay>(graph_, sequences, locations);
  }

  /// Node sequence for a given combination (bit i of @p combo selects alt at SNP i, else ref).
  Graph::NodeIdSeq BuildPath(size_t combo) const {
    Graph::NodeIdSeq path = {graph_.get_id(h_prefix_)};
    for (size_t i = 0; i < num_snps_; ++i) {
      bool alt = (combo >> i) & 1u;
      path.push_back(graph_.get_id(alt ? h_alt_[i] : h_ref_[i]));
      if (i + 1 < num_snps_) path.push_back(graph_.get_id(h_mid_[i]));
    }
    path.push_back(graph_.get_id(h_suffix_));
    return path;
  }

  double BruteForceBestScore(const HaplotypeSamplerOverlay& sampler) const {
    return BruteForceTopScores(sampler, 1).front();
  }

  /// Top @p n scores among all 2^K brute-forced combinations, sorted descending.
  std::vector<double> BruteForceTopScores(const HaplotypeSamplerOverlay& sampler, size_t n) const {
    std::vector<double> scores;
    scores.reserve(size_t{1} << num_snps_);
    for (size_t combo = 0; combo < (size_t{1} << num_snps_); ++combo) {
      scores.push_back(sampler.Score(BuildPath(combo)));
    }
    std::sort(scores.begin(), scores.end(), std::greater<double>());
    scores.resize(std::min(n, scores.size()));
    return scores;
  }

 private:
  static UniqueKmersOverlay::KmerLocation MakeLoc(std::vector<handlegraph::handle_t> handles, size_t offset = 0) {
    return UniqueKmersOverlay::KmerLocation({std::move(handles), offset});
  }

  static std::string BuildFasta(size_t num_snps) {
    std::string seq = "AAAAAAAAA"; // 9-base left flank
    for (size_t i = 0; i < num_snps; ++i) {
      seq += "C";
      seq += (i + 1 < num_snps) ? "AAAA" : "AAAAAAAAAA"; // 4-base spacer, 10-base right flank after the last SNP
    }
    return ">chr1\n" + seq;
  }

  static std::string BuildVCF(size_t num_snps) {
    std::string vcf = R"VCF(##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##contig=<ID=chr1>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	Sample1
)VCF";
    for (size_t i = 0; i < num_snps; ++i) {
      vcf += "chr1\t" + std::to_string(10 + 5 * i) + "\t.\tC\tG\t.\tPASS\t.\tGT\t0/1\n";
    }
    return vcf;
  }

  test::TestFastaFile fasta_;
  test::TestVCFFile vcf_;
  Graph graph_;
  size_t num_snps_;
  handlegraph::handle_t h_prefix_, h_suffix_;
  std::vector<handlegraph::handle_t> h_ref_, h_alt_, h_mid_;
};

TEST(HaplotypeSamplerOptimalityTest, MatchesBruteForceWidthOneAcrossRandomAdversarialScores) {
  constexpr size_t kNumSnps = 9;  // 2^9 = 512 combinations, cheap to brute force
  constexpr int kNumTrials = 300;

  SNPChainFixture fixture(kNumSnps);
  auto sampler = fixture.MakeSampler();

  static const KmerZygosity kZygosities[] = {KmerZygosity::ABSENT, KmerZygosity::HETEROZYGOUS, KmerZygosity::HOMOZYGOUS};
  std::mt19937 rng(12345);  // fixed seed: deterministic, reproducible across runs
  std::uniform_int_distribution<int> zyg_dist(0, 2);

  for (int trial = 0; trial < kNumTrials; ++trial) {
    std::vector<KmerZygosity> zygosities(2 * kNumSnps);
    for (auto& z : zygosities) z = kZygosities[zyg_dist(rng)];
    IndexedKmerClassify counts(zygosities);
    sampler->InitializeScores(counts);

    const double brute_force_best = fixture.BruteForceBestScore(*sampler);

    auto found = sampler->FindBestPaths(1);
    ASSERT_EQ(found.size(), 1u) << "trial " << trial;
    EXPECT_DOUBLE_EQ(sampler->Score(found[0]), brute_force_best)
        << "FindBestPaths(1) diverged from brute-force ground truth; trial " << trial;
  }
}

TEST(HaplotypeSamplerOptimalityTest, MatchesBruteForceWidthTwoAcrossRandomAdversarialScores) {
  constexpr size_t kNumSnps = 6;  // 2^6 = 64 combinations
  constexpr int kNumTrials = 150;

  SNPChainFixture fixture(kNumSnps);
  auto sampler = fixture.MakeSampler();

  static const KmerZygosity kZygosities[] = {KmerZygosity::ABSENT, KmerZygosity::HETEROZYGOUS, KmerZygosity::HOMOZYGOUS};
  std::mt19937 rng(67890);  // different fixed seed than the n=1 test
  std::uniform_int_distribution<int> zyg_dist(0, 2);

  for (int trial = 0; trial < kNumTrials; ++trial) {
    std::vector<KmerZygosity> zygosities(2 * kNumSnps);
    for (auto& z : zygosities) z = kZygosities[zyg_dist(rng)];
    IndexedKmerClassify counts(zygosities);
    sampler->InitializeScores(counts);

    auto brute_top2 = fixture.BruteForceTopScores(*sampler, 2);
    ASSERT_EQ(brute_top2.size(), 2u) << "trial " << trial;

    auto found = sampler->FindBestPaths(2);
    ASSERT_EQ(found.size(), 2u) << "trial " << trial;

    // FindBestPaths(n) already returns results sorted by descending score.
    for (size_t i = 0; i < 2; ++i) {
      EXPECT_DOUBLE_EQ(sampler->Score(found[i]), brute_top2[i])
          << "FindBestPaths(2)'s top-" << i << " diverged from brute-force ground truth; trial " << trial;
    }
  }
}

// -----------------------------------------------------------------------------------------------
// Fixture for pending-pool width-limit stress tests. Structurally similar to SNPChainFixture (a chain
// of K biallelic SNPs, brute-forceable via 2^K combinations), but with:
//
// 1. K-mers are auto-derived from the graph via UniqueKmersOverlay (production-shaped), not hand-listed
//    one-node-per-allele locations. With k large enough to span more than one SNP's worth of sequence,
//    this naturally creates overlapping multi-node k-mers whose automaton states stay pending across
//    multiple SNPs.
// 2. The inter-SNP spacer is a non-repetitive sequence, not SNPChainFixture's "AAAA" run. UniqueKmersOverlay
//    only keeps *sequence-unique* k-mers; an all-"AAAA" spacer makes many auto-derived k-mer windows
//    literally identical text at different genomic offsets (the same period-5 "C,AAAA" pattern repeats
//    once per SNP), so almost everything gets filtered out as non-unique before it ever reaches the DP.
class OverlappingSNPClusterFixture {
 public:
  explicit OverlappingSNPClusterFixture(size_t num_snps)
      : fasta_(BuildFasta(num_snps)),
        vcf_(BuildVCF(num_snps)),
        graph_(fasta_.file_path_, vcf_.file_path_, Range("chr1", 5, 10 + 5 * num_snps + 15)),
        num_snps_(num_snps) {
    auto ref_handles = graph_.PathHandles("chr1");
    EXPECT_EQ(ref_handles.size(), 2 * num_snps + 1);

    h_prefix_ = ref_handles.front();
    h_suffix_ = ref_handles.back();
    h_ref_.resize(num_snps);
    h_alt_.resize(num_snps);
    h_mid_.resize(num_snps > 0 ? num_snps - 1 : 0);

    size_t idx = 1;
    for (size_t i = 0; i < num_snps; ++i) {
      h_ref_[i] = ref_handles[idx++];
      if (i + 1 < num_snps) h_mid_[i] = ref_handles[idx++];
    }
    for (size_t i = 0; i < num_snps; ++i) {
      handlegraph::handle_t predecessor = (i == 0) ? h_prefix_ : h_mid_[i - 1];
      handlegraph::handle_t found = h_ref_[i];
      graph_.follow_edges(predecessor, false, [&](const handlegraph::handle_t& next) {
        if (graph_.get_id(next) != graph_.get_id(h_ref_[i])) found = next;
        return true;
      });
      h_alt_[i] = found;
    }
  }

  /// Fresh sampler with k-mers auto-derived from the graph
  std::unique_ptr<HaplotypeSamplerOverlay> MakeSampler(size_t k, size_t max_edges = 5) const {
    UniqueKmersOverlay unique_kmers(graph_, k, max_edges, /*exclude_universal=*/true, /*canonicalize=*/false);
    return std::make_unique<HaplotypeSamplerOverlay>(graph_, unique_kmers);
  }

  Graph::NodeIdSeq BuildPath(size_t combo) const {
    Graph::NodeIdSeq path = {graph_.get_id(h_prefix_)};
    for (size_t i = 0; i < num_snps_; ++i) {
      bool alt = (combo >> i) & 1u;
      path.push_back(graph_.get_id(alt ? h_alt_[i] : h_ref_[i]));
      if (i + 1 < num_snps_) path.push_back(graph_.get_id(h_mid_[i]));
    }
    path.push_back(graph_.get_id(h_suffix_));
    return path;
  }

  std::vector<double> BruteForceTopScores(const HaplotypeSamplerOverlay& sampler, size_t n) const {
    std::vector<double> scores;
    scores.reserve(size_t{1} << num_snps_);
    for (size_t combo = 0; combo < (size_t{1} << num_snps_); ++combo) {
      scores.push_back(sampler.Score(BuildPath(combo)));
    }
    std::sort(scores.begin(), scores.end(), std::greater<double>());
    scores.resize(std::min(n, scores.size()));
    return scores;
  }

 private:
  // A long, non-repetitive base sequence sliced into per-SNP spacers below.
  static constexpr const char* kBase =
      "GCATGCTAGGATCCAGTGCATCGGATTACAGGTCAAGCTTGGCATTAGGCTGACCATTGGA"
      "CATGGTACCGAGCTCGAATTCACTGGCCGTCGTTTTACAACGTCGTGACTGGGAAAACCCT";

  static std::string BuildFasta(size_t num_snps) {
    std::string base(kBase);
    std::string seq = base.substr(0, 9);  // 9-base left flank
    size_t offset = 9;
    for (size_t i = 0; i < num_snps; ++i) {
      seq += "C";  // REF allele, matches BuildVCF's REF=C below
      size_t spacer_len = (i + 1 < num_snps) ? 4 : 10;
      seq += base.substr(offset, spacer_len);
      offset += spacer_len;
    }
    return ">chr1\n" + seq;
  }

  static std::string BuildVCF(size_t num_snps) {
    std::string vcf = R"VCF(##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##contig=<ID=chr1>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	Sample1
)VCF";
    for (size_t i = 0; i < num_snps; ++i) {
      vcf += "chr1\t" + std::to_string(10 + 5 * i) + "\t.\tC\tG\t.\tPASS\t.\tGT\t0/1\n";
    }
    return vcf;
  }

  test::TestFastaFile fasta_;
  test::TestVCFFile vcf_;
  Graph graph_;
  size_t num_snps_;
  handlegraph::handle_t h_prefix_, h_suffix_;
  std::vector<handlegraph::handle_t> h_ref_, h_alt_, h_mid_;
};

// Ground-truth regression test for width-trimming pending automaton-state pools. Uses 
// OverlappingSNPClusterFixture specifically to put more than n distinct covered_paths
// classes into a single pending pool before trimming (unlike SNPChainFixture's
// single-node locations, which only ever produce ephemeral one-hop pending states).
TEST(PendingPoolWidthLimitTest, MatchesBruteForceAcrossRandomAdversarialScores) {
  constexpr size_t kNumSnps = 8;      // 2^8 = 256 combinations, cheap to brute force
  constexpr size_t kKmerLength = 14;  // spans ~2-3 SNPs at this fixture's 5bp-per-SNP spacing
  constexpr int kNumTrials = 60;
  constexpr size_t kWidths[] = {1, 2, 3};

  OverlappingSNPClusterFixture fixture(kNumSnps);
  auto sampler = fixture.MakeSampler(kKmerLength);
  ASSERT_GT(sampler->NumKmers(), 2 * kNumSnps) << "Expected overlapping multi-node k-mers beyond one per allele";

  for (size_t n : kWidths) {
    for (int trial = 0; trial < kNumTrials; ++trial) {
      RandomKmerClassify counts(1000u * static_cast<unsigned>(n) + static_cast<unsigned>(trial));
      sampler->InitializeScores(counts);

      const auto brute_top = fixture.BruteForceTopScores(*sampler, n);
      auto found = sampler->FindBestPaths(n);

      // Tolerance for floating-point summation-order noise between near-tied paths (the DP's incremental
      // score accumulation vs. Score()'s independent summation can legitimately differ in the last few
      // bits) -- same rationale as RealRegionSanityTest::CheckSane's identical tolerance above.
      constexpr double kScoreTolerance = 1e-6;
      for (size_t i = 0; i < brute_top.size(); ++i) {
        ASSERT_LT(i, found.size()) << "n=" << n << " trial " << trial;
        EXPECT_NEAR(sampler->Score(found[i]), brute_top[i], kScoreTolerance)
            << "n=" << n << " trial " << trial << ": FindBestPaths(n) rank " << i << " diverged from brute force";
      }
    }
  }
}

// Exercise SampleHaplotypes' multi-draw loop, which mutates k-mer scores before
// the next draw runs. Two freshly-constructed samplers given identical initial scores must produce
// identical draws (the algorithm is deterministic, not order-dependent on anything outside graph/scores).
// Note: haplotype-by-haplotype Score() calls made *after* the loop finishes reflect the final,
// fully-mutated k-mer scores, not each draw's scores at selection time, so those can't be used to check a
// "scores non-increasing across draws" property after the fact. Only equality between two runs is checked.
TEST(SampleHaplotypesSanityTest, DeterministicAcrossRandomAdversarialScores) {
  constexpr size_t kNumSnps = 9;
  constexpr int kNumTrials = 100;
  constexpr size_t kMaxHaplotypes = 6;  // matches haplotype.py's max_haplotypes default

  SNPChainFixture fixture(kNumSnps);

  static const KmerZygosity kZygosities[] = {KmerZygosity::ABSENT, KmerZygosity::HETEROZYGOUS, KmerZygosity::HOMOZYGOUS};
  std::mt19937 rng(24680);  // different fixed seed than the single-call optimality tests above
  std::uniform_int_distribution<int> zyg_dist(0, 2);

  for (int trial = 0; trial < kNumTrials; ++trial) {
    std::vector<KmerZygosity> zygosities(2 * kNumSnps);
    for (auto& z : zygosities) z = kZygosities[zyg_dist(rng)];
    IndexedKmerClassify counts(zygosities);

    // Two independent samplers over the identical graph/initial scores. This checks that
    // the algorithm itself is deterministic, not dependent on anything outside (graph, scores) that differs.
    auto sampler_a = fixture.MakeSampler();
    sampler_a->InitializeScores(counts);
    auto haplotypes_a = sampler_a->SampleHaplotypes(kMaxHaplotypes);

    auto sampler_b = fixture.MakeSampler();
    sampler_b->InitializeScores(counts);
    auto haplotypes_b = sampler_b->SampleHaplotypes(kMaxHaplotypes);

    ASSERT_EQ(haplotypes_a.size(), haplotypes_b.size()) << "trial " << trial;
    for (size_t i = 0; i < haplotypes_a.size(); ++i) {
      EXPECT_EQ(haplotypes_a[i], haplotypes_b[i])
          << "draw " << i << " differed between two identically-initialized samplers; trial " << trial;
    }
  }
}

// Test on real, production-scale graphs instead of a synthetic SNP chain. The chr1:148538120-148538280
// regions in HG00733 has 159 nodes and 1,714 k-mers. There's no tractable brute force here, so this
// test checks that FindBestPaths(n) returns up to n valid, distinct, descending-sorted paths without
// crashing. Real k-mer coverage counts aren't needed to stress the beam search itself, so scores are
// assigned via RandomKmerClassify instead of a KMC database, matching this file's existing no-KMC-required
// test pattern.
namespace {
const char* const kHG00733PopulationVCF =
    "/storage/mlinderman/projects/sv/npsv3-experiments/resources/"
    "HG00733.hgsvc3-hprc-2024-02-23.dipcall.population.passing.hg38.vcf.gz";
}  // namespace

class RealRegionSanityTest : public GraphConstructionTest {
 protected:
  void SetUp() override {
    GraphConstructionTest::SetUp();
    if (IsSkipped()) return;
    if (!fs::exists(kHG00733PopulationVCF)) {
      GTEST_SKIP() << "HG00733 population VCF not found: " << kHG00733PopulationVCF;
    }
  }

  /// Run FindBestPaths(n) across @p num_trials random score assignments, for each width in @p widths,
  /// asserting it returns up to n valid, distinct, descending-sorted paths with no crash/hang.
  void CheckSane(HaplotypeSamplerOverlay& sampler, const std::vector<size_t>& widths, int num_trials,
                unsigned seed, const std::string& label) {
    for (size_t n : widths) {
      for (int trial = 0; trial < num_trials; ++trial) {
        RandomKmerClassify counts(seed + static_cast<unsigned>(n) * 100000u + static_cast<unsigned>(trial));
        sampler.InitializeScores(counts);

        auto paths = sampler.FindBestPaths(n);
        ASSERT_LE(paths.size(), n)
            << label << ": n=" << n << " trial " << trial << ": returned more than n paths";
        for (size_t i = 1; i < paths.size(); ++i) {
          // Tolerance for floating-point summation-order noise between two near-tied paths' scores
          // (e.g. 58.400000000000404 vs. 58.400000000000432) -- not a meaningful ordering violation.
          EXPECT_GE(sampler.Score(paths[i - 1]), sampler.Score(paths[i]) - 1e-6)
              << label << ": n=" << n << " trial " << trial << ": rank " << i << " scored higher than rank "
              << (i - 1);
          EXPECT_NE(paths[i - 1], paths[i])
              << label << ": n=" << n << " trial " << trial << ": ranks " << (i - 1) << "/" << i
              << " are the same path";
        }
      }
    }
  }

};

TEST_F(RealRegionSanityTest, FindBestPathsSaneOnSmallBenchmarkRegion) {
  auto region = Range("chr1", 148538120, 148538280);
  Graph graph(HG38FastaPath_, kHG00733PopulationVCF, region);

  const size_t k = 31, max_edges = 5;
  UniqueKmersOverlay unique_kmers(graph, k, max_edges);
  HaplotypeSamplerOverlay sampler(graph, unique_kmers);
  ASSERT_GT(sampler.NumKmers(), 0u);

  CheckSane(sampler, /*widths=*/{1, 8}, /*num_trials=*/30, /*seed=*/1, "SmallBenchmarkRegion");
}

TEST_F(RealRegionSanityTest, HaplotypeSamplerPopulationPriorStressOnDenseStage2Region) {
  // Confirm the new (automaton_state, population_state) pool growth doesn't reproduce the OOM pattern in a 521
  // variants/659b region. This exercises the *online* per-sample DP pool, not just HaplotypePriorOverlay's own offline
  // construction. Run wrapped in `ulimit -v 64000000` to ensure memory-safety, e.g.: 
  // ( ulimit -v 64000000; exec ctest --test-dir $EXT_BUILD_DIR -R 'HaplotypeSamplerPopulationPriorStressOnDenseStage2Region' )
  auto region = Range("chr1", 31431661, 31432319);
  Graph graph(HG38FastaPath_, kHG00733PopulationVCF, region);

  const size_t k = 31, max_edges = 5;
  UniqueKmersOverlay unique_kmers(graph, k, max_edges);

  HaplotypePriorOverlay prior(graph);

  HaplotypeSamplerOverlay::Params params{};
  params.haplotype_prior_weight = 0.5;
  params.panel_fallback_penalty = -2.0;
  params.transition_prior_alpha = 1.0;
  params.population_state_pool_cap = 8;  // default; exercised explicitly here rather than left implicit

  HaplotypeSamplerOverlay sampler(graph, unique_kmers, &prior, params);
  ASSERT_GT(sampler.NumKmers(), 0u);

  CheckSane(sampler, /*widths=*/{1, 8}, /*num_trials=*/30, /*seed=*/1, "DenseStage2RegionWithPopulationPrior");
}

TEST_F(RealRegionSanityTest, HaplotypePriorOverlayConstructsOnSmallBenchmarkRegion) {
  // The "small" benchmark region (chr1:148538120-148538280, 129 graph nodes, panel K=212, 1,651 induced non-empty
  // states).
  auto region = Range("chr1", 148538120, 148538280);
  Graph graph(HG38FastaPath_, kHG00733PopulationVCF, region);

  HaplotypePriorOverlay overlay(graph);

  // Walk the reference contig's own node sequence (spans the whole region) via Find() once at its first
  // node, then Extend() thereafter. Checks Width()/ScoreToGo() stay well-formed throughout a real,
  // densely-variant region.
  auto ref_nodes = graph.PathNodes(region.contig());
  ASSERT_FALSE(ref_nodes.empty());
  auto state = overlay.Find(ref_nodes.front());
  bool saw_nonempty = !overlay.Empty(state);
  for (size_t i = 1; i < ref_nodes.size() && !overlay.Empty(state); i++) {
    state = overlay.Extend(state, ref_nodes[i]);
    if (overlay.Empty(state)) break;
    saw_nonempty = true;
    auto width = overlay.Width(state);
    EXPECT_GT(width, 0u) << "position " << i;
    EXPECT_LT(width, 1000u) << "position " << i << ": implausibly large panel width for this cohort";
    EXPECT_LE(overlay.ScoreToGo(state), 0.0) << "position " << i << ": score-to-go should never be positive "
                                              << "(every continuation's width is <= the current state's)";
  }
  EXPECT_TRUE(saw_nonempty);
}

namespace {

// V1 (biallelic SNV) is followed immediately by V2 (triallelic SNV); path ids are assigned sequentially
// during Graph construction (1 = reference contig path, then each variant's REF/ALT allele paths in VCF
// order), so with these two variants: 2 = V1 REF (r1), 3 = V1 ALT (a1), 4 = V2 REF (r2), 5 = V2 ALT1 (a2),
// 6 = V2 ALT2 (a3, never taken by any genotype below -- the cold-start case).
constexpr size_t kR1 = 2, kA1 = 3, kR2 = 4, kA2 = 5, kA3 = 6;

// 7 samples: 3 homozygous-REF (both variants), 3 homozygous-ALT1 (both variants), and 1 "crossed" sample
// whose two haplotypes take (r1, a2) and (a1, r2) -- giving asymmetric-enough, hand-computable transition
// counts (r1: 6x r1->r2, 1x r1->a2; a1: 1x a1->r2, 6x a1->a2) while exercising a non-trivial multi-allelic
// (triallelic V2) case and a cold-start allele (a3) that's never observed.
test::TestVCFFile MakeTransitionOverlayFixtureVCF() {
  return test::TestVCFFile(R"VCF(##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##contig=<ID=chr1,length=248956422>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	Sample1	Sample2	Sample3	Sample4	Sample5	Sample6	Sample7
chr1	1000000	.	G	A	100	PASS	.	GT	0|0	0|0	0|0	1|1	1|1	1|1	0|1
chr1	1000001	.	G	C,T	100	PASS	.	GT	0|0	0|0	0|0	1|1	1|1	1|1	1|0
)VCF");
}

// Node id for a single-node allele path. Every allele in MakeTransitionOverlayFixtureVCF() is a plain
// SNV, so PathNodes() returns exactly one node.
odgi::nid_t AlleleNode(const Graph& graph, size_t path_id) {
  return graph.PathNodes(handlegraph::as_path_handle(path_id)).front();
}

std::vector<std::pair<odgi::nid_t, uint32_t>> TransitionRow(const HaplotypePriorOverlay& overlay,
                                                              const Graph& graph, odgi::nid_t from_node) {
  const size_t row = static_cast<size_t>(from_node - graph.min_node_id());
  std::vector<std::pair<odgi::nid_t, uint32_t>> result;
  for (size_t i = overlay.transition_starts()[row]; i < overlay.transition_starts()[row + 1]; i++) {
    result.emplace_back(overlay.transitions()[i].to_node, overlay.transitions()[i].count);
  }
  return result;
}

}  // namespace

TEST_F(GraphConstructionTest, HaplotypePriorOverlayCountsAdjacentPairTransitions) {
  auto vcf = MakeTransitionOverlayFixtureVCF();
  auto region = Range("chr1", 999989, 1000010);
  Graph graph(HG38FastaPath_, vcf.file_path_, region);

  // Sanity-check the assumed path-id assignment: Sample1 (homozygous REF/REF) and Sample4 (homozygous
  // ALT1/ALT1) haplotype 0's node sequence should be exactly the concatenation of the corresponding REF/ALT
  // allele paths' nodes, in order.
  // Sample1/Sample4 hap0's full node sequence additionally includes flanking reference padding nodes, so
  // check the expected variant nodes are included (in order) rather than an exact match.
  auto ExpectIncludesInOrder = [&](const std::string& sample_path, size_t first, size_t second) {
    auto haplotype_nodes = graph.PathNodes(sample_path);
    auto first_nodes = graph.PathNodes(handlegraph::as_path_handle(first));
    auto second_nodes = graph.PathNodes(handlegraph::as_path_handle(second));
    EXPECT_TRUE(std::includes(haplotype_nodes.begin(), haplotype_nodes.end(), first_nodes.begin(), first_nodes.end()))
        << "path-id assumption violated for " << sample_path;
    EXPECT_TRUE(std::includes(haplotype_nodes.begin(), haplotype_nodes.end(), second_nodes.begin(), second_nodes.end()))
        << "path-id assumption violated for " << sample_path;
  };
  ExpectIncludesInOrder("Sample1#0#chr1#0", kR1, kR2);
  ExpectIncludesInOrder("Sample4#0#chr1#0", kA1, kA2);

  HaplotypePriorOverlay overlay(graph);

  const odgi::nid_t r1_node = AlleleNode(graph, kR1);
  const odgi::nid_t a1_node = AlleleNode(graph, kA1);
  const odgi::nid_t r2_node = AlleleNode(graph, kR2);
  const odgi::nid_t a2_node = AlleleNode(graph, kA2);
  const odgi::nid_t a3_node = AlleleNode(graph, kA3);

  const size_t node_space = static_cast<size_t>(graph.max_node_id() - graph.min_node_id() + 1);
  ASSERT_EQ(overlay.transition_starts().size(), node_space + 1);
  ASSERT_EQ(overlay.node_totals().size(), node_space);

  EXPECT_EQ(overlay.node_totals()[r1_node - graph.min_node_id()], 7u);  // 6 (homozygous-ref) + 1 (crossed)
  EXPECT_EQ(overlay.node_totals()[a1_node - graph.min_node_id()], 7u);  // 6 (homozygous-alt) + 1 (crossed)
  EXPECT_EQ(overlay.node_totals()[a3_node - graph.min_node_id()], 0u);  // Cold start: never observed as a "from"

  // Expected rows sorted by node id (not path id) since the CSR is sorted by to_node -- build/sort the
  // expectation the same way rather than assuming which allele's node id is numerically smaller.
  auto sorted = [](std::vector<std::pair<odgi::nid_t, uint32_t>> v) {
    std::sort(v.begin(), v.end());
    return v;
  };
  EXPECT_EQ(TransitionRow(overlay, graph, r1_node), sorted({{r2_node, 6}, {a2_node, 1}}));
  EXPECT_EQ(TransitionRow(overlay, graph, a1_node), sorted({{r2_node, 1}, {a2_node, 6}}));
  EXPECT_TRUE(TransitionRow(overlay, graph, a3_node).empty());  // Cold start: never observed as a "from"
}

TEST_F(GraphConstructionTest, HaplotypePriorOverlayExcludesRequestedSamples) {
  // Sample7 is the "crossed" sample (r1,a2)/(a1,r2): its haplotypes contribute the lone r1->a2 and a1->r2
  // transitions (see MakeTransitionOverlayFixtureVCF's comment), so excluding it should drop both tiers'
  // counts back to a clean 6/0 split and remove Sample7's own haplotypes from the tier-1 panel entirely.
  auto vcf = MakeTransitionOverlayFixtureVCF();
  auto region = Range("chr1", 999989, 1000010);
  Graph graph(HG38FastaPath_, vcf.file_path_, region);

  const odgi::nid_t r1_node = AlleleNode(graph, kR1);
  const odgi::nid_t a1_node = AlleleNode(graph, kA1);
  const odgi::nid_t r2_node = AlleleNode(graph, kR2);
  const odgi::nid_t a2_node = AlleleNode(graph, kA2);

  HaplotypePriorOverlay full(graph);
  HaplotypePriorOverlay excluded(graph, {"Sample7"});

  // Tier 2: the crossed sample's minority transitions disappear entirely once excluded.
  using TransitionRowVec = std::vector<std::pair<odgi::nid_t, uint32_t>>;
  EXPECT_EQ(TransitionRow(excluded, graph, r1_node), TransitionRowVec({{r2_node, 6}}));
  EXPECT_EQ(TransitionRow(excluded, graph, a1_node), TransitionRowVec({{a2_node, 6}}));
  EXPECT_EQ(excluded.node_totals()[r1_node - graph.min_node_id()], 6u);
  EXPECT_EQ(excluded.node_totals()[a1_node - graph.min_node_id()], 6u);

  // Tier 1: Sample7's own haplotype walk is panel-consistent (nonzero width) with the full panel, but
  // exactly empty (width 0) once Sample7 is excluded.
  auto WalkWidth = [](const HaplotypePriorOverlay& overlay, const Graph::NodeIdSeq& nodes) -> size_t {
    auto state = overlay.Find(nodes.front());
    for (size_t i = 1; i < nodes.size() && !overlay.Empty(state); i++) {
      state = overlay.Extend(state, nodes[i]);
    }
    return overlay.Empty(state) ? 0 : overlay.Width(state);
  };
  auto sample7_hap0 = graph.PathNodes("Sample7#0#chr1#0");
  ASSERT_FALSE(sample7_hap0.empty());
  EXPECT_GE(WalkWidth(full, sample7_hap0), 1u);
  EXPECT_EQ(WalkWidth(excluded, sample7_hap0), 0u);

  // Excluding a name absent from the panel is a no-op.
  HaplotypePriorOverlay excluded_unrelated(graph, {"NotASample"});
  EXPECT_EQ(excluded_unrelated.node_totals(), full.node_totals());

  // Excluding every sample name in the panel leaves both tiers empty.
  HaplotypePriorOverlay excluded_all(
      graph, {"Sample1", "Sample2", "Sample3", "Sample4", "Sample5", "Sample6", "Sample7"});
  EXPECT_EQ(std::accumulate(excluded_all.node_totals().begin(), excluded_all.node_totals().end(), 0u), 0u);
  EXPECT_TRUE(excluded_all.Empty(excluded_all.Find(r1_node)));
}

TEST_F(GraphConstructionTest, HaplotypePriorOverlayProbabilityMatchesHandComputedLaplaceSmoothing) {
  // TransitionProbability/LogTransitionProbability, the building block ApplyPopulationEdge's
  // uses to score every edge *after* a fallback.
  auto vcf = MakeTransitionOverlayFixtureVCF();
  auto region = Range("chr1", 999989, 1000010);
  Graph graph(HG38FastaPath_, vcf.file_path_, region);

  const odgi::nid_t r1_node = AlleleNode(graph, kR1);
  const odgi::nid_t r2_node = AlleleNode(graph, kR2);
  const odgi::nid_t a2_node = AlleleNode(graph, kA2);
  const odgi::nid_t a3_node = AlleleNode(graph, kA3);

  HaplotypePriorOverlay overlay(graph);

  // K_to = 2 (r1's own row has exactly 2 distinct observed successors: r2_node, a2_node).
  // P(r2|r1) = (count+alpha)/(total+alpha*K_to) = (6+1)/(7+1*2) = 7/9.
  // P(a2|r1) = (1+1)/(7+1*2) = 2/9.
  EXPECT_NEAR(overlay.TransitionProbability(r1_node, r2_node, /*alpha=*/1.0), 7.0 / 9.0, 1e-9);
  EXPECT_NEAR(overlay.TransitionProbability(r1_node, a2_node, /*alpha=*/1.0), 2.0 / 9.0, 1e-9);
  EXPECT_NEAR(overlay.LogTransitionProbability(r1_node, r2_node, /*alpha=*/1.0), std::log(7.0 / 9.0), 1e-9);

  // Cold start (a3_node has zero observed successors at all, K_to floors to 1): P = alpha/(0+alpha*1) = 1.0
  // for any `to`, regardless of alpha (no penalty when there is not information available).
  EXPECT_DOUBLE_EQ(overlay.TransitionProbability(a3_node, r2_node, /*alpha=*/1.0), 1.0);
  EXPECT_DOUBLE_EQ(overlay.TransitionProbability(a3_node, r2_node, /*alpha=*/0.3), 1.0);
  EXPECT_DOUBLE_EQ(overlay.LogTransitionProbability(a3_node, r2_node, /*alpha=*/1.0), 0.0);
}

TEST_F(GraphConstructionTest, HaplotypePriorOverlayTransitionTableSerializationRoundtrip) {
  auto vcf = MakeTransitionOverlayFixtureVCF();
  auto region = Range("chr1", 999989, 1000010);
  Graph graph(HG38FastaPath_, vcf.file_path_, region);

  HaplotypePriorOverlay original(graph);

  test::TempDir dir;
  auto bin_path = (dir.path_ / "transitions.bin").string();
  original.Save(bin_path);

  // Load into a heap-allocated HaplotypePriorOverlay via placement new, mirroring UniqueKmersOverlay's
  // Load pattern: it holds a const reference, so it is neither copyable nor movable.
  auto* raw = static_cast<HaplotypePriorOverlay*>(::operator new(sizeof(HaplotypePriorOverlay)));
  HaplotypePriorOverlay::Load(raw, graph, bin_path);
  std::unique_ptr<HaplotypePriorOverlay, void (*)(HaplotypePriorOverlay*)> loaded_owner(
      raw, [](HaplotypePriorOverlay* p) { p->~HaplotypePriorOverlay(); ::operator delete(p); });
  HaplotypePriorOverlay* loaded = loaded_owner.get();

  EXPECT_EQ(loaded->transition_starts(), original.transition_starts());
  EXPECT_EQ(loaded->node_totals(), original.node_totals());
  ASSERT_EQ(loaded->transitions().size(), original.transitions().size());
  for (size_t i = 0; i < original.transitions().size(); i++) {
    EXPECT_EQ(loaded->transitions()[i].to_node, original.transitions()[i].to_node);
    EXPECT_EQ(loaded->transitions()[i].count, original.transitions()[i].count);
  }
}

TEST_F(GraphConstructionTest, HaplotypePriorOverlayChainsHarmlesslyAcrossMultiNodeAlleleSpan) {
  // V1 is an 8bp deletion; V2 is a SNV positioned inside V1's deleted span. V2's breakpoints fragment V1's
  // REF allele span into multiple graph nodes that all carry only V1's REF path-id bit. A sample that is
  // homozygous REF at both variants walks straight through that fragmented span, so its haplotype path visits
  // several consecutive distinguishing nodes that all belong to the same conceptual V1-REF allele choice.
  //
  // Node-keying doesn't collapse these into one state. Each fragment is a distinct graph node, hence a distinct
  // Markov state. No self-eges should be observed.
  test::TestVCFFile vcf(R"VCF(##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##contig=<ID=chr1,length=248956422>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	Sample1	Sample2
chr1	1000000	.	NNNNNNNNN	N	100	PASS	.	GT	0|0	0|0
chr1	1000004	.	N	A	100	PASS	.	GT	0|0	0|0
)VCF");

  auto region = Range("chr1", 999989, 1000010);
  Graph graph(HG38FastaPath_, vcf.file_path_, region);

  // Find the multi-node allele span this fixture is designed to exercise.
  std::vector<odgi::nid_t> span_nodes;
  for (size_t id = 2; id <= 5; id++) {
    auto nodes = graph.PathNodes(handlegraph::as_path_handle(id));
    if (nodes.size() > 1) {
      span_nodes = nodes;
      break;
    }
  }
  ASSERT_FALSE(span_nodes.empty()) << "test fixture did not produce a multi-node allele span";

  HaplotypePriorOverlay overlay(graph);

  // No literal self-transition anywhere.
  for (size_t from = 0; from + 1 < overlay.transition_starts().size(); from++) {
    for (size_t i = overlay.transition_starts()[from]; i < overlay.transition_starts()[from + 1]; i++) {
      EXPECT_NE(static_cast<size_t>(overlay.transitions()[i].to_node - graph.min_node_id()), from)
          << "self-transition recorded for node index " << from;
    }
  }

  // Every consecutive pair of fragment nodes chains with certainty (P == 1.0, independent of alpha since
  // count == node_totals and only one successor is ever observed at each fragment).
  for (size_t i = 0; i + 1 < span_nodes.size(); i++) {
    EXPECT_DOUBLE_EQ(overlay.TransitionProbability(span_nodes[i], span_nodes[i + 1], /*alpha=*/0.1), 1.0)
        << "expected a certain pass-through transition between multi-node span fragments";
  }
}

namespace {

// Graph node ids (not path ids) for a single-node allele's distinguishing span
odgi::nid_t SingleAlleleNode(const Graph& graph, size_t path_id) {
  auto nodes = graph.PathNodes(handlegraph::as_path_handle(path_id));
  return nodes.front();
}

// Walk overlay.Find()/Extend() through @p nodes, a real panel member's own node sequence or any other
// genuine walk starting at nodes.front(), up to and including the first occurrence of @p target_node,
// returning the resulting state.
HaplotypePriorOverlay::PanelState WalkToNode(const HaplotypePriorOverlay& overlay, const Graph::NodeIdSeq& nodes,
                                             odgi::nid_t target_node) {
  auto target = std::find(nodes.begin(), nodes.end(), target_node);
  auto state = overlay.Find(nodes.front());
  for (auto it = nodes.begin() + 1; it <= target; ++it) {
    state = overlay.Extend(state, *it);
  }
  return state;
}

}  // namespace

TEST_F(GraphConstructionTest, HaplotypePriorOverlayFindExtendWidthsMatchTransitionCounts) {
  // Reuse the HaplotypePriorOverlay fixture: same panel, so Find/Extend widths (deduplicated by
  // (sample, haplotype_index), but this fixture has no phase breaks so every haplotype is a single GBWT
  // sequence) should exactly match that overlay's independently-computed allele_totals()/transition counts
  // -- 7 (kR1), 7 (kA1), and the {6,1}/{1,6} split at V2, already hand-verified there. Walked from each
  // representative sample's own path start (Sample1: hom-ref, Sample4: hom-alt).
  auto vcf = MakeTransitionOverlayFixtureVCF();
  auto region = Range("chr1", 999989, 1000010);
  Graph graph(HG38FastaPath_, vcf.file_path_, region);

  odgi::nid_t r1_node = SingleAlleleNode(graph, kR1);
  odgi::nid_t a1_node = SingleAlleleNode(graph, kA1);
  odgi::nid_t r2_node = SingleAlleleNode(graph, kR2);
  odgi::nid_t a2_node = SingleAlleleNode(graph, kA2);

  HaplotypePriorOverlay overlay(graph);

  auto ref_walk = graph.PathNodes("Sample1#0#chr1#0");  // hom-ref: r1, r2
  auto alt_walk = graph.PathNodes("Sample4#0#chr1#0");  // hom-alt: a1, a2

  auto state_r1 = WalkToNode(overlay, ref_walk, r1_node);
  auto state_a1 = WalkToNode(overlay, alt_walk, a1_node);
  ASSERT_FALSE(overlay.Empty(state_r1));
  ASSERT_FALSE(overlay.Empty(state_a1));
  EXPECT_EQ(overlay.Width(state_r1), 7u);  // matches allele_totals()[kR1]
  EXPECT_EQ(overlay.Width(state_a1), 7u);  // matches allele_totals()[kA1]

  EXPECT_EQ(overlay.Width(overlay.Extend(state_r1, r2_node)), 6u);
  EXPECT_EQ(overlay.Width(overlay.Extend(state_r1, a2_node)), 1u);
  EXPECT_EQ(overlay.Width(overlay.Extend(state_a1, r2_node)), 1u);
  EXPECT_EQ(overlay.Width(overlay.Extend(state_a1, a2_node)), 6u);
}

TEST_F(GraphConstructionTest, HaplotypePriorOverlayExtendIntoUnobservedCombinationReturnsEmptyState) {
  auto vcf = MakeTransitionOverlayFixtureVCF();
  auto region = Range("chr1", 999989, 1000010);
  Graph graph(HG38FastaPath_, vcf.file_path_, region);

  odgi::nid_t r1_node = SingleAlleleNode(graph, kR1);
  odgi::nid_t a1_node = SingleAlleleNode(graph, kA1);

  HaplotypePriorOverlay overlay(graph);
  auto ref_walk = graph.PathNodes("Sample1#0#chr1#0");

  // No panel member takes r1 then a1 (a1 is a different allele of the *same* variant V1, never a valid
  // continuation from a state already committed to r1 at V1).
  auto state_r1 = WalkToNode(overlay, ref_walk, r1_node);
  auto bogus = overlay.Extend(state_r1, a1_node);
  EXPECT_TRUE(overlay.Empty(bogus));
  EXPECT_EQ(overlay.Width(bogus), 0u);
  EXPECT_EQ(overlay.ScoreToGo(bogus), 0.0);

  // A node never indexed at all (far outside the panel's node-id range) is likewise empty.
  auto never_indexed = overlay.Extend(state_r1, graph.max_node_id() + 1000);
  EXPECT_TRUE(overlay.Empty(never_indexed));
}

TEST_F(GraphConstructionTest, HaplotypePriorOverlayScoreToGoMatchesHandComputedLogRatios) {
  // At V1, r1 (width 7) splits 6:1 towards r2:a2; a1 (width 7) splits 1:6 towards r2:a2 -- a symmetric,
  // hand-verifiable case (design doc §4a worked example, same shape). Both r1's and a1's best continuation
  // therefore has ratio 6/7, and V2 is the region's last variant (only shared trailing reference follows,
  // a no-op on width), so ScoreToGo at V2 is exactly 0.0 and ScoreToGo at V1 is exactly log(6/7).
  auto vcf = MakeTransitionOverlayFixtureVCF();
  auto region = Range("chr1", 999989, 1000010);
  Graph graph(HG38FastaPath_, vcf.file_path_, region);

  odgi::nid_t r1_node = SingleAlleleNode(graph, kR1);
  odgi::nid_t a1_node = SingleAlleleNode(graph, kA1);
  odgi::nid_t r2_node = SingleAlleleNode(graph, kR2);
  odgi::nid_t a2_node = SingleAlleleNode(graph, kA2);

  HaplotypePriorOverlay overlay(graph);

  auto ref_walk = graph.PathNodes("Sample1#0#chr1#0");
  auto alt_walk = graph.PathNodes("Sample4#0#chr1#0");
  auto state_r1 = WalkToNode(overlay, ref_walk, r1_node);
  auto state_a1 = WalkToNode(overlay, alt_walk, a1_node);

  const double kExpected = std::log(6.0 / 7.0);
  EXPECT_NEAR(overlay.ScoreToGo(state_r1), kExpected, 1e-9);
  EXPECT_NEAR(overlay.ScoreToGo(state_a1), kExpected, 1e-9);

  EXPECT_EQ(overlay.ScoreToGo(overlay.Extend(state_r1, r2_node)), 0.0);
  EXPECT_EQ(overlay.ScoreToGo(overlay.Extend(state_r1, a2_node)), 0.0);
  EXPECT_EQ(overlay.ScoreToGo(overlay.Extend(state_a1, r2_node)), 0.0);
  EXPECT_EQ(overlay.ScoreToGo(overlay.Extend(state_a1, a2_node)), 0.0);
}

TEST_F(GraphConstructionTest, HaplotypeSamplerAccumulatesGbwtOnlyPopulationPriorAlongRealPanelPath) {
  // GBWT scoring wired through HaplotypeSamplerOverlay::Score() via ApplyPopulationEdge. The prior overlay carries a
  // tier-2 fallback table too, but this fixture's haplotype is itself one of the panel's own paths, so Extend() never
  // goes empty and the fallback is never reached. The accumulated population term must telescope to exactly weight *
  // log(Width(final state)/Width(initial state)), independent of exactly how many intermediate edges/nodes the walk
  // crosses, since every no-op (unbranched) edge contributes log(1) = 0.
  auto vcf = MakeTransitionOverlayFixtureVCF();
  auto region = Range("chr1", 999989, 1000010);
  Graph graph(HG38FastaPath_, vcf.file_path_, region);

  HaplotypePriorOverlay gbwt_overlay(graph);

  HaplotypeSamplerOverlay::Params params;
  params.haplotype_prior_weight = 2.5;  // arbitrary nonzero weight, applied multiplicatively throughout

  // Zero k-mers: isolates the population-prior term entirely from k-mer scoring.
  HaplotypeSamplerOverlay sampler(graph, std::vector<std::string>{},
                                  std::vector<std::vector<UniqueKmersOverlay::KmerLocation>>{}, &gbwt_overlay, params);

  auto ref_walk = graph.PathNodes("Sample1#0#chr1#0");  // hom-ref: r1, r2 -- a real panel path
  ASSERT_GE(ref_walk.size(), 2u);

  // Independently compute the expected telescoped log-width-ratio by walking the same overlay directly
  // (not via ApplyPopulationEdge/Score() -- a genuinely separate computation of the same overlay's API).
  auto state = gbwt_overlay.Find(ref_walk.front());
  ASSERT_FALSE(gbwt_overlay.Empty(state));
  const size_t initial_width = gbwt_overlay.Width(state);
  for (size_t i = 1; i < ref_walk.size(); ++i) {
    state = gbwt_overlay.Extend(state, ref_walk[i]);
    ASSERT_FALSE(gbwt_overlay.Empty(state)) << "ref_walk is a real panel path; Extend() should never go empty";
  }
  const size_t final_width = gbwt_overlay.Width(state);

  // Cross-check against the independently-hand-verified widths from the sibling GBWT tests: 14 haplotypes
  // total (7 samples x ploidy 2, no phase breaks in this fixture) narrows to 7 at r1, then to 6 at r2.
  EXPECT_EQ(initial_width, 14u);
  EXPECT_EQ(final_width, 6u);

  const double expected =
      params.haplotype_prior_weight * std::log(static_cast<double>(final_width) / static_cast<double>(initial_width));
  EXPECT_NEAR(sampler.Score(ref_walk), expected, 1e-9);
}

TEST_F(GraphConstructionTest, HaplotypeSamplerFallsBackToTier2WhenPanelDiverges) {
  // Force a Gbwt->FellBack transition by substituting V2's never-observed 3rd allele (a3, zero genotype count in this
  // fixture) into an otherwise real panel path (Sample1's own hom-ref walk), right after the real, panel-consistent r1
  // choice at V1. This is a graph-valid haplotype (a3 is a real branch of V2's own bubble, reached by a real edge) but
  // a panel-unobserved continuation from r1, exactly the case ApplyPopulationEdge's fallback branch exists for. The
  // triggering edge pays exactly panel_fallback_penalty (no tier-2 term on that same edge, even though it does
  // distinguish), and nothing further is added afterward since this fixture's trailing reference carries no further
  // distinguishing nodes.
  auto vcf = MakeTransitionOverlayFixtureVCF();
  auto region = Range("chr1", 999989, 1000010);
  Graph graph(HG38FastaPath_, vcf.file_path_, region);

  odgi::nid_t r1_node = SingleAlleleNode(graph, kR1);
  odgi::nid_t r2_node = SingleAlleleNode(graph, kR2);
  odgi::nid_t a3_node = SingleAlleleNode(graph, kA3);

  HaplotypePriorOverlay gbwt_overlay(graph);

  auto ref_walk = graph.PathNodes("Sample1#0#chr1#0");
  auto r1_it = std::find(ref_walk.begin(), ref_walk.end(), r1_node);
  auto r2_it = std::find(ref_walk.begin(), ref_walk.end(), r2_node);
  ASSERT_NE(r1_it, ref_walk.end());
  ASSERT_NE(r2_it, ref_walk.end());

  // Confirm the fixture premise directly: r1 is panel-consistent, but no panel member ever continues from
  // r1 into a3.
  auto state_r1 = WalkToNode(gbwt_overlay, ref_walk, r1_node);
  ASSERT_FALSE(gbwt_overlay.Empty(state_r1));
  ASSERT_TRUE(gbwt_overlay.Empty(gbwt_overlay.Extend(state_r1, a3_node)))
      << "test premise requires a3 to be an unobserved continuation from r1";

  // Construct a graph-valid but panel-unobserved haplotype: Sample1's own prefix through r1 (and any
  // reference in between), then a3 instead of r2, then Sample1's own trailing reference (the same
  // reconvergence node every V2 allele -- including a3 -- shares).
  Graph::NodeIdSeq haplotype(ref_walk.begin(), r2_it);
  haplotype.push_back(a3_node);
  haplotype.insert(haplotype.end(), r2_it + 1, ref_walk.end());

  HaplotypeSamplerOverlay::Params params;
  params.haplotype_prior_weight = 2.5;
  params.panel_fallback_penalty = -1.25;  // arbitrary, distinguishable from 0 and from any log-probability
  params.transition_prior_alpha = 1.0;

  // Zero k-mers: isolates the population-prior term entirely from k-mer scoring.
  HaplotypeSamplerOverlay sampler(graph, std::vector<std::string>{},
                                  std::vector<std::vector<UniqueKmersOverlay::KmerLocation>>{}, &gbwt_overlay, params);

  // Independently compute the expected score: telescoped log-width-ratio up through whatever precedes a3
  // (still tier 1, so this includes r1's own real branch), then exactly panel_fallback_penalty for the
  // triggering edge into a3 -- nothing after that, since no further distinguishing node follows in this
  // fixture's trailing reference.
  auto expected_state = gbwt_overlay.Find(haplotype.front());
  ASSERT_FALSE(gbwt_overlay.Empty(expected_state));
  const size_t initial_width = gbwt_overlay.Width(expected_state);
  size_t width_before_fallback = initial_width;
  for (size_t i = 1; i < haplotype.size(); ++i) {
    if (haplotype[i] == a3_node) break;  // fallback triggers exactly here
    expected_state = gbwt_overlay.Extend(expected_state, haplotype[i]);
    ASSERT_FALSE(gbwt_overlay.Empty(expected_state));
    width_before_fallback = gbwt_overlay.Width(expected_state);
  }
  ASSERT_NE(width_before_fallback, initial_width) << "test premise requires a real branch (r1) before a3";

  const double expected = params.haplotype_prior_weight *
      (std::log(static_cast<double>(width_before_fallback) / static_cast<double>(initial_width)) +
       params.panel_fallback_penalty);
  EXPECT_NEAR(sampler.Score(haplotype), expected, 1e-9);
}

TEST_F(GraphConstructionTest, HaplotypeSamplerFindBestPathsMatchesBruteForceWithPopulationPriorActive) {
  // Confirm PropagateBestPathState's trim/rank-key changes (population_state_pool_cap widening the intermediate cap,
  // SamePoolClass's population-aware dedup, PopulationScoreToGo-informed ranking) don't lose any of this small
  // fixture's 6 true V1 x V2 combinations at any output width. FindBestPaths(n) must match the brute-force top-n
  // ranking by Score() (which itself already includes the population term, both tiers) exactly, the same general idiom
  // as PendingPoolWidthLimitTest.
  auto vcf = MakeTransitionOverlayFixtureVCF();
  auto region = Range("chr1", 999989, 1000010);
  Graph graph(HG38FastaPath_, vcf.file_path_, region);

  odgi::nid_t r1_node = SingleAlleleNode(graph, kR1);
  odgi::nid_t a1_node = SingleAlleleNode(graph, kA1);
  odgi::nid_t r2_node = SingleAlleleNode(graph, kR2);
  odgi::nid_t a2_node = SingleAlleleNode(graph, kA2);
  odgi::nid_t a3_node = SingleAlleleNode(graph, kA3);

  HaplotypePriorOverlay prior(graph);

  HaplotypeSamplerOverlay::Params params;
  params.haplotype_prior_weight = 1.0;
  params.panel_fallback_penalty = -5.0;  // clearly worse than any real log-ratio here, but still finite/valid
  params.transition_prior_alpha = 1.0;

  // Zero k-mers: isolates the population-prior term, so brute-force ranking is entirely GBWT/tier-2-driven.
  HaplotypeSamplerOverlay sampler(graph, std::vector<std::string>{},
                                  std::vector<std::vector<UniqueKmersOverlay::KmerLocation>>{}, &prior, params);

  // Enumerate all 6 true haplotypes (2 V1 alleles x 3 V2 alleles) by node-sequence construction, mirroring
  // OverlappingSNPClusterFixture::BruteForceTopScores.
  auto ref_walk = graph.PathNodes("Sample1#0#chr1#0");  // r1, r2 -- used as the shared flanking-reference template
  auto r1_it = std::find(ref_walk.begin(), ref_walk.end(), r1_node);
  auto r2_it = std::find(ref_walk.begin(), ref_walk.end(), r2_node);
  ASSERT_NE(r1_it, ref_walk.end());
  ASSERT_NE(r2_it, ref_walk.end());

  auto BuildCombo = [&](odgi::nid_t v1_node, odgi::nid_t v2_node) {
    Graph::NodeIdSeq path(ref_walk.begin(), r1_it);
    path.push_back(v1_node);
    path.insert(path.end(), r1_it + 1, r2_it);
    path.push_back(v2_node);
    path.insert(path.end(), r2_it + 1, ref_walk.end());
    return path;
  };

  std::vector<Graph::NodeIdSeq> combos = {
      BuildCombo(r1_node, r2_node), BuildCombo(r1_node, a2_node), BuildCombo(r1_node, a3_node),
      BuildCombo(a1_node, r2_node), BuildCombo(a1_node, a2_node), BuildCombo(a1_node, a3_node),
  };

  std::vector<double> brute_scores;
  for (const auto& combo : combos) brute_scores.push_back(sampler.Score(combo));
  std::sort(brute_scores.begin(), brute_scores.end(), std::greater<double>());

  constexpr double kScoreTolerance = 1e-6;
  for (size_t n : {1u, 2u, 3u, 4u, 6u}) {
    auto found = sampler.FindBestPaths(n);
    const size_t expected_count = std::min(n, combos.size());
    ASSERT_EQ(found.size(), expected_count) << "n=" << n;
    for (size_t i = 0; i < expected_count; ++i) {
      EXPECT_NEAR(sampler.Score(found[i]), brute_scores[i], kScoreTolerance)
          << "n=" << n << ": FindBestPaths(n) rank " << i << " diverged from brute force";
    }
  }
}

namespace {
// A few k-mer locations over MakeTransitionOverlayFixtureVCF()'s allele nodes, for tests that need real,
// varying k-mer scores rather than isolating the population term with zero k-mers (checkpoints 4-6 above).
std::pair<std::vector<std::string>, std::vector<std::vector<UniqueKmersOverlay::KmerLocation>>>
MakeTransitionOverlayFixtureKmers(const Graph& graph) {
  std::vector<odgi::nid_t> nodes = {AlleleNode(graph, kR1), AlleleNode(graph, kA1), AlleleNode(graph, kR2),
                                     AlleleNode(graph, kA2)};
  std::vector<std::string> sequences = {"r1", "a1", "r2", "a2"};
  std::vector<std::vector<UniqueKmersOverlay::KmerLocation>> locations;
  for (auto node : nodes) {
    locations.push_back({UniqueKmersOverlay::KmerLocation({{graph.get_handle(node)}, 0})});
  }
  return {sequences, locations};
}
}  // namespace

TEST_F(GraphConstructionTest, HaplotypeSamplerByteIdenticalWithPriorDisabled) {
  // A sampler with the population-prior overlay configured but haplotype_prior_weight left at 0.0 must produce
  // byte-identical output to a sampler with no overlay configured at all. Modeled on
  // SampleHaplotypesSanityTest::DeterministicAcrossRandomAdversarialScores's two-sampler-comparison.
  auto vcf = MakeTransitionOverlayFixtureVCF();
  auto region = Range("chr1", 999989, 1000010);
  Graph graph(HG38FastaPath_, vcf.file_path_, region);

  HaplotypePriorOverlay prior(graph);
  auto [sequences, locations] = MakeTransitionOverlayFixtureKmers(graph);

  HaplotypeSamplerOverlay::Params disabled_params{};  // fully default: no prior fields touched at all

  HaplotypeSamplerOverlay::Params configured_but_inert_params{};
  configured_but_inert_params.panel_fallback_penalty = -3.0;  // nonzero, to prove it's genuinely unused too
  configured_but_inert_params.haplotype_prior_weight = 0.0;   // the actual no-op guard

  static const KmerZygosity kZygosities[] = {KmerZygosity::ABSENT, KmerZygosity::HETEROZYGOUS, KmerZygosity::HOMOZYGOUS};
  std::mt19937 rng(13579);
  std::uniform_int_distribution<int> zyg_dist(0, 2);

  constexpr int kNumTrials = 50;
  for (int trial = 0; trial < kNumTrials; ++trial) {
    std::vector<KmerZygosity> zygosities(sequences.size());
    for (auto& z : zygosities) z = kZygosities[zyg_dist(rng)];
    IndexedKmerClassify counts(zygosities);

    HaplotypeSamplerOverlay sampler_a(graph, sequences, locations, nullptr, disabled_params);
    sampler_a.InitializeScores(counts);
    auto haplotypes_a = sampler_a.SampleHaplotypes(4);

    HaplotypeSamplerOverlay sampler_b(graph, sequences, locations, &prior, configured_but_inert_params);
    sampler_b.InitializeScores(counts);
    auto haplotypes_b = sampler_b.SampleHaplotypes(4);

    ASSERT_EQ(haplotypes_a.size(), haplotypes_b.size()) << "trial " << trial;
    for (size_t i = 0; i < haplotypes_a.size(); ++i) {
      EXPECT_EQ(haplotypes_a[i], haplotypes_b[i]) << "trial " << trial << " draw " << i;
      EXPECT_EQ(sampler_a.Score(haplotypes_a[i]), sampler_b.Score(haplotypes_b[i])) << "trial " << trial << " draw " << i;
    }
  }
}

TEST_F(GraphConstructionTest, HaplotypeSamplerPopulationPriorNoOpWhenOverlaysNull) {
  // Companion to the above: with Params{} entirely default (no overlay pointers set at all, regardless of
  // haplotype_prior_weight), PopulationScoreToGo/ApplyPopulationEdge must be exact 0.0 no-ops.
  auto vcf = MakeTransitionOverlayFixtureVCF();
  auto region = Range("chr1", 999989, 1000010);
  Graph graph(HG38FastaPath_, vcf.file_path_, region);
  auto [sequences, locations] = MakeTransitionOverlayFixtureKmers(graph);

  HaplotypeSamplerOverlay::Params params{};
  params.haplotype_prior_weight = 3.0;  // nonzero on its own does nothing without an overlay configured

  static const KmerZygosity kZygosities[] = {KmerZygosity::ABSENT, KmerZygosity::HETEROZYGOUS, KmerZygosity::HOMOZYGOUS};
  std::mt19937 rng(97531);
  std::uniform_int_distribution<int> zyg_dist(0, 2);

  constexpr int kNumTrials = 30;
  for (int trial = 0; trial < kNumTrials; ++trial) {
    std::vector<KmerZygosity> zygosities(sequences.size());
    for (auto& z : zygosities) z = kZygosities[zyg_dist(rng)];
    IndexedKmerClassify counts(zygosities);

    HaplotypeSamplerOverlay sampler(graph, sequences, locations, nullptr, params);
    sampler.InitializeScores(counts);
    auto haplotypes = sampler.SampleHaplotypes(4);

    HaplotypeSamplerOverlay baseline(graph, sequences, locations, nullptr, HaplotypeSamplerOverlay::Params{});
    baseline.InitializeScores(counts);
    auto baseline_haplotypes = baseline.SampleHaplotypes(4);

    ASSERT_EQ(haplotypes.size(), baseline_haplotypes.size()) << "trial " << trial;
    for (size_t i = 0; i < haplotypes.size(); ++i) {
      EXPECT_EQ(haplotypes[i], baseline_haplotypes[i]) << "trial " << trial << " draw " << i;
    }
  }
}

TEST_F(GraphConstructionTest, HaplotypeSamplerFindCalledExactlyOnceAcrossDPRun) {
  // Find() must be called exactly once, at the seed, never mid-DP. Not directly interceptable (HaplotypePriorOverlay
  // isn't mockable/virtual), so verify indirectly: run FindBestPaths with the prior enabled, then independently re-walk
  // the winning haplotype's own node sequence through a *second*, fresh Find()-once/Extend()-thereafter reference walk
  // (same as HaplotypePriorOverlayConstructsOnSmallBenchmarkRegion), and confirm Score()'s recovered population
  // contribution matches that reference walk's telescoped log-width-ratio sum exactly. A stray mid-DP Find() would
  // silently zero Width()/ScoreToGo() partway through and produce a detectably different, generally worse-explained
  // accumulated value than this independent reference walk.
  auto vcf = MakeTransitionOverlayFixtureVCF();
  auto region = Range("chr1", 999989, 1000010);
  Graph graph(HG38FastaPath_, vcf.file_path_, region);

  HaplotypePriorOverlay gbwt_overlay(graph);
  auto [sequences, locations] = MakeTransitionOverlayFixtureKmers(graph);

  HaplotypeSamplerOverlay::Params params{};
  params.haplotype_prior_weight = 1.5;

  HaplotypeSamplerOverlay sampler(graph, sequences, locations, &gbwt_overlay, params);
  ConstantKmerClassify counts;
  sampler.InitializeScores(counts);

  auto best_paths = sampler.FindBestPaths(1);
  ASSERT_FALSE(best_paths.empty());
  const auto& winner = best_paths.front();

  // Independent reference walk: Find() once at the winner's own front node, Extend() thereafter.
  auto ref_state = gbwt_overlay.Find(winner.front());
  ASSERT_FALSE(gbwt_overlay.Empty(ref_state));
  double reference_population_score = 0.0;
  for (size_t i = 1; i < winner.size(); ++i) {
    auto extended = gbwt_overlay.Extend(ref_state, winner[i]);
    if (gbwt_overlay.Empty(extended)) break;  // this fixture's k-mers never force a fallback; defensive only
    reference_population_score +=
        std::log(static_cast<double>(gbwt_overlay.Width(extended)) / static_cast<double>(gbwt_overlay.Width(ref_state)));
    ref_state = extended;
  }
  reference_population_score *= params.haplotype_prior_weight;

  // Recover the DP's own accumulated population contribution by comparing Score() with and without the
  // prior configured (isolates the population term from the k-mer term, which is identical either way).
  HaplotypeSamplerOverlay::Params no_prior_params{};
  HaplotypeSamplerOverlay no_prior_sampler(graph, sequences, locations, nullptr, no_prior_params);
  no_prior_sampler.InitializeScores(counts);

  const double dp_population_contribution = sampler.Score(winner) - no_prior_sampler.Score(winner);
  EXPECT_NEAR(dp_population_contribution, reference_population_score, 1e-9);
}

TEST_F(GraphConstructionTest, HaplotypeSamplerPopulationPriorTreatsCoLocatedAllelesAsOneState) {
  // A single graph node carrying two distinguishing path bits (co-located/overlapping variants) is one Markov state,
  // not two sequential sub-transitions. Confirm ApplyPopulationEdge scores exactly one transition into that node (not a
  // spurious extra one). Reuses the same fixture as HaplotypePriorOverlayChainsHarmlesslyAcrossMultiNodeAlleleSpan (V1:
  // 8bp deletion; V2: SNV inside V1's deleted span): V1's REF allele's own alt_ref_handles span its *whole* deleted
  // region, which -- after V2's breakpoint fragments it -- includes the single-node fragment exactly at V2's own
  // position, so that one fragment node gets *both* V1-REF's and V2-REF's path bit set (graph.cpp's per-variant
  // node_variant_paths_ construction sets REF's bit on every node in alt_ref_handles for *each* variant independently,
  // so overlapping alt_ref_handles ranges naturally accumulate multiple bits on shared nodes) -- a real haplotype only
  // crosses it by taking REF at both variants.
  test::TestVCFFile vcf(R"VCF(##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##contig=<ID=chr1,length=248956422>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	Sample1	Sample2
chr1	1000000	.	NNNNNNNNN	N	100	PASS	.	GT	0|0	0|0
chr1	1000004	.	N	A	100	PASS	.	GT	0|0	0|0
)VCF");
  auto region = Range("chr1", 999989, 1000010);
  Graph graph(HG38FastaPath_, vcf.file_path_, region);

  // Confirm the fixture premise via public path lookups alone: V1's REF path (path id 2) and V2's REF path
  // (path id 4) share a node -- i.e., a node distinguishing both variants' REF alleles simultaneously.
  auto ref_walk = graph.PathNodes("Sample1#0#chr1#0");  // Sample1 is REF/REF at both variants
  auto v1_ref_nodes = graph.PathNodes(handlegraph::as_path_handle(size_t{2}));
  auto v2_ref_nodes = graph.PathNodes(handlegraph::as_path_handle(size_t{4}));
  std::vector<odgi::nid_t> co_located_nodes;
  std::set_intersection(v1_ref_nodes.begin(), v1_ref_nodes.end(), v2_ref_nodes.begin(), v2_ref_nodes.end(),
                        std::back_inserter(co_located_nodes));
  ASSERT_FALSE(co_located_nodes.empty()) << "test fixture did not produce a co-located node";

  HaplotypePriorOverlay gbwt_overlay(graph);

  HaplotypeSamplerOverlay::Params params{};
  params.haplotype_prior_weight = 1.0;
  params.transition_prior_alpha = 1.0;

  HaplotypeSamplerOverlay sampler(graph, std::vector<std::string>{},
                                  std::vector<std::vector<UniqueKmersOverlay::KmerLocation>>{}, &gbwt_overlay, params);

  // Independently compute the expected telescoped score by walking the overlay directly -- one Extend()
  // call per node (regardless of how many path bits it carries), not one per bit.
  auto state = gbwt_overlay.Find(ref_walk.front());
  ASSERT_FALSE(gbwt_overlay.Empty(state));
  const size_t initial_width = gbwt_overlay.Width(state);
  for (size_t i = 1; i < ref_walk.size(); ++i) {
    state = gbwt_overlay.Extend(state, ref_walk[i]);
    ASSERT_FALSE(gbwt_overlay.Empty(state)) << "Sample1's own path; Extend() should never go empty";
  }
  const double expected = params.haplotype_prior_weight *
      std::log(static_cast<double>(gbwt_overlay.Width(state)) / static_cast<double>(initial_width));
  EXPECT_NEAR(sampler.Score(ref_walk), expected, 1e-9);
}

TEST_F(GraphConstructionTest, HaplotypeSamplerPopulationPriorRespectsUnfilteredDistinguishingMask) {
  // population_distinguishing_mask_ must stay unfiltered by an active inference-VCF filter, unlike
  // contributes_paths_mask_. Build with an inference VCF that excludes V1 (so V1's path bits never contribute to
  // covered_paths), and confirm the population-prior term still realizes a real transition at V1's node (via the DP,
  // exercised through Score()) while covered_paths output stays correctly restricted to V2 alone.
  auto vcf = MakeTransitionOverlayFixtureVCF();
  auto region = Range("chr1", 999989, 1000010);
  Graph graph(HG38FastaPath_, vcf.file_path_, region);

  // Inference VCF containing only V2 (same position/alleles as the main fixture's V2), so V1 is entirely
  // excluded from inference_node_mask_/inference_path_mask_ (see Graph::PopulateNodeAndPathMasks).
  test::TestVCFFile inference_vcf(R"VCF(##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##contig=<ID=chr1,length=248956422>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	Sample1
chr1	1000001	.	G	C,T	100	PASS	.	GT	0|0
)VCF");

  UniqueKmersOverlay unique_kmers(graph, /*k=*/7, /*max_edges=*/5);

  HaplotypePriorOverlay gbwt_overlay(graph);
  HaplotypeSamplerOverlay::Params params{};
  params.haplotype_prior_weight = 1.0;

  HaplotypeSamplerOverlay sampler(graph, unique_kmers, inference_vcf.file_path_, region, /*min_size=*/0, &gbwt_overlay, params);
  ConstantKmerClassify counts;
  sampler.InitializeScores(counts);

  auto ref_walk = graph.PathNodes("Sample1#0#chr1#0");

  // The population term must still reflect the real GBWT transition at V1 (r1), even though V1 is excluded
  // from covered_paths/inference filtering -- computed independently, mirroring earlier checkpoints.
  auto state = gbwt_overlay.Find(ref_walk.front());
  ASSERT_FALSE(gbwt_overlay.Empty(state));
  const size_t initial_width = gbwt_overlay.Width(state);
  for (size_t i = 1; i < ref_walk.size(); ++i) {
    state = gbwt_overlay.Extend(state, ref_walk[i]);
    ASSERT_FALSE(gbwt_overlay.Empty(state));
  }
  const double expected_population_term = std::log(static_cast<double>(gbwt_overlay.Width(state)) /
                                                     static_cast<double>(initial_width));

  // Isolate the population term the same way HaplotypeSamplerFindCalledExactlyOnceAcrossDPRun does: diff
  // against an otherwise-identical sampler with no prior configured (k-mer term is identical either way,
  // and both samplers share the same apply_path_filter_/inference masking).
  HaplotypeSamplerOverlay::Params no_prior_params{};
  HaplotypeSamplerOverlay no_prior_sampler(graph, unique_kmers, inference_vcf.file_path_, region, /*min_size=*/0,
                                           nullptr, no_prior_params);
  no_prior_sampler.InitializeScores(counts);

  const double dp_population_contribution = sampler.Score(ref_walk) - no_prior_sampler.Score(ref_walk);
  EXPECT_NEAR(dp_population_contribution, expected_population_term, 1e-9);

  // covered_paths/output semantics stay correctly V2-only: DecodeHaplotype should report exactly one
  // covered allele (V2's REF, allele index 0), never V1 at all.
  auto decoded = sampler.DecodeHaplotype(ref_walk);
  ASSERT_EQ(decoded.size(), 1u);
  EXPECT_EQ(decoded.front().second, 0u);  // REF allele of V2
}

TEST_F(GraphConstructionTest, HaplotypePriorOverlayWidthDedupsPhaseBrokenHaplotypeSegments) {
  // 3 unbroken samples (Sample1/2 hom-ref, Sample3 hom-alt: 6 true haplotypes, one GBWT sequence each) plus
  // Sample4, whose GT:PS phase-set changes between V1 and V2 -- the exact (GT:PS 0|1:1000000 / GT:PS
  // 0|1:1000001) pair already validated elsewhere (VariantTransitionsGraphConstructionTest) to force both
  // of Sample4's haplotypes into 2 segments each (Sample#H#chr1#0 and Sample#H#chr1#1), so this panel has 8
  // true haplotypes (4 samples x ploidy 2) materialized as 10 GBWT-inserted sequences (6 unbroken + 4
  // segments for Sample4's 2 broken haplotypes). If Width() ever double-counted a phase-broken haplotype's
  // segments as distinct panel members (the exact class of bug the design doc's real-panel GBWT
  // measurement found: raw vg gbwt interval width 301 vs. 212 true haplotypes), walking through any of
  // these 10 real segments end to end would eventually reveal a state whose Width() exceeds 8 -- Width()
  // computes membership directly from the panel-path walk (see haplotype.cpp), not raw GBWT interval size,
  // specifically to avoid this.
  test::TestVCFFile vcf(R"VCF(##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##contig=<ID=chr1,length=248956422>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=PS,Number=1,Type=Integer,Description="Phase set identifier">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	Sample1	Sample2	Sample3	Sample4
chr1	1000000	.	G	A	100	PASS	.	GT:PS	0/0	0/0	1/1	0|1:1000000
chr1	1000001	.	G	C	100	PASS	.	GT:PS	0/0	0/0	1/1	0|1:1000001
)VCF");

  auto region = Range("chr1", 999989, 1000010);
  Graph graph(HG38FastaPath_, vcf.file_path_, region);

  // Confirm the fixture actually exercises a phase break (otherwise this test isn't testing the scenario
  // it claims to) -- mirrors VariantTransitionsGraphConstructionTest's already-validated expectation for
  // this exact GT:PS pair.
  const std::vector<std::string> kPanelPaths = {
      "Sample1#0#chr1#0", "Sample1#1#chr1#0", "Sample2#0#chr1#0", "Sample2#1#chr1#0",
      "Sample3#0#chr1#0", "Sample3#1#chr1#0", "Sample4#0#chr1#0", "Sample4#0#chr1#1",
      "Sample4#1#chr1#0", "Sample4#1#chr1#1"};
  for (const auto& path_name : kPanelPaths) {
    ASSERT_TRUE(graph.has_path(path_name)) << path_name;
  }

  HaplotypePriorOverlay overlay(graph);
  constexpr size_t kTrueHaplotypeCount = 8;  // 4 samples x ploidy 2

  // Walk every real panel segment's own node sequence end to end via Find()/Extend(), checking Width()
  // never exceeds the true haplotype count at any point along the way.
  for (const auto& path_name : kPanelPaths) {
    auto nodes = graph.PathNodes(path_name);
    ASSERT_FALSE(nodes.empty()) << path_name;
    auto state = overlay.Find(nodes.front());
    ASSERT_FALSE(overlay.Empty(state)) << path_name << " at first node";
    EXPECT_LE(overlay.Width(state), kTrueHaplotypeCount) << path_name << " at first node";
    for (size_t i = 1; i < nodes.size(); i++) {
      state = overlay.Extend(state, nodes[i]);
      ASSERT_FALSE(overlay.Empty(state)) << path_name << " at position " << i;
      EXPECT_LE(overlay.Width(state), kTrueHaplotypeCount)
          << path_name << " at position " << i << ": Width() exceeds the true haplotype count -- likely "
          << "double-counting a phase-broken haplotype's segments as distinct panel members";
    }
  }
}

TEST_F(GraphConstructionTest, HaplotypePriorOverlayGbwtTableSerializationRoundtrip) {
  auto vcf = MakeTransitionOverlayFixtureVCF();
  auto region = Range("chr1", 999989, 1000010);
  Graph graph(HG38FastaPath_, vcf.file_path_, region);

  odgi::nid_t r1_node = SingleAlleleNode(graph, kR1);
  odgi::nid_t r2_node = SingleAlleleNode(graph, kR2);

  HaplotypePriorOverlay original(graph);

  test::TempDir dir;
  auto bin_path = (dir.path_ / "gbwt_overlay.bin").string();
  original.Save(bin_path);

  // Load into a heap-allocated HaplotypePriorOverlay via placement new, mirroring
  // HaplotypePriorOverlay's Load pattern: it holds a const reference, so it is neither copyable nor
  // movable.
  auto* raw = static_cast<HaplotypePriorOverlay*>(::operator new(sizeof(HaplotypePriorOverlay)));
  HaplotypePriorOverlay::Load(raw, graph, bin_path);
  std::unique_ptr<HaplotypePriorOverlay, void (*)(HaplotypePriorOverlay*)> loaded_owner(
      raw, [](HaplotypePriorOverlay* p) { p->~HaplotypePriorOverlay(); ::operator delete(p); });
  HaplotypePriorOverlay* loaded = loaded_owner.get();

  auto ref_walk = graph.PathNodes("Sample1#0#chr1#0");
  auto orig_r1 = WalkToNode(original, ref_walk, r1_node);
  auto load_r1 = WalkToNode(*loaded, ref_walk, r1_node);
  EXPECT_FALSE(loaded->Empty(load_r1));
  EXPECT_EQ(loaded->Width(load_r1), original.Width(orig_r1));
  EXPECT_EQ(loaded->ScoreToGo(load_r1), original.ScoreToGo(orig_r1));

  auto orig_r1_r2 = original.Extend(orig_r1, r2_node);
  auto load_r1_r2 = loaded->Extend(load_r1, r2_node);
  EXPECT_FALSE(loaded->Empty(load_r1_r2));
  EXPECT_EQ(loaded->Width(load_r1_r2), original.Width(orig_r1_r2));
  EXPECT_EQ(loaded->ScoreToGo(load_r1_r2), original.ScoreToGo(orig_r1_r2));
}
