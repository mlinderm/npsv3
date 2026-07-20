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

  HaplotypeSamplerOverlay sampler(graph_, sequences, locations, params);
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

  HaplotypeSamplerOverlay sampler(graph_, sequences, locations, params);
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

  HaplotypeSamplerOverlay sampler(graph_, sequences, locations, params);
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

// Verification tests for adaptive beam search
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

  HaplotypeSamplerOverlay::Params params; // Defaults: heterozygous=0.0, homozygous=1.0, absent=-0.8
  HaplotypeSamplerOverlay sampler(graph_, sequences, locations, params);
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

  // FindBestPaths(1) forces n=1, the narrowest possible beam, at every single node throughout the
  // DP. Per the Markovian-future argument above, PropagateCertifiedBestPathState should certify this on
  // its very first attempt (no widening needed) and return exactly the brute-forced optimum.
  auto paths = sampler.FindBestPaths(1);
  ASSERT_EQ(paths.size(), 1u);
  EXPECT_EQ(paths[0], best_path);
  EXPECT_DOUBLE_EQ(sampler.Score(paths[0]), best_score);
}
