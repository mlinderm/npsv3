#include <cstdlib>
#include <filesystem>
#include <fmt/std.h>

#include "graph.hpp"
#include "haplotype.hpp"
#include "kmer.hpp"

#include "test_helpers.hpp"

using namespace npsv3;
using npsv3::test::GraphConstructionTest;
namespace fs = std::filesystem;

/// Classifies every k-mer with a fixed zygosity (no KMC database required).
class ConstantKmerCounts : public KmerCounts {
 public:
  explicit ConstantKmerCounts(KmerZygosity z = KmerZygosity::HOMOZYGOUS) : z_(z) {}

  void ClassifySorted( const std::vector<std::string>& sequences, const ClassificationCallback& callback) const override {
    for (size_t i = 0; i < sequences.size(); ++i) {
      callback(i, z_);
    }
  }

 private:
  KmerZygosity z_;
};

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
  
  ConstantKmerCounts counts;
  sampler.InitializeScores(counts);

  EXPECT_GT(sampler.NumKmers(), 0u);
  EXPECT_TRUE(sampler.HasNodeKmers()) << "Graph has 18bp deletion, so should have some node k-mers";
  EXPECT_TRUE(sampler.HasEdgeKmers()) << "Graph should have edge k-mers crossing the breakpoints";

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

  {  // Still samples paths, even when all k-mers are ABSENT
    HaplotypeSamplerOverlay sampler(graph, UniqueKmersOverlay(graph, k, max_edges));
  
    ConstantKmerCounts counts(KmerZygosity::ABSENT);
    sampler.InitializeScores(counts);
    
    auto haplotypes = sampler.SampleHaplotypes(2);
    EXPECT_EQ(haplotypes.size(), 2u) << "Sampler should return exactly n paths";
  }
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

  const size_t k = 7, max_edges = 5;
  
  UniqueKmersOverlay unique_kmers(graph, k, max_edges);
  HaplotypeSamplerOverlay unfiltered(graph, unique_kmers);
  HaplotypeSamplerOverlay filtered(graph, unique_kmers, vcf.file_path_, region, 5 /* min_size */);
  
  ConstantKmerCounts counts;
  unfiltered.InitializeScores(counts);
  filtered.InitializeScores(counts);

  auto unfiltered_paths = unfiltered.FindBestPaths(10 /* finite limit larger than the expected number of paths */);
  auto filtered_paths = filtered.FindBestPaths(10);

  // Single bi-allelic variant included in inference nodes, so nothing should be filtered out.
  ASSERT_EQ(unfiltered_paths.size(), 4u);
  ASSERT_EQ(filtered_paths.size(), 2u);

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
    EXPECT_FALSE(std::includes(filtered_paths[i].begin(), filtered_paths[i].end(), small_variant_nodes.begin(), small_variant_nodes.end()));
  }
}

TEST_F(GraphConstructionTest, SampleDiplotypesSelectsHomozygousRefWithAllHomozygousKmers) {
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
  ConstantKmerCounts counts(KmerZygosity::HOMOZYGOUS);
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

