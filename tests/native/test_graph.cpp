#include <fmt/format.h>
#include <gtest/gtest.h>
#include <htslib/bgzf.h>
#include <htslib/tbx.h>

#include <cstdlib>
#include <filesystem>
#include <stdexcept>

#include "graph.hpp"
#include "test_helpers.hpp"
#include "variant.hpp"

namespace fs = std::filesystem;
using namespace npsv3;

class GraphConstructionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!fs::exists(kHG38FastaPath) || !fs::exists(kB37FastaPath)) {
      GTEST_SKIP() << "Reference FASTA(s) not available";
    }
  }

  static const fs::path kB37FastaPath;
  static const fs::path kHG38FastaPath;
};
const fs::path GraphConstructionTest::kB37FastaPath(
    "/storage/mlinderman/projects/sv/npsv3-experiments/resources/human_g1k_v37.fasta");
const fs::path GraphConstructionTest::kHG38FastaPath(
    "/storage/mlinderman/projects/sv/npsv3-experiments/resources/Homo_sapiens_assembly38.fasta");

TEST_F(GraphConstructionTest, OverlappingVariants) {
  test::TestVCFFile vcf(R"VCF(##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##contig=<ID=chr1,length=248956422,md5=2648ae1bacce4ec4b6cf337dcae37816>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	Sample1	Sample2	Sample3	Sample4
chr1	3693767	.	C	CCCATGCAGCCTCAGCCCCTCCTCCCGCAATCCCAGCCATGCAGCCTCAGCTCCTCCTCCCACAATCCCAGCCCTGCAGCCTCAGCTCCTCCTCCCACAATCCCAGCCCTGCAGCCTCAGCCCCTCCTCCCACAATCCCACCCATGCAGCCTCAGCCCCTCCTCCCGCAATCCCAGCCCTGCAGCCTCAGCCCCTCCTCCCGCAATCCCAG	30	.	.	GT	./.	0|1	./.	./.
chr1	3693767	.	C	CCCATGCAGCCTCAGCCCCTCCTCCCGCAATCCCAGCCATGCAGCCTCAGCTCCTCCTCCCACAATCCCAGCCCTGCAGCCTCAGCTCCTCCTCCCACAATCCCAGCCCTGCAGCCTCAGCCCCTCCTCCCACAATCCCACCCATGCAGCCTCAGCCCCTCCTCCCGCAATCCCAGCCCTGCAGCCTCAGCCCCTCCTCCCGCAATCCCAGCCATGCAGCCTCAGCCCCTCCTCCCGCAATCCCAG	30	.	.	GT	./.	./.	./.	0|1
chr1	3693767	.	C	G	30	.	.	GT	./.	./.	./.	./.
chr1	3693767	.	C	G,CCCATGCAGCCTCAGCCCCTCCTCCCGCAATCCCAGCCATGCAGCCTCAGCTCCTCCTCCCACAATCCCAGCCCTGCAGCCTCAGCTCCTCCTCCCACAATCCCAGCCCTGCAGCCTCAGCCCCTCCTCCCACAATCCCACCCATGCAGCCTCAGCCCCTCCTCCCGCAATCCCAGCCCTGCAGCCTCAGCCCCTCCTCCCGCAATCCCAG	30	.	.	GT	./.	./.	1|2	./.)VCF");

  auto region = Range("chr1", 3693757, 3693777);
  Graph graph(kHG38FastaPath, vcf.file_path_, region);
  // Although there 4 variants there are only 2 unique REF and 3 unique ALT alleles. With padding nodes, we expect:
  ASSERT_EQ(graph.NodeCount(), 7);

  // Variants 1(1) and 4(2), and 3(1) and 4(1) should share alternate allele nodes
  auto handles1_1 = graph.PathHandles("_alt_f4a6f765120d8399c009da996db3017b8aa7d488_1");
  auto handles2_1 = graph.PathHandles("_alt_aabc116f14970388d60d7746c32fbf7dcde9b4fc_1");
  auto handles3_1 = graph.PathHandles("_alt_9fd4437f7542de4535db3d1a3ad718d1281c90b1_1");
  auto handles4_1 = graph.PathHandles("_alt_34d6d0a9fdf8d0c44b8e9ba6c250cb86149ed9b9_1");
  auto handles4_2 = graph.PathHandles("_alt_34d6d0a9fdf8d0c44b8e9ba6c250cb86149ed9b9_2");

  ASSERT_EQ(handles4_1.size(), 2) << "Variant 4 ALT allele should have 2 nodes";
  ASSERT_EQ(handles4_2.size(), 2) << "Variant 4 ALT allele should have 2 nodes";

  std::sort(handles1_1.begin(), handles1_1.end());
  std::sort(handles2_1.begin(), handles2_1.end());
  std::sort(handles3_1.begin(), handles3_1.end());
  std::sort(handles4_1.begin(), handles4_1.end());
  std::sort(handles4_2.begin(), handles4_2.end());
  ASSERT_TRUE(std::includes(handles4_1.begin(), handles4_1.end(), handles3_1.begin(), handles3_1.end()));
  ASSERT_TRUE(std::includes(handles4_2.begin(), handles4_2.end(), handles1_1.begin(), handles1_1.end()));

  // Genotype paths
  auto ref_handles = graph.PathHandles(region.contig());
  for (int i = 0; i < 2 /* ploidy */; i++) {
    ASSERT_EQ(graph.PathHandles(fmt::format("Sample1#{}#{}#0", i, region.contig())), ref_handles);
  }
  ASSERT_EQ(graph.PathHandles(fmt::format("Sample2#{}#{}#0", 0, region.contig())), ref_handles);
  ASSERT_NE(graph.PathHandles(fmt::format("Sample2#{}#{}#0", 1, region.contig())), ref_handles);

  // Identify samples traversing the two relevant alternate nodes
  auto nodes1_1 = graph.PathNodes("_alt_f4a6f765120d8399c009da996db3017b8aa7d488_1");
  auto nodes2_1 = graph.PathNodes("_alt_aabc116f14970388d60d7746c32fbf7dcde9b4fc_1");
  std::vector<odgi::nid_t> alt_nodes;
  std::set_union(nodes1_1.begin(), nodes1_1.end(), nodes2_1.begin(), nodes2_1.end(), std::back_inserter(alt_nodes));

  auto samples_with_alt_nodes = graph.SamplesIncluding(alt_nodes);
  std::sort(samples_with_alt_nodes.begin(), samples_with_alt_nodes.end());
  ASSERT_EQ(samples_with_alt_nodes, decltype(samples_with_alt_nodes)({"Sample2", "Sample3", "Sample4"}));
}

class VariantTransitionsGraphConstructionTest
    : public GraphConstructionTest,
      public testing::WithParamInterface<std::tuple<std::string_view, std::string_view, int, int>> {};

TEST_P(VariantTransitionsGraphConstructionTest, TestVariantTransitions) {
  auto [variant0, variant1, sample_0_paths, sample_1_paths] = GetParam();
  test::TestVCFFile vcf(fmt::format(R"VCF(##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##contig=<ID=chr1,length=248956422>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=PS,Number=1,Type=Integer,Description="Phase set identifier">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	Sample
{}
{})VCF",
                                    variant0, variant1));
  auto region = Range("chr1", 999989, 1000010);
  Graph graph(kHG38FastaPath, vcf.file_path_, region);

  int i;
  for (i = 0; i < sample_0_paths; i++) {
    ASSERT_TRUE(graph.HasPath(fmt::format("Sample#0#{}#{}", region.contig(), i)));
  }
  ASSERT_FALSE(graph.HasPath(fmt::format("Sample#0#{}#{}", region.contig(), i)));
  for (i = 0; i < sample_1_paths; i++) {
    ASSERT_TRUE(graph.HasPath(fmt::format("Sample#1#{}#{}", region.contig(), i)));
  }
  ASSERT_FALSE(graph.HasPath(fmt::format("Sample#1#{}#{}", region.contig(), i)));
}

// clang-format off
INSTANTIATE_TEST_SUITE_P(
  VariantTransitions,
  VariantTransitionsGraphConstructionTest,
  testing::Values(std::make_tuple("chr1	1000000	.	G	A	100	PASS	.	GT	0/1", "chr1	1000001	.	G	C	100	PASS	.	GT	0/1", 2, 2),
                  std::make_tuple("chr1	1000000	.	G	A	100	PASS	.	GT	0/1", "chr1	1000001	.	G	C	100	PASS	.	GT	0|1", 2, 2),
                  std::make_tuple("chr1	1000000	.	G	A	100	PASS	.	GT	0/1", "chr1	1000001	.	G	C	100	PASS	.	GT:PS	0|1:1000001", 2, 2),
                  std::make_tuple("chr1	1000000	.	G	A	100	PASS	.	GT	0/1", "chr1	1000001	.	G	C	100	PASS	.	GT	1/1", 1, 1),

                  std::make_tuple("chr1	1000000	.	G	A	100	PASS	.	GT	0|1", "chr1	1000001	.	G	C	100	PASS	.	GT	0/1", 2, 2),
                  std::make_tuple("chr1	1000000	.	G	A	100	PASS	.	GT	0|1", "chr1	1000001	.	G	C	100	PASS	.	GT	0|1", 1, 1),
                  std::make_tuple("chr1	1000000	.	G	A	100	PASS	.	GT	0|1", "chr1	1000001	.	G	C	100	PASS	.	GT:PS	0|1:1000001", 2, 2),
                  std::make_tuple("chr1	1000000	.	G	A	100	PASS	.	GT	0|1", "chr1	1000001	.	G	C	100	PASS	.	GT	1/1", 1, 1),
                
                  std::make_tuple("chr1	1000000	.	G	A	100	PASS	.	GT:PS	0|1:1000000", "chr1	1000001	.	G	C	100	PASS	.	GT	0/1", 2, 2),
                  std::make_tuple("chr1	1000000	.	G	A	100	PASS	.	GT:PS	0|1:1000000", "chr1	1000001	.	G	C	100	PASS	.	GT	0|1", 2, 2),
                  std::make_tuple("chr1	1000000	.	G	A	100	PASS	.	GT:PS	0|1:1000000", "chr1	1000001	.	G	C	100	PASS	.	GT:PS	0|1:1000000", 1, 1),
                  std::make_tuple("chr1	1000000	.	G	A	100	PASS	.	GT:PS	0|1:1000000", "chr1	1000001	.	G	C	100	PASS	.	GT:PS	0|1:1000001", 2, 2),
                  std::make_tuple("chr1	1000000	.	G	A	100	PASS	.	GT:PS	0|1:1000000", "chr1	1000001	.	G	C	100	PASS	.	GT	1/1", 1, 1))
);
// clang-format on

TEST_F(GraphConstructionTest, StarAlleleVariant) {
  test::TestVCFFile vcf(R"VCF(##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##contig=<ID=chr14,length=107043718>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	Sample
chr14	76721238	.	GCC	G	603.88	PASS	.	GT	0/1
chr14	76721239	.	C	CAAAAAAAAAA,*	344.04	PASS	.	GT	1/2)VCF");

  auto region = Range("chr14", 76721228, 76721250);
  Graph graph(kHG38FastaPath, vcf.file_path_, region);
  for (int i=0; i < 2 /* ploidy */; i++) {
    // Each haplotype should only have a single patch because we can implicitly phase the '*' allele
    ASSERT_TRUE(graph.HasPath(fmt::format("Sample#{}#{}#0", i, region.contig())));
    ASSERT_FALSE(graph.HasPath(fmt::format("Sample#{}#{}#1", i, region.contig())));
  }
};

TEST_F(GraphConstructionTest, PermuteStarAlleleVariant) {
  test::TestVCFFile vcf(R"VCF(##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##contig=<ID=chr1,length=248956422>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	Sample
chr1	5414152	.	CGGGCATCTATGATGCTGATTGATGTCCCCAGCATCCCGGGCATCTATGATGCTGATTGATGTCCCCAGCATCCCG	C	.	PASS	.	GT	0/1
chr1	5414226	.	CG	*,C	.	PASS	.	GT	1/2)VCF");

  auto region = Range("chr1", 5414141, 5414237);
  Graph graph(kHG38FastaPath, vcf.file_path_, region);
  for (int i=0; i < 2 /* ploidy */; i++) {
    // Each haplotype should only have a single patch because we can implicitly phase the '*' allele
    ASSERT_TRUE(graph.HasPath(fmt::format("Sample#{}#{}#0", i, region.contig())));
    ASSERT_FALSE(graph.HasPath(fmt::format("Sample#{}#{}#1", i, region.contig())));
  }
};

TEST_F(GraphConstructionTest, ImplicitOverlap) {
  test::TestVCFFile vcf(R"VCF(##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##contig=<ID=chr1,length=248956422,md5=2648ae1bacce4ec4b6cf337dcae37816>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	Sample
chr1	8978661	.	AAAAAAAAAAAAAAC	A	.	PASS	.	GT	0|1
chr1	8978664	.	A	C	.	PASS	.	GT	0|0)VCF");

  auto region = Range("chr1", 8978650, 8978685);
  Graph graph(kHG38FastaPath, vcf.file_path_, region);
  for (int i=0; i < 2 /* ploidy */; i++) {
    // Each haplotype should only have a single patch because we can implicitly phase the '*' allele
    ASSERT_TRUE(graph.HasPath(fmt::format("Sample#{}#{}#0", i, region.contig())));
    ASSERT_FALSE(graph.HasPath(fmt::format("Sample#{}#{}#1", i, region.contig())));
  }
};

TEST_F(GraphConstructionTest, InconsistentHaplotypes) {
  test::TestVCFFile vcf(R"VCF(##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##contig=<ID=chr1,length=248956422,md5=2648ae1bacce4ec4b6cf337dcae37816>
##INFO=<ID=SVTYPE,Number=1,Type=String,Description="Type of structural variant">
##INFO=<ID=SVLEN,Number=A,Type=Integer,Description="Difference in length between REF and ALT alleles">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	Sample
chr1	6012332	.	TGGTGGAGGTGATGAAGGCGGAGGTGGGTGGAGGTGGAGATGGAGGTAGTGGTGGAGGTGATGAAGGCGGA	T	.	.	SVTYPE=DEL;SVLEN=-70	GT	1|1
chr1	6012378	.	T	C	.	.	.	GT	0|1)VCF");

  auto region = Range("chr1", 6012321, 6012412);
  Graph graph(kHG38FastaPath, vcf.file_path_, region);

  // Haplotype 0 should have 1 path, haplotype 1 should be broken at the inconsistent 2nd variant
  ASSERT_TRUE(graph.HasPath(fmt::format("Sample#0#{}#0", region.contig())));
  ASSERT_FALSE(graph.HasPath(fmt::format("Sample#0#{}#1", region.contig())));

  for (int i=0; i < 2 /* segments*/; i++) {
    ASSERT_TRUE(graph.HasPath(fmt::format("Sample#1#{}#{}", region.contig(), i)));
  }
  ASSERT_FALSE(graph.HasPath(fmt::format("Sample#1#{}#{}", region.contig(), 2)));
    
};

TEST_F(GraphConstructionTest, InconsistentHaplotypes2) {
  test::TestVCFFile vcf(R"VCF(##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##contig=<ID=2,length=243199373>
##INFO=<ID=SVTYPE,Number=1,Type=String,Description="Type of structural variant">
##INFO=<ID=SVLEN,Number=A,Type=Integer,Description="Difference in length between REF and ALT alleles">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	Sample
2	39714038	.	T	G,TCG	30	.	.	GT	2|1
2	39714038	.	T	TCTTTCTCTCTTTCTCTCTTTCTCTCTCTCTCTCTCTTTCTCTCTCTCTCTCTCG	20	PASS	SVTYPE=INS;SVLEN=54	GT	1/1)VCF");

  auto region = Range("2", 39714028, 39714048);
  Graph graph(kB37FastaPath, vcf.file_path_, region);

  // Haplotype 0 should be broken at the inconsistent 2nd variant, haplotype 1 should have 1 path
  ASSERT_TRUE(graph.HasPath(fmt::format("Sample#1#{}#0", region.contig())));
  ASSERT_FALSE(graph.HasPath(fmt::format("Sample10#{}#1", region.contig())));

  for (int i=0; i < 2 /* segments*/; i++) {
    ASSERT_TRUE(graph.HasPath(fmt::format("Sample#0#{}#{}", region.contig(), i)));
  }
  ASSERT_FALSE(graph.HasPath(fmt::format("Sample#0#{}#{}", region.contig(), 2)));
    
};