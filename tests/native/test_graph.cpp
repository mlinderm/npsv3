#include <fmt/format.h>
#include <gtest/gtest.h>
#include <htslib/bgzf.h>
#include <htslib/tbx.h>

#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <algorithm>

#include "graph.hpp"
#include "test_helpers.hpp"
#include "variant.hpp"

namespace fs = std::filesystem;
using namespace npsv3;
using npsv3::test::GraphConstructionTest;

TEST_F(GraphConstructionTest, OverlappingVariants) {
  test::TestVCFFile vcf(R"VCF(##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##contig=<ID=chr1,length=248956422,md5=2648ae1bacce4ec4b6cf337dcae37816>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	Sample1	Sample2	Sample3	Sample4	Sample5	Sample6
chr1	3693767	.	C	CCCATGCAGCCTCAGCCCCTCCTCCCGCAATCCCAGCCATGCAGCCTCAGCTCCTCCTCCCACAATCCCAGCCCTGCAGCCTCAGCTCCTCCTCCCACAATCCCAGCCCTGCAGCCTCAGCCCCTCCTCCCACAATCCCACCCATGCAGCCTCAGCCCCTCCTCCCGCAATCCCAGCCCTGCAGCCTCAGCCCCTCCTCCCGCAATCCCAG	30	.	.	GT	./.	0|1	./.	./.	0|1	1|0
chr1	3693767	.	C	CCCATGCAGCCTCAGCCCCTCCTCCCGCAATCCCAGCCATGCAGCCTCAGCTCCTCCTCCCACAATCCCAGCCCTGCAGCCTCAGCTCCTCCTCCCACAATCCCAGCCCTGCAGCCTCAGCCCCTCCTCCCACAATCCCACCCATGCAGCCTCAGCCCCTCCTCCCGCAATCCCAGCCCTGCAGCCTCAGCCCCTCCTCCCGCAATCCCAGCCATGCAGCCTCAGCCCCTCCTCCCGCAATCCCAG	30	.	.	GT	./.	./.	./.	0|1	1|0	1|0
chr1	3693767	.	C	G	30	.	.	GT	./.	./.	./.	./.	./.	./.
chr1	3693767	.	C	G,CCCATGCAGCCTCAGCCCCTCCTCCCGCAATCCCAGCCATGCAGCCTCAGCTCCTCCTCCCACAATCCCAGCCCTGCAGCCTCAGCTCCTCCTCCCACAATCCCAGCCCTGCAGCCTCAGCCCCTCCTCCCACAATCCCACCCATGCAGCCTCAGCCCCTCCTCCCGCAATCCCAGCCCTGCAGCCTCAGCCCCTCCTCCCGCAATCCCAG	30	.	.	GT	./.	./.	1|2	./.	./.	./.)VCF");

  auto region = Range("chr1", 3693757, 3693777);
  Graph graph(HG38FastaPath_, vcf.file_path_, region);
 
  //graph.ToGFA(std::cout);
  // TODO: Should co-located insertions stack, such there could be paths traversing both insertions? At present, we split paths (e.g., Sample6 above).
  // An alternative could be to generate all combinations of co-located insertions as distinct alternate alleles, tagged with the relevant original
  // variant IDs. These would not be part of the allele paths, could be used when generating possible haplotypes.

  // Although there 4 variants there are only 2 unique REF and 3 unique ALT alleles. With padding nodes, we expect:
  ASSERT_EQ(graph.get_node_count(), 7);

  // Nodes 2 & 3, the SNV, should link to the insertion REF and ALT alleles (4, 5, 6)
  for (int l=2; l<=3; l++) {
    for (int r=4; r<=6; r++) {
      ASSERT_TRUE(graph.has_edge(graph.get_handle(l), graph.get_handle(r)));
    }
  }

  // Variants 1(1) and 4(2), and 3(1) and 4(1) should share alternate allele nodes
  auto handles1_1 = graph.PathHandles("_alt_f4a6f765120d8399c009da996db3017b8aa7d488_1");
  auto handles2_1 = graph.PathHandles("_alt_aabc116f14970388d60d7746c32fbf7dcde9b4fc_1");
  auto handles3_1 = graph.PathHandles("_alt_9fd4437f7542de4535db3d1a3ad718d1281c90b1_1");
  auto handles4_1 = graph.PathHandles("_alt_34d6d0a9fdf8d0c44b8e9ba6c250cb86149ed9b9_1");
  auto handles4_2 = graph.PathHandles("_alt_34d6d0a9fdf8d0c44b8e9ba6c250cb86149ed9b9_2");

  ASSERT_EQ(handles4_1.size(), 2) << "Variant 4 ALT allele 1 should have 2 nodes";
  ASSERT_EQ(handles4_2.size(), 2) << "Variant 4 ALT allele 2 should have 2 nodes";

  std::sort(handles1_1.begin(), handles1_1.end());
  std::sort(handles2_1.begin(), handles2_1.end());
  std::sort(handles3_1.begin(), handles3_1.end());
  std::sort(handles4_1.begin(), handles4_1.end());
  std::sort(handles4_2.begin(), handles4_2.end());
  ASSERT_TRUE(std::includes(handles4_1.begin(), handles4_1.end(), handles3_1.begin(), handles3_1.end()));
  ASSERT_TRUE(std::includes(handles4_2.begin(), handles4_2.end(), handles1_1.begin(), handles1_1.end()));

  // Genotype paths
  for (int s=1; s <= 5 /* samples */; s++) {
    for (int p=0; p < 2 /* ploidy */; p++) {
      // Each haplotype should only have a single path
      ASSERT_TRUE(graph.has_path(fmt::format("Sample{}#{}#{}#0", s, p, region.contig())));
      ASSERT_FALSE(graph.has_path(fmt::format("Sample{}#{}#{}#1", s, p, region.contig())));
    }
  } 

  // Identify samples traversing the relevant alternate nodes
  std::set<odgi::nid_t> alt_nodes;
  for (auto path_name : std::vector<std::string>{"_alt_f4a6f765120d8399c009da996db3017b8aa7d488_1",
                                                 "_alt_aabc116f14970388d60d7746c32fbf7dcde9b4fc_1",
                                                 "_alt_34d6d0a9fdf8d0c44b8e9ba6c250cb86149ed9b9_2"}) {
    auto path_nodes = graph.PathNodes(path_name);
    auto ref_nodes = graph.PathNodes(path_name.substr(0, path_name.size() - 1) + "0");
    auto [path_begin, path_end, _ref_begin, _ref_end] = detail::TrimSequence(path_nodes, ref_nodes);
    alt_nodes.insert(path_begin, path_end);
  }

  auto samples_with_alt_nodes = graph.SamplesIncluding(std::vector<odgi::nid_t>(alt_nodes.begin(), alt_nodes.end()));
  std::sort(samples_with_alt_nodes.begin(), samples_with_alt_nodes.end());
  ASSERT_EQ(samples_with_alt_nodes, decltype(samples_with_alt_nodes)({"Sample2", "Sample3", "Sample4", "Sample5", "Sample6"}));

  auto paths = graph.AllPaths(vcf.file_path_, region.contig(), region);
  ASSERT_EQ(paths.total_paths(), 3);  // 1 REF + 2 unique ALT alleles (without "stacking" co-located insertions)

  std::vector<std::pair<Graph::PathIdSet, std::string>> path_sequences;
  paths.ForEachPath([&](const AllPathGraphOverlay::CallbackIter& begin, const AllPathGraphOverlay::CallbackIter& end, const Graph::PathIdSet& inference_paths) {
    path_sequences.emplace_back(inference_paths, graph.PathSequence(begin, end));
  });
  ASSERT_EQ(path_sequences.size(), paths.total_paths());
  std::sort(std::begin(path_sequences), std::end(path_sequences));
  ASSERT_EQ(path_sequences[0],
            std::make_pair(Graph::PathIdSet(std::string("00101010100")), std::string("ACAATCCCACCCATGCAGCC")));
  ASSERT_EQ(path_sequences[1],
            std::make_pair(Graph::PathIdSet(std::string("00101100100")), std::string("ACAATCCCACCCATGCAGCCTCAGCCCCTCCTCCCGCAATCCCAGCCATGCAGCCTCAGCTCCTCCTCCCACAATCCCAGCCCTGCAGCCTCAGCTCCTCCTCCCACAATCCCAGCCCTGCAGCCTCAGCCCCTCCTCCCACAATCCCACCCATGCAGCCTCAGCCCCTCCTCCCGCAATCCCAGCCCTGCAGCCTCAGCCCCTCCTCCCGCAATCCCAGCCATGCAGCCTCAGCCCCTCCTCCCGCAATCCCAGCCATGCAGCC")));
  ASSERT_EQ(path_sequences[2],
            std::make_pair(Graph::PathIdSet(std::string("10001011000")), std::string("ACAATCCCACCCATGCAGCCTCAGCCCCTCCTCCCGCAATCCCAGCCATGCAGCCTCAGCTCCTCCTCCCACAATCCCAGCCCTGCAGCCTCAGCTCCTCCTCCCACAATCCCAGCCCTGCAGCCTCAGCCCCTCCTCCCACAATCCCACCCATGCAGCCTCAGCCCCTCCTCCCGCAATCCCAGCCCTGCAGCCTCAGCCCCTCCTCCCGCAATCCCAGCCATGCAGCC")));
}

TEST_F(GraphConstructionTest, NoLinksBetweenAltAllelesInSameVariant) {
  test::TestVCFFile vcf(R"VCF(##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##contig=<ID=chr1,length=248956422,md5=2648ae1bacce4ec4b6cf337dcae37816>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	Sample1	Sample2	Sample3	Sample4
chr1	3693767	.	C	G,CCCATGCAGCCTCAGCCCCTCCTCCCGCAATCCCAGCCATGCAGCCTCAGCTCCTCCTCCCACAATCCCAGCCCTGCAGCCTCAGCTCCTCCTCCCACAATCCCAGCCCTGCAGCCTCAGCCCCTCCTCCCACAATCCCACCCATGCAGCCTCAGCCCCTCCTCCCGCAATCCCAGCCCTGCAGCCTCAGCCCCTCCTCCCGCAATCCCAG	30	.	.	GT	./.	./.	1|2	./.)VCF");

  auto region = Range("chr1", 3693757, 3693777);
  Graph graph(HG38FastaPath_, vcf.file_path_, region);
  graph.ToGFA(std::cout);

  // There are 2 unique REF and 2 unique ALT alleles. With padding nodes, we expect:
  ASSERT_EQ(graph.get_node_count(), 6);

  // The 'G' should not link to the insertion as that is excluded by the multi-allelic variant representation. Note 
  // we may want to change this in the future to allow paths between ALT alleles within a variant.
  auto ref_nodes = graph.PathHandles("_alt_34d6d0a9fdf8d0c44b8e9ba6c250cb86149ed9b9_0");
  auto alt1_nodes = graph.PathHandles("_alt_34d6d0a9fdf8d0c44b8e9ba6c250cb86149ed9b9_1");
  auto alt2_nodes = graph.PathHandles("_alt_34d6d0a9fdf8d0c44b8e9ba6c250cb86149ed9b9_2");
  ASSERT_TRUE(graph.has_edge(ref_nodes.front(), alt1_nodes.back()));
  ASSERT_TRUE(graph.has_edge(ref_nodes.front(), alt2_nodes.back()));
  ASSERT_TRUE(graph.has_edge(alt1_nodes.front(), alt1_nodes.back()));
  ASSERT_FALSE(graph.has_edge(alt1_nodes.front(), alt2_nodes.back()));
  
  auto paths = graph.AllPaths(vcf.file_path_, region.contig(), region);
  ASSERT_EQ(paths.total_paths(), 2);
  
  std::vector<std::pair<Graph::PathIdSet, std::string>> path_sequences;
  paths.ForEachPath([&](const AllPathGraphOverlay::CallbackIter& begin, const AllPathGraphOverlay::CallbackIter& end, const Graph::PathIdSet& inference_paths) {
    path_sequences.emplace_back(inference_paths, graph.PathSequence(begin, end));
  });
  ASSERT_EQ(path_sequences.size(), paths.total_paths());

  // Since inference paths are "one-hot" with reference allele first, sorting should ensure the reference path is always first
  std::sort(std::begin(path_sequences), std::end(path_sequences));
  ASSERT_EQ(path_sequences[0],
            std::make_pair(Graph::PathIdSet(std::string("00100")), std::string("ACAATCCCACCCATGCAGCC")));
  ASSERT_EQ(path_sequences[1],
            std::make_pair(Graph::PathIdSet(std::string("10000")),
                           std::string("ACAATCCCACCCATGCAGCCTCAGCCCCTCCTCCCGCAATCCCAGCCATGCAGCCTCAGCTCCTCCTCCCACAATCCCAGCCCTGCAGCCTCAGCTCCTCCTCCCACAATCCCAGCCCTGCAGCCTCAGCCCCTCCTCCCACAATCCCACCCATGCAGCCTCAGCCCCTCCTCCCGCAATCCCAGCCCTGCAGCCTCAGCCCCTCCTCCCGCAATCCCAGCCATGCAGCC")));
}

TEST_F(GraphConstructionTest, LinksBetweenAltAllelesInSameVariant) {
  test::TestVCFFile vcf(R"VCF(##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##contig=<ID=chr1,length=248956422,md5=2648ae1bacce4ec4b6cf337dcae37816>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	Sample1	Sample2	Sample3	Sample4
chr1	3693767	.	C	G,CCCATGCAGCCTCAGCCCCTCCTCCCGCAATCCCAGCCATGCAGCCTCAGCTCCTCCTCCCACAATCCCAGCCCTGCAGCCTCAGCTCCTCCTCCCACAATCCCAGCCCTGCAGCCTCAGCCCCTCCTCCCACAATCCCACCCATGCAGCCTCAGCCCCTCCTCCCGCAATCCCAGCCCTGCAGCCTCAGCCCCTCCTCCCGCAATCCCAG	30	.	.	GT	./.	./.	1|2	./.)VCF");

  auto region = Range("chr1", 3693757, 3693777);
  Graph graph(HG38FastaPath_, vcf.file_path_, region, false /* enforce_multiallelic */);
  //graph.ToGFA(std::cout);

  auto alt1_nodes = graph.PathHandles("_alt_34d6d0a9fdf8d0c44b8e9ba6c250cb86149ed9b9_1");
  auto alt2_nodes = graph.PathHandles("_alt_34d6d0a9fdf8d0c44b8e9ba6c250cb86149ed9b9_2");
  ASSERT_TRUE(graph.has_edge(alt1_nodes.front(), alt2_nodes.back()));
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
  Graph graph(HG38FastaPath_, vcf.file_path_, region);

  int i;
  for (i = 0; i < sample_0_paths; i++) {
    ASSERT_TRUE(graph.has_path(fmt::format("Sample#0#{}#{}", region.contig(), i)));
  }
  ASSERT_FALSE(graph.has_path(fmt::format("Sample#0#{}#{}", region.contig(), i)));
  for (i = 0; i < sample_1_paths; i++) {
    ASSERT_TRUE(graph.has_path(fmt::format("Sample#1#{}#{}", region.contig(), i)));
  }
  ASSERT_FALSE(graph.has_path(fmt::format("Sample#1#{}#{}", region.contig(), i)));
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
  Graph graph(HG38FastaPath_, vcf.file_path_, region);
  for (int i=0; i < 2 /* ploidy */; i++) {
    // Each haplotype should only have a single patch because we can implicitly phase the '*' allele
    ASSERT_TRUE(graph.has_path(fmt::format("Sample#{}#{}#0", i, region.contig())));
    ASSERT_FALSE(graph.has_path(fmt::format("Sample#{}#{}#1", i, region.contig())));
  }

  ASSERT_EQ(graph.PathSequence("Sample#0#chr14#0"), "GATTCCGTTGCAAAAAAAAAACAAAAAAAAGA");
  ASSERT_EQ(graph.PathSequence("Sample#1#chr14#0"), "GATTCCGTTGAAAAAAAAGA");
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
  Graph graph(HG38FastaPath_, vcf.file_path_, region);
  for (int i=0; i < 2 /* ploidy */; i++) {
    // Each haplotype should only have a single patch because we can implicitly phase the '*' allele
    ASSERT_TRUE(graph.has_path(fmt::format("Sample#{}#{}#0", i, region.contig())));
    ASSERT_FALSE(graph.has_path(fmt::format("Sample#{}#{}#1", i, region.contig())));
  }

  ASSERT_EQ(graph.PathSequence("Sample#0#chr1#0"), "CCCAGCATCCCGGGCATCTATGATGCTGATTGATGTCCCCAGCATCCCGGGCATCTATGATGCTGATTGATGTCCCCAGCATCCCGGGCATCTAT");
  ASSERT_EQ(graph.PathSequence("Sample#1#chr1#0"), "CCCAGCATCCCGGGCATCTAT");
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
  Graph graph(HG38FastaPath_, vcf.file_path_, region);
  for (int i=0; i < 2 /* ploidy */; i++) {
    // Each haplotype should only have a single path because we can implicitly phase the '*' allele
    ASSERT_TRUE(graph.has_path(fmt::format("Sample#{}#{}#0", i, region.contig())));
    ASSERT_FALSE(graph.has_path(fmt::format("Sample#{}#{}#1", i, region.contig())));
  }
};

TEST_F(GraphConstructionTest, InconsistentHaplotypes1) {
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
  Graph graph(HG38FastaPath_, vcf.file_path_, region);

  // Haplotype 0 should have 1 path, haplotype 1 should be broken at the inconsistent 2nd variant
  ASSERT_TRUE(graph.has_path(fmt::format("Sample#0#{}#0", region.contig())));
  ASSERT_FALSE(graph.has_path(fmt::format("Sample#0#{}#1", region.contig())));

  for (int i=0; i < 2 /* segments*/; i++) {
    ASSERT_TRUE(graph.has_path(fmt::format("Sample#1#{}#{}", region.contig(), i)));
  }
  ASSERT_FALSE(graph.has_path(fmt::format("Sample#1#{}#{}", region.contig(), 2)));
  
  ASSERT_EQ(graph.PathSequence("Sample#0#chr1#0"), "ATGGAGGTAGTGGTGGAGGTG");
  ASSERT_EQ(graph.PathSequence("Sample#1#chr1#0"), "ATGGAGGTAGT");
  // When breaking on inconsistent genotypes, we don't re-add all references nodes that might
  // have been previously deleted, just those needed to restart the haplotype.
  ASSERT_EQ(graph.PathSequence("Sample#1#chr1#1"), "CAGTGGTGGAGGTGATGAAGGCGGAGGTGGAGGTG");
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
  Graph graph(B37FastaPath_, vcf.file_path_, region);

  // Haplotype 0 should be broken at the inconsistent 2nd variant, haplotype 1 should have 1 path
  ASSERT_TRUE(graph.has_path(fmt::format("Sample#1#{}#0", region.contig())));
  ASSERT_FALSE(graph.has_path(fmt::format("Sample10#{}#1", region.contig())));

  for (int i=0; i < 2 /* segments*/; i++) {
    ASSERT_TRUE(graph.has_path(fmt::format("Sample#0#{}#{}", region.contig(), i)));
  }
  ASSERT_FALSE(graph.has_path(fmt::format("Sample#0#{}#{}", region.contig(), 2))); 
  
  ASSERT_EQ(graph.PathSequence("Sample#0#2#0"), "CTCTCTCTCTCG");
  ASSERT_EQ(graph.PathSequence("Sample#0#2#1"), "CTTTCTCTCTTTCTCTCTTTCTCTCTCTCTCTCTCTTTCTCTCTCTCTCTCTCGCTTTCTCGCT");
  ASSERT_EQ(graph.PathSequence("Sample#1#2#0"), "CTCTCTCTCGCTTTCTCTCTTTCTCTCTTTCTCTCTCTCTCTCTCTTTCTCTCTCTCTCTCTCGCTTTCTCGCT");
};

TEST_F(GraphConstructionTest, InconsistentHaplotypes3) {
  test::TestVCFFile vcf(R"VCF(##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##contig=<ID=chr1,length=248956422>
##INFO=<ID=SVTYPE,Number=1,Type=String,Description="Type of structural variant">
##INFO=<ID=SVLEN,Number=A,Type=Integer,Description="Difference in length between REF and ALT alleles">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	Sample
chr1	904479	.	GCCTCCTCCGAACGCGGCCGCCTCCTCCTCCGAACGTGGCCTCCTCCGAACGCGGCCGCCTCCTCCTCCGAACGCGGCCGCCTCCTCCTCCGAACGTGGCCTCCTCCGAACGTGGCCGCCTCCTCCTCCGAACGTGGCCTCCTCCGAACGCGGCCGCCGCCTCCTCCGAACGCGGCCTCCT	T	30	PASS	.	GT	0|1
chr1	904493	.	CGGCCGCCTCCTCCTCCGAACGTGGCCTCCTCCGAACGCGGCCGCCTCCTCCTCCGAACGCGGCCGCCTCCTCCTCCGAACGTGGCCTCCTCCGAACGT	C	30	PASS	.	GT	.
chr1	904493	.	CGGCCGCCTCCTCCTCCGAACGTGGCCTCCTCCGAACGCGGCCGCCTCCTCCTCCGAACGCGGCCGCCTCCTCCTCCGAACGTGGCCTCCTCCGAACGT	C,TGGCCGCCTCCTCCTCCGAACGTGGCCTCCTCCGAACGCGGCCGCCTCCTCCTCCGAACGCGGCCGCCTCCTCCTCCGAACGTGGCCTCCTCCGAACGT	30	PASS	.	GT	2|1)VCF");

  auto region = Range("chr1", 904469, 904671);
  Graph graph(HG38FastaPath_, vcf.file_path_, region);
  graph.ToGFA(std::cout);

  // Haplotype 1 should be broken at the inconsistent 3rd variant, haplotype 0 should have 1 path
  ASSERT_TRUE(graph.has_path(fmt::format("Sample#0#{}#0", region.contig())));
  ASSERT_FALSE(graph.has_path(fmt::format("Sample#0#{}#1", region.contig())));
  for (int i=0; i < 2 /* segments*/; i++) {
    ASSERT_TRUE(graph.has_path(fmt::format("Sample#1#{}#{}", region.contig(), i)));
  }
  ASSERT_FALSE(graph.has_path(fmt::format("Sample#1#{}#{}", region.contig(), 2)));

  // TODO, getting duplicate deletion nodes for variants 2 and 3, i.e.,
  // _alt_31734a3cce2a0fe500860950a5b96e012efee56e_1 and _alt_8a8eec2f139003c73f9bbc38c62fd85e6ded0f7d_1
  // should share nodes.

  auto handles2_1 = graph.PathHandles("_alt_31734a3cce2a0fe500860950a5b96e012efee56e_1");
  auto handles3_1 = graph.PathHandles("_alt_8a8eec2f139003c73f9bbc38c62fd85e6ded0f7d_1");

  std::sort(handles2_1.begin(), handles2_1.end());
  std::sort(handles3_1.begin(), handles3_1.end());
  
  // The simple deletion should be a subpath of the complex deletion
  ASSERT_TRUE(std::includes(handles3_1.begin(), handles3_1.end(), handles2_1.begin(), handles2_1.end()));
}

TEST_F(GraphConstructionTest, AdjacentInsertions) {
  test::TestVCFFile vcf(R"VCF(##fileformat=VCFv4.2
##contig=<ID=chr4,length=190214555>
##INFO=<ID=SVTYPE,Number=1,Type=String,Description="Type of structural variant">
##INFO=<ID=SVLEN,Number=A,Type=Integer,Description="Difference in length between REF and ALT alleles">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	Sample
chr4	99589035	.	A	ACATATATATGTTCATATATATATTCATATATATATGTTCATGTATATTCATATATATATGTTCATATATATATTCATATATATGTTCATATATATATTCATATATATATGTTCATATATATATTCATATATATATGTTCATATATATATTCATATATATATGTTCATATATATATTCATATATATATGTTCATATATATATTCATATATATGTTCATATATATATTCATATATATATGTTCATATATATATTCATATATATATGTTCATATATATATTCATATATATATGTTCATATATATATTCATATATATATGTTCATATATATATTCATATATATATGTTCATATATATATTCATATATATATGTTCATATATATATTCATATATATATGTTCATATATATATTCATATATATATGTTCATATATATATTCATATATATATGTTCATATATATATTCATATATATATGTTCATATATATATTCATATATATATGTTCATATATATATTCATATATATATGTTCATATATATATTCATATATATATGTTCATATATATATT	30	.	SVTYPE=INS;SVLEN=564	GT	0|1
chr4	99589036	.	C	TATATATATGTTCATATATATATTC	30	.	SVTYPE=INS;SVLEN=24	GT	1|0)VCF");

  auto region = Range("chr4", 99589024, 99589046);
  Graph graph(HG38FastaPath_, vcf.file_path_, region);

  // The second insertion has a nominal right padding of 1. With that, the two insertions collapse. To prevent that
  // collapse we only remove right padding when both (ref, alt) alleles have length > 1.

  for (int i=0; i < 2 /* ploidy */; i++) {
    // Each haplotype should only have a single path
    ASSERT_TRUE(graph.has_path(fmt::format("Sample#{}#{}#0", i, region.contig())));
    ASSERT_FALSE(graph.has_path(fmt::format("Sample#{}#{}#1", i, region.contig())));
  }

  ASSERT_EQ(graph.PathSequence("Sample#0#chr4#0"), "ATATATACACATATATATATGTTCATATATATATTCATATATATGT");
  ASSERT_EQ(graph.PathSequence("Sample#1#chr4#0"), "ATATATACACACATATATATGTTCATATATATATTCATATATATATGTTCATGTATATTCATATATATATGTTCATATATATATTCATATATATGTTCATATATATATTCATATATATATGTTCATATATATATTCATATATATATGTTCATATATATATTCATATATATATGTTCATATATATATTCATATATATATGTTCATATATATATTCATATATATGTTCATATATATATTCATATATATATGTTCATATATATATTCATATATATATGTTCATATATATATTCATATATATATGTTCATATATATATTCATATATATATGTTCATATATATATTCATATATATATGTTCATATATATATTCATATATATATGTTCATATATATATTCATATATATATGTTCATATATATATTCATATATATATGTTCATATATATATTCATATATATATGTTCATATATATATTCATATATATATGTTCATATATATATTCATATATATATGTTCATATATATATTCATATATATATGTTCATATATATATTCATATATATATGTTCATATATATATTCATATATATGT");
};

TEST_F(GraphConstructionTest, MixedAlleles) {
  test::TestVCFFile vcf(R"VCF(##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##contig=<ID=1,length=249250621>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=PS,Number=1,Type=Integer,Description="Phase set identifier">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	Sample
1	1000000	.	T	A	100	PASS	.	GT	0/1
1	1000001	.	G	C	100	PASS	.	GT:PS	0|1:1000000
1	1000002	.	G	T	100	PASS	.	GT:PS	1|1:1000000
1	1000003	.	G	A	100	PASS	.	GT:PS	1|0:1000003
1	1000004	.	C	T	100	PASS	.	GT	0/0
1	1000005	.	A	G	100	PASS	.	GT	1/1
1	1000006	.	C	G	100	PASS	.	GT	0|1)VCF");

  auto region = Range("1", 999990, 1000016);
  Graph graph(B37FastaPath_, vcf.file_path_, region);

  ASSERT_EQ(graph.PathSequence("Sample#0#1#0"), "CCAGGGCCGT");
  ASSERT_EQ(graph.PathSequence("Sample#0#1#1"), "GT");
  ASSERT_EQ(graph.PathSequence("Sample#0#1#2"), "ACG");
  ASSERT_EQ(graph.PathSequence("Sample#0#1#3"), "CAGCCTCACCC");

  ASSERT_EQ(graph.PathSequence("Sample#1#1#0"), "CCAGGGCCGA");
  ASSERT_EQ(graph.PathSequence("Sample#1#1#1"), "CT");
  ASSERT_EQ(graph.PathSequence("Sample#1#1#2"), "GCG");
  ASSERT_EQ(graph.PathSequence("Sample#1#1#3"), "GAGCCTCACCC");

   for (int i=0; i < 2 /* ploidy */; i++) {
    // Each haplotype should only have 4 path segments
    ASSERT_FALSE(graph.has_path(fmt::format("Sample#{}#{}#4", i, region.contig())));
    ASSERT_FALSE(graph.has_path(fmt::format("Sample#{}#{}#4", i, region.contig())));
  }
};

TEST_F(GraphConstructionTest, VariantsOverlappingGraphRegion) {
  test::TestVCFFile vcf(R"VCF(##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##contig=<ID=chr1,length=248956422,md5=2648ae1bacce4ec4b6cf337dcae37816>
##INFO=<ID=SVLEN,Number=.,Type=Integer,Description="Difference in length between REF and ALT alleles">
##INFO=<ID=SVTYPE,Number=1,Type=String,Description="Type of structural variant">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=PS,Number=1,Type=Integer,Description="Phase set identifier">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	NA12877	NA12878	NA12879	NA12881	NA12882	NA12885	NA12886
chr1	1978993	HIFI_sawfish:1:193:0:0	GAGGCTGCACAGAACACGTGTGTCGTGCTGAGCTGGGCGTGGGAAGGCGTCATGTGACGAGGCTGCACAGAACATGCGTGTGGTACTGAGCTGGGCGTGGGAAGGTGTCACGTGACAAGGCTGCACAGAACATGTGTGTGGTACTGAGCTGGGCGTGGGAAGGCATCATGTGACA	G	999	PASS	SVTYPE=DEL;SVLEN=-174	GT:PS	0|1:1978993	1|0:1978993	0|1:1978993	1|0:1978993	1|1:.	0|1:1978993	1|0:1978993
chr1	1979575	HIFI_sawfish:1:193:0:1	GGCTGCGCAGAACATGCGTGTGGTACTGAGCTGGGTGTGGGAAGGCATCACGTGACGAGGCTGCGCAGAACACGTGTGTCGTGCTGAGCTGGGCGTGGGAAGGTGTCGCGTGACGAGGCTGCGCAGAACACGCATGTCATGCTGAGCTGGGTGTGGGAAGGCGTCACGTGACGAGGCTGTGCAGAACACGCGTGTGGTACTGACCTGGGTGTGGGAAGGCGTCACATGACGAAGCTGCGCAGAACACGCGTGTGGTACTGACCTGGGTGTGGGAAGGCGTCACATGACGAA	G	999	PASS	SVTYPE=DEL;SVLEN=-290	GT:PS	0|1:1978993	1|0:1978993	0|1:1978993	1|0:1978993	1|1:.	0|1:1978993	1|0:1978993
chr1	1980058	HIFI_sawfish:1:193:0:2	ACCCTCTTACCGCGTGGGGAGGACGGGTGAACGAGAGTGTATCTAAGCCACCGGCACAGATCGCAGTGGGCGCCCTCTTACCGCGTGGGGAGGACGGGTGAACGAGAGACTGTATCTAAGCCACCGGCACAGATCGCAGTGGGCGCCCTCTTACCGCGTGGGGAGGACGGGTGAACGAGAGACTGTATCTAAGCCACCGGCACAGATCGCAGTGGGCG	A	999	PASS	SVTYPE=DEL;SVLEN=-217	GT:PS	0|1:1978993	1|0:1978993	0|1:1978993	1|0:1978993	1|1:.	0|1:1978993	1|0:1978993)VCF");

  auto region = Range("chr1", 1979058, 1981275);
  Graph graph(HG38FastaPath_, vcf.file_path_, region);

  // The first variant only partially overlaps the region, so it should be ignored during graph construction
  ASSERT_FALSE(graph.has_path("_alt_8eeff63608a6ce2fa6141be1409abe9bf4a08eef_0")); 
};

TEST_F(GraphConstructionTest, VariantsAtGraphRegionBoundary) {
  test::TestVCFFile vcf(R"VCF(##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##contig=<ID=chr1,length=248956422,md5=2648ae1bacce4ec4b6cf337dcae37816>
##INFO=<ID=SVLEN,Number=.,Type=Integer,Description="Difference in length between REF and ALT alleles">
##INFO=<ID=SVTYPE,Number=1,Type=String,Description="Type of structural variant">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=PS,Number=1,Type=Integer,Description="Phase set identifier">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	NA12877	NA12878	NA12879	NA12881	NA12882	NA12885	NA12886
chr1	52277191	.	TCTATTGTTAGTAAAATAC	T	.	PASS	.	GT	0|1	0|0	0|0	0|0	0|0	0|0	0|0
chr1	52278191	HIFI_sawfish:0:1743:0:0	T	TGGAAAATTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTGAGACGGAGTCTCGCTCTGTCGCCCAGGCTGGAGTGCAGTGGCGGGATCTCGGCTCACTGCAAGCTCCGCCTCCCGGGTTCACGCCATTCTCCTGCCTCAGCCTCCCAAGTAGCTGGGACTACAGGCGCCCGCCACTACGCCCGGCTAATTTTTTTGTATTTTTAGTAGAGACGGGGTTTCACCGTTTTAGCCGGGATGGTCTCGATCTCCTGACCTCGTGATCCGCCCGCCTCGGC	999	PASS	SVTYPE=INS;SVLEN=279	GT	0|1	0|0	0|0	0|0	0|0	0|0	0|0)VCF");
  auto region = Range("chr1", 52277191, 52279191);
  Graph graph(HG38FastaPath_, vcf.file_path_, region);

  // The first variant only partially overlaps the region, so it should be ignored during graph construction
  ASSERT_FALSE(graph.has_path("_alt_70e6adb077463a6f66e692cdb11f2ca4540ff066_0"));
};

TEST_F(GraphConstructionTest, IgnoreMissingGenotypes) {
  test::TestVCFFile vcf(R"VCF(##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##contig=<ID=chr1,length=248956422,md5=2648ae1bacce4ec4b6cf337dcae37816>
##INFO=<ID=SVLEN,Number=.,Type=Integer,Description="Difference in length between REF and ALT alleles">
##INFO=<ID=SVTYPE,Number=1,Type=String,Description="Type of structural variant">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=PS,Number=1,Type=Integer,Description="Phase set identifier">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	NA12877	NA12878
chr1	52277191	.	TCTATTGTTAGTAAAATAC	T	.	PASS	.	GT	0|1	.)VCF");
  auto region = Range("chr1", 52277181, 52277220);
  ASSERT_NO_THROW({
    Graph graph(HG38FastaPath_, vcf.file_path_, region);
  });
};

TEST_F(GraphConstructionTest, HandleGraphInterfaceErrors) {
  test::TestVCFFile vcf(R"VCF(##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##contig=<ID=chr1,length=248956422,md5=2648ae1bacce4ec4b6cf337dcae37816>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	Sample1
chr1	52277191	.	TCTATTGTTAGTAAAATAC	T	.	PASS	.	GT	0/1
)VCF");

  auto region = Range("chr1", 52277181, 52277219);
  Graph graph(HG38FastaPath_, vcf.file_path_, region);

  ASSERT_THROW(graph.PathSequence("1"), std::out_of_range);
};

TEST_F(GraphConstructionTest, GraphSerializationRoundtrip) {
  test::TestVCFFile vcf(R"VCF(##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##contig=<ID=chr1,length=248956422,md5=2648ae1bacce4ec4b6cf337dcae37816>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	Sample1
chr1	52277191	.	TCTATTGTTAGTAAAATAC	T	.	PASS	.	GT	0/1
)VCF");

  auto region = Range("chr1", 52277181, 52277219);
  Graph original(HG38FastaPath_, vcf.file_path_, region);

  test::TempDir dir;
  auto bin_path = (dir.path_ / "graph.bin").string();
  original.Save(bin_path);

  auto loaded = Graph::Load(bin_path);
  ASSERT_NE(loaded, nullptr);

  EXPECT_EQ(loaded->get_node_count(), original.get_node_count());
  EXPECT_EQ(loaded->min_node_id(), original.min_node_id());
  EXPECT_EQ(loaded->max_node_id(), original.max_node_id());

  // Verify all nodes have matching presence, sequence, and outgoing edges.
  for (odgi::nid_t nid = original.min_node_id(); nid <= original.max_node_id(); ++nid) {
    ASSERT_EQ(original.has_node(nid), loaded->has_node(nid)) << "has_node mismatch at node " << nid;
    if (!original.has_node(nid)) continue;

    auto orig_h = original.get_handle(nid);
    auto load_h = loaded->get_handle(nid);
    EXPECT_EQ(original.get_sequence(orig_h), loaded->get_sequence(load_h)) << "sequence mismatch at node " << nid;

    std::vector<handlegraph::handle_t> orig_succs, load_succs;
    original.follow_edges(orig_h, false, [&](const handlegraph::handle_t& h) { orig_succs.push_back(h); return true; });
    loaded->follow_edges(load_h,  false, [&](const handlegraph::handle_t& h) { load_succs.push_back(h);  return true; });
    std::sort(orig_succs.begin(), orig_succs.end());
    std::sort(load_succs.begin(), load_succs.end());
    EXPECT_EQ(orig_succs, load_succs) << "successor edges mismatch at node " << nid;
  }

  // Verify paths are identical 
  std::vector<handlegraph::path_handle_t> orig_paths, load_paths;
  original.for_each_path_handle([&](const handlegraph::path_handle_t& h) { orig_paths.push_back(h); });
  loaded->for_each_path_handle([&](const handlegraph::path_handle_t& h) { load_paths.push_back(h); });
  std::sort(orig_paths.begin(), orig_paths.end());
  std::sort(load_paths.begin(), load_paths.end());
  EXPECT_EQ(orig_paths, load_paths) << "path handle mismatch";
  
  // Verify path node lists and sequences for the reference contig and each sample haplotype.
  for (const auto& path : orig_paths) {
    EXPECT_EQ(original.PathNodes(path), loaded->PathNodes(path)) << "node list mismatch for path " << original.get_path_name(path);
    EXPECT_EQ(original.PathSequence(path), loaded->PathSequence(path)) << "sequence mismatch for path "  << original.get_path_name(path);
  }
}