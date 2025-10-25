#include <cstdlib>
#include <filesystem>
#include <stdexcept>

#include <gtest/gtest.h>
#include <htslib/bgzf.h>
#include <htslib/tbx.h>

#include "test_helpers.hpp"
#include "variant.hpp"
#include "graph.hpp"

namespace fs = std::filesystem;
using namespace npsv3;

class GraphContructionTest : public ::testing::Test {
protected:
  void SetUp() override {
    if (!fs::exists(kHG38FastaPath)) {
      GTEST_SKIP() << "Reference FASTA not available";
    }
  }

  static const fs::path kHG38FastaPath;
};
const fs::path GraphContructionTest::kHG38FastaPath("/storage/mlinderman/projects/sv/npsv3-experiments/resources/Homo_sapiens_assembly38.fasta");

TEST_F(GraphContructionTest, OverlappingVariants) {
  test::TestVCFFile vcf(R"VCF(##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##contig=<ID=chr1,length=248956422,md5=2648ae1bacce4ec4b6cf337dcae37816>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO
chr1	3693767	.	C	CCCATGCAGCCTCAGCCCCTCCTCCCGCAATCCCAGCCATGCAGCCTCAGCTCCTCCTCCCACAATCCCAGCCCTGCAGCCTCAGCTCCTCCTCCCACAATCCCAGCCCTGCAGCCTCAGCCCCTCCTCCCACAATCCCACCCATGCAGCCTCAGCCCCTCCTCCCGCAATCCCAGCCCTGCAGCCTCAGCCCCTCCTCCCGCAATCCCAG	30	.	.
chr1	3693767	.	C	CCCATGCAGCCTCAGCCCCTCCTCCCGCAATCCCAGCCATGCAGCCTCAGCTCCTCCTCCCACAATCCCAGCCCTGCAGCCTCAGCTCCTCCTCCCACAATCCCAGCCCTGCAGCCTCAGCCCCTCCTCCCACAATCCCACCCATGCAGCCTCAGCCCCTCCTCCCGCAATCCCAGCCCTGCAGCCTCAGCCCCTCCTCCCGCAATCCCAGCCATGCAGCCTCAGCCCCTCCTCCCGCAATCCCAG	30	.	.
chr1	3693767	.	C	G	30	.	.
chr1	3693767	.	C	G,CCCATGCAGCCTCAGCCCCTCCTCCCGCAATCCCAGCCATGCAGCCTCAGCTCCTCCTCCCACAATCCCAGCCCTGCAGCCTCAGCTCCTCCTCCCACAATCCCAGCCCTGCAGCCTCAGCCCCTCCTCCCACAATCCCACCCATGCAGCCTCAGCCCCTCCTCCCGCAATCCCAGCCCTGCAGCCTCAGCCCCTCCTCCCGCAATCCCAG	30	.	.)VCF");
  ASSERT_NO_THROW({
    Graph graph(kHG38FastaPath, vcf.file_path_, Range("chr1", 3693757, 3693777));
    // Although there 4 variants there are only 2 unique REF and 3 unique ALT alleles. With padding nodes, we expect:
    ASSERT_EQ(graph.NodeCount(), 7);

    // Variants 1(1) and 4(2), and 3(1) and 4(1) should share alternate allele nodes
    auto handles1_1 = graph.PathHandles("_alt_61c28d7874e1d96ba8786aa9797877126814f239_1");
    auto handles3_1 = graph.PathHandles("_alt_c1eb22337a811ed8352c5d2a7bd2bf084f1c04fe_1");
    auto handles4_1 = graph.PathHandles("_alt_d2e93459c14c5d5b81b0d5bfe57ab76e1a8b94f5_1");
    auto handles4_2 = graph.PathHandles("_alt_d2e93459c14c5d5b81b0d5bfe57ab76e1a8b94f5_2");

    ASSERT_EQ(handles4_1.size(), 2) << "Variant 4 ALT allele should have 2 nodes";
    ASSERT_EQ(handles4_2.size(), 2) << "Variant 4 ALT allele should have 2 nodes";
    
    std::sort(handles1_1.begin(), handles1_1.end());
    std::sort(handles3_1.begin(), handles3_1.end());
    std::sort(handles4_1.begin(), handles4_1.end());
    std::sort(handles4_2.begin(), handles4_2.end());
    ASSERT_TRUE(std::includes(handles4_1.begin(), handles4_1.end(), handles3_1.begin(), handles3_1.end()));
    ASSERT_TRUE(std::includes(handles4_2.begin(), handles4_2.end(), handles1_1.begin(), handles1_1.end())); 
  });
}