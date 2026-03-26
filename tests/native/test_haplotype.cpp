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
  KmerZygosity Classify(const std::string&) const override { return z_; }
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

  const size_t k = 7, max_edge = 5;
  ConstantKmerCounts counts;
  HaplotypeSamplerOverlay sampler(graph, k, max_edge, counts);

  EXPECT_GT(sampler.NumKmers(), 0u);

  EXPECT_TRUE(sampler.HasNodeKmers()) << "Graph has 18bp deletion, so should have some node k-mers";
  EXPECT_TRUE(sampler.HasEdgeKmers()) << "Graph should have edge k-mers crossing the breakpoints";

  auto paths = sampler.FindBestPaths(4 /* more than the expected number of paths */);
  ASSERT_EQ(paths.size(), 2u)
      << "For a single bi-allelic variant, there should be exactly 2 distinct paths (ref and alt)";
  
  const auto & best_path = paths[0];
  ASSERT_FALSE(best_path.empty());
  EXPECT_EQ(best_path.front(), graph.min_node_id()); // Path must span the whole graph.
  EXPECT_EQ(best_path.back(),  graph.max_node_id());

  // Node IDs are in topological (ascending) order.
  EXPECT_TRUE(std::is_sorted(best_path.begin(), best_path.end()));

  // With all k-mers scored as HOMOZYGOUS, the path that maximizes k-mer coverage is the longer reference path.
  EXPECT_EQ(best_path, graph.PathNodes("chr1"));
  EXPECT_NE(paths[1], graph.PathNodes("chr1"));
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
  graph.ToGFA(std::cout);

  const size_t k = 7, max_edge = 5;

  {  // Still samples paths, even when all k-mers are ABSENT
    ConstantKmerCounts counts(KmerZygosity::ABSENT);
    HaplotypeSamplerOverlay sampler(graph, k, max_edge, counts);
    auto haplotypes = sampler.SampleHaplotypes(2);
    EXPECT_EQ(haplotypes.size(), 2u) << "Sampler should return exactly n paths";
  }
};

TEST_F(GraphConstructionTest, HaplotypeSamplerFromBAMFindsHomAltForDeletion) {
  if (std::system("kmc --version >/dev/null 2>&1") != 0) {
    GTEST_SKIP() << "kmc binary not available";
  }

  const fs::path bam_path = fs::path(TEST_DATA_DIR) / "12_22127565_22132387.bam";
  const fs::path vcf_path = fs::path(TEST_DATA_DIR) / "12_22129565_22130387.vcf.gz";

  // Build a KMC k-mer database from the BAM file.
  test::TempDir tmp;
  const fs::path db_prefix = tmp / "kmers";
  const fs::path kmc_tmp = tmp / "kmc_tmp";
  fs::create_directories(kmc_tmp);

  const size_t k = 31, max_edge = 5;
  std::string cmd =
      fmt::format("kmc -k{} -ci1 -fbam -cs65535 {} {} {} >/dev/null 2>&1", k, bam_path, db_prefix, kmc_tmp);
  ASSERT_EQ(std::system(cmd.c_str()), 0) << "kmc failed: " << cmd;

  // Build graph around the deletion (region matches the BAM extent).
  auto region = Range("12", 22127565, 22132387);
  Graph graph(B37FastaPath_, vcf_path.string(), region);
  graph.ToGFA(std::cout);

  // The BAM is a small slice (~17x haploid k-mer depth); provide coverage explicitly
  // to avoid auto-estimation failure on an atypical histogram.
  KmerCounts::Params kmer_params { 17.0 };
  KmerCounts counts(db_prefix.string(), kmer_params);

 
  HaplotypeSamplerOverlay sampler(graph, k, max_edge, counts);

  auto haplotypes = sampler.SampleHaplotypes(2);
  ASSERT_EQ(haplotypes.size(), 2u);

  // The sample is homozygous for the deletion, so we should sample the non-reference path first,
  // then the reference path.
  auto ref_path = graph.PathNodes("12");
  EXPECT_NE(haplotypes[0], ref_path) << "First haplotype should not be the reference path";
  EXPECT_EQ(haplotypes[1], ref_path) << "Second haplotype should be the reference path";
}

