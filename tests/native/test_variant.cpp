#include <iostream>

#include <gtest/gtest.h>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/ostream_sink.h>

#include "test_helpers.hpp"
#include "variant.hpp"

namespace fs = std::filesystem;
using namespace npsv3;

class VariantFileReaderTest : public ::testing::Test {
protected:
  void SetUp() override {
    // https://stackoverflow.com/a/66490155
    auto ostream_logger = spdlog::get("gtest_logger");
    if (!ostream_logger) {
      auto ostream_sink = std::make_shared<spdlog::sinks::ostream_sink_st>(oss_);
      ostream_logger = std::make_shared<spdlog::logger>("gtest_logger", ostream_sink);
      ostream_logger->set_pattern(">%v<");
      ostream_logger->set_level(spdlog::level::debug);
    }
    spdlog::set_default_logger(ostream_logger);
  }

  std::ostringstream oss_;
};

TEST_F(VariantFileReaderTest, CreatesFileAndIndex) {
  test::TestVCFFile vcf(R"VCF(##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##contig=<ID=chr1,length=248956422,md5=2648ae1bacce4ec4b6cf337dcae37816>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO
chr1	3693767	.	C	CCCATGCAGCCTCAGCCCCTCCTCCCGCAATCCCAGCCATGCAGCCTCAGCTCCTCCTCCCACAATCCCAGCCCTGCAGCCTCAGCTCCTCCTCCCACAATCCCAGCCCTGCAGCCTCAGCCCCTCCTCCCACAATCCCACCCATGCAGCCTCAGCCCCTCCTCCCGCAATCCCAGCCCTGCAGCCTCAGCCCCTCCTCCCGCAATCCCAG	30	.	.
chr1	3693767	.	C	CCCATGCAGCCTCAGCCCCTCCTCCCGCAATCCCAGCCATGCAGCCTCAGCTCCTCCTCCCACAATCCCAGCCCTGCAGCCTCAGCTCCTCCTCCCACAATCCCAGCCCTGCAGCCTCAGCCCCTCCTCCCACAATCCCACCCATGCAGCCTCAGCCCCTCCTCCCGCAATCCCAGCCCTGCAGCCTCAGCCCCTCCTCCCGCAATCCCAGCCATGCAGCCTCAGCCCCTCCTCCCGCAATCCCAG	30	.	.
chr1	3693767	.	C	G	30	.	.
chr1	3693767	.	C	G,CCCATGCAGCCTCAGCCCCTCCTCCCGCAATCCCAGCCATGCAGCCTCAGCTCCTCCTCCCACAATCCCAGCCCTGCAGCCTCAGCTCCTCCTCCCACAATCCCAGCCCTGCAGCCTCAGCCCCTCCTCCCACAATCCCACCCATGCAGCCTCAGCCCCTCCTCCCGCAATCCCAGCCCTGCAGCCTCAGCCCCTCCTCCCGCAATCCCAG	30	.	.)VCF");

  // Check the VCF file and index exists
  ASSERT_TRUE(fs::exists(vcf.file_path_));
  auto index_path = vcf.file_path_; index_path += ".tbi";
  ASSERT_TRUE(fs::exists(index_path));

  // Try loading the index via HTSlib
  std::unique_ptr<tbx_t, npsv3::detail::tbx_deleter> idx(tbx_index_load(vcf.file_path_.c_str()));
  ASSERT_TRUE(idx);
}

TEST_F(VariantFileReaderTest, OpensAndIteratesVCF) {
  test::TestVCFFile vcf(R"VCF(##fileformat=VCFv4.2
##contig=<ID=chr1,length=1000>
##contig=<ID=chr2,length=1000>
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO
chr1	101	.	A	C	.	.	.
chr2	201	.	G	T	.	.	.
)VCF");

  auto reader = VariantFileReader::Open(vcf.file_path_);
  ASSERT_TRUE(reader);

  // Default behavior is to iterate through the entire file
  int count = 0;
  while (auto var = reader->NextVariant()) {
    ++count;
  }
  EXPECT_EQ(count, 2);
}

TEST_F(VariantFileReaderTest, SetRegionLimitsToContig) {
  test::TestVCFFile vcf(R"VCF(##fileformat=VCFv4.2
##contig=<ID=chr1,length=1000>
##contig=<ID=chr2,length=1000>
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO
chr1	101	.	A	C	.	.	.
chr2	201	.	G	T	.	.	.
)VCF");

  auto reader = VariantFileReader::Open(vcf.file_path_);
  ASSERT_TRUE(reader);

  Range chr2_region(ContigName("chr2"), 0, 1000);
  reader->SetRegion(chr2_region);

  auto var = reader->NextVariant();
  ASSERT_TRUE(var);
  EXPECT_EQ(var->contig(), ContigName("chr2"));
  
  EXPECT_FALSE(reader->NextVariant());
}

TEST_F(VariantFileReaderTest, OpenMissingFileThrows) {
  EXPECT_THROW(VariantFileReader::Open("/junk.vcf.gz"), std::runtime_error);
}

TEST_F(VariantFileReaderTest, SetRegionInvalidContigThrows) {
  test::TestVCFFile vcf(R"VCF(##fileformat=VCFv4.2
##contig=<ID=chr1,length=1000>
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO
chr1	101	.	A	C	.	.	.
)VCF");

  auto reader = VariantFileReader::Open(vcf.file_path_.native());
  ASSERT_TRUE(reader);

  // Use a contig name that does not exist in the file
  Range bad_region(ContigName("no_such_chr"), 0, 1000);
  EXPECT_THROW(reader->SetRegion(bad_region), std::runtime_error);
}

TEST_F(VariantFileReaderTest, MalformedRecordThrowsOnNextVariant) {
  test::TestVCFFile vcf(R"VCF(##fileformat=VCFv4.2
##contig=<ID=chr1,length=1000>
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO
chr1	101	.	A	C
)VCF");

  auto reader = VariantFileReader::Open(vcf.file_path_);
  ASSERT_TRUE(reader);
  EXPECT_THROW(reader->NextVariant(), std::runtime_error);
}

TEST_F(VariantFileReaderTest, InvalidPSTypeTriggersWarning) {
  test::TestVCFFile vcf(R"VCF(##fileformat=VCFv4.2
##contig=<ID=chr1,length=1000>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=PS,Number=1,Type=String,Description="Phase set identifier">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO
chr1	101	.	A	C	.	.	.
)VCF");
  auto reader = VariantFileReader::Open(vcf.file_path_);
  ASSERT_NE(oss_.str().find("PS format field is not of expected integer type; ignoring PS."), std::string::npos);
}

TEST_F(VariantFileReaderTest, ReferenceRegionWithStarAllele) {
  test::TestVCFFile vcf(R"VCF(##fileformat=VCFv4.2
##contig=<ID=chr1,length=248956422>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=PS,Number=1,Type=String,Description="Phase set identifier">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO
chr1	5414226	.	CG	*,C	.	PASS	.
)VCF");
  auto reader = VariantFileReader::Open(vcf.file_path_);
  auto var = reader->NextVariant();
  // Variant region should not include '*' alleles
  ASSERT_EQ(var->ReferenceRegion(), Range("chr1", 5414226, 5414227));
  ASSERT_EQ(var->AlleleReferenceRegion(0), Range("chr1", 5414226, 5414227));
  ASSERT_FALSE(var->AlleleReferenceRegion(1));
  ASSERT_EQ(var->AlleleReferenceRegion(0), Range("chr1", 5414226, 5414227));
}

TEST_F(VariantFileReaderTest, ReferenceRegionInsertion) {
  test::TestVCFFile vcf(R"VCF(##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##contig=<ID=2,length=243199373>
##INFO=<ID=SVTYPE,Number=1,Type=String,Description="Type of structural variant">
##INFO=<ID=SVLEN,Number=A,Type=Integer,Description="Difference in length between REF and ALT alleles">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	Sample
2	39714038	.	T	TCTTTCTCTCTTTCTCTCTTTCTCTCTCTCTCTCTCTTTCTCTCTCTCTCTCTCG	20	PASS	SVTYPE=INS;SVLEN=54	GT	1/1)VCF");
  auto reader = VariantFileReader::Open(vcf.file_path_);
  auto var = reader->NextVariant();
  // HTSLib 1.22 has an error in the rlen calculation https://github.com/samtools/htslib/issues/1940
  ASSERT_EQ(var->ReferenceRegion(), Range("2", 39714038, 39714038));
  ASSERT_EQ(var->AlleleReferenceRegion(0), Range("2", 39714038, 39714038));
  ASSERT_EQ(var->AlleleReferenceRegion(1), Range("2", 39714038, 39714038));
}


TEST(GenotypeTest, ExpectedGenotypeSizes) {
  auto genotype = Variant::Genotype();
  EXPECT_EQ(sizeof(genotype), 8);
}

TEST(GenotypeTest, ParsesGenotypeAndPhasing) {
  test::TestVCFFile vcf(R"VCF(##fileformat=VCFv4.2
##contig=<ID=chr1,length=1000>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=PS,Number=1,Type=Integer,Description="Phase set identifier">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	Sample1	Sample2	Sample3	Sample4	Sample5
chr1	101	.	A	C	.	.	.	GT	0/1	0|1	0/0	1/1	0|1
chr1	102	.	A	C	.	.	.	GT:PS	0/1:102	0|1:102	0/0:.	1/1:.	0|1
)VCF");

  auto reader = VariantFileReader::Open(vcf.file_path_);
  ASSERT_TRUE(reader);
  
  using Indices = Variant::Genotype::AlleleIndices;
  {
    auto variant = reader->NextVariant();
    auto genotypes = variant->Genotypes();
    ASSERT_EQ(genotypes.size(), 5);
    
    // std::vector<std::tuple<Variant::Genotype::AlleleIndices, bool, Phase::Value> >
    for (auto & [idx, indices, phase] : std::vector<std::tuple<size_t, Indices, Phase::Value>>({
      { 0, Indices({0,1}), Phase::kUnphased },
      { 1, Indices({0,1}), Phase::kGlobal },
      { 2, Indices({0,0}), Phase::kImplicit },
      { 3, Indices({1,1}), Phase::kImplicit },
      { 4, Indices({0,1}), Phase::kGlobal },
    })) {
      EXPECT_EQ(genotypes[idx].allele_indices(), indices);
      EXPECT_EQ(genotypes[idx].phase(), Phase(phase));
    }
  }

  {
    auto variant = reader->NextVariant();
    auto genotypes = variant->Genotypes();
    ASSERT_EQ(genotypes.size(), 5);
    
    // std::vector<std::tuple<Variant::Genotype::AlleleIndices, bool, Phase::Value> >
    for (auto & [idx, indices, phase, phase_set] : std::vector<std::tuple<size_t, Indices, Phase::Value, int32_t>>({
      { 0, Indices({0,1}), Phase::kUnphased, -1 },
      { 1, Indices({0,1}), Phase::kLocal, 102 },
      { 2, Indices({0,0}), Phase::kImplicit, -1 },
      { 3, Indices({1,1}), Phase::kImplicit, -1 },
      { 4, Indices({0,1}), Phase::kGlobal, -1 },
    })) {
      EXPECT_EQ(genotypes[idx].allele_indices(), indices);
      EXPECT_EQ(genotypes[idx].phase(), Phase(phase, phase_set));
    }
  }
}