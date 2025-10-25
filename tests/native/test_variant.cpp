#include <gtest/gtest.h>

#include "test_helpers.hpp"
#include "variant.hpp"

namespace fs = std::filesystem;
using namespace npsv3;

TEST(VariantFileReaderTest, CreatesFileAndIndex) {
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
  std::unique_ptr<tbx_t, npsv3::internal::tbx_deleter> idx(tbx_index_load(vcf.file_path_.c_str()));
  ASSERT_TRUE(idx);
}

TEST(VariantFileReaderTest, OpensAndIteratesVCF) {
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

TEST(VariantFileReaderTest, SetRegionLimitsToContig) {
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
  EXPECT_EQ(var->Contig(), ContigName("chr2"));
  
  EXPECT_FALSE(reader->NextVariant());
}

TEST(VariantFileReaderTest, OpenMissingFileThrows) {
  EXPECT_THROW(VariantFileReader::Open("/junk.vcf.gz"), std::runtime_error);
}

TEST(VariantFileReaderTest, SetRegionInvalidContigThrows) {
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

TEST(VariantFileReaderTest, MalformedRecordThrowsOnNextVariant) {
  test::TestVCFFile vcf(R"VCF(##fileformat=VCFv4.2
##contig=<ID=chr1,length=1000>
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO
chr1	101
)VCF");

  auto reader = VariantFileReader::Open(vcf.file_path_);
  ASSERT_TRUE(reader);
  EXPECT_THROW(reader->NextVariant(), std::runtime_error);
}