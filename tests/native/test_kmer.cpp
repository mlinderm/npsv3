#include <gtest/gtest.h>
#include <odgi.hpp>
#include <algorithms/kmer.hpp>
#include <unordered_map>

#include "test_helpers.hpp"
#include "graph.hpp"

using namespace npsv3;
using npsv3::test::GraphConstructionTest;

TEST_F(GraphConstructionTest, KmersFromSimpleDeletionSV) {
  test::TestVCFFile vcf(R"VCF(##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##contig=<ID=12,length=133851895>
##INFO=<ID=END,Number=1,Type=Integer,Description="End position of the structural variant">
##INFO=<ID=SVTYPE,Number=1,Type=String,Description="Type of SV:DEL=Deletion, CON=Contraction, INS=Insertion, DUP=Duplication, INV=Inversion">
##INFO=<ID=SVLEN,Number=.,Type=Integer,Description="Difference in length between REF and ALT alleles">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	S1
12	22129565	.	CAGGGGCATACTGTGAAGAACTTGACCTCTAATTAATAGCTAAGGCCGATCCTAAGAGAGCCAATTGTGGGAGATTGTCAGCTACTATATTCCTCATAGCTGGGTAGAAAGCCCTCTTGAAGGAAGATCTGAGCAGTACATCTTAGTGTCTGTCACAGACACACAGAGCTTGGATGACTCAAAAAAAGAAAAAGAGAAATAATTCTTCTGATTCTAAATATGTAACCCTCATTCCCTGAGGCGCAGTACTTCAAATTTAAGAACAAAGTTATAAAAACAACTAGTTAAGAAAAAAAGATCTGTAATCCTACTTACTCCTCAAGCAATATAACCCCCAGAAGTTCTTCTCGAGTAAATTTATGAATATCCAGTGGGTGTCTCACAAGAGTTCTAATAACATGCTGTTGACTACCATCGGGGATTCTACCAATTTTCCTATCTCCTAATCTAGATCACTGGATAATGTGTCTAATTGCTCCTAAGTTAAGAGTGGTAGCTATGCCAAACCATTGGCAGTTTCACTTCCCAGACACTACTCCTGAGGATGCTACATAGCCCAAGACTGAGGGTTCTGACTTCTATTCAGGGGTTCTGATGTTTTATATCCAGAGAATACAAGGCACTGAAATCAGCATTTTATCATTTTATCAATAACACAACTCATCAACATTGCTAACATTCTGTCCCTGTGTCATCAATGTCATCACTTCTAAGAGGACTCAATGTCTCATGAAGGTTATAGAACAACAGCTTTTTGAGATTTTACTTACTTTTTTGTTGCAGCTTTCTTGCTCTCAGATTGAGAATGGCTGGTCTAATTGAT	C	.	PASS	SVTYPE=DEL;SVLEN=-822;END=22130387	GT	1/1
)VCF");

  auto region = Range("12", 22129555, 22130397);
  Graph graph(B37FastaPath_, vcf.file_path_, region);
  graph.ToGFA(std::cout);
  std::unordered_map<std::string, int> counts;
  graph.Kmers(13, 5, [&counts](const odgi::kmer_t &kmer) {
    ASSERT_EQ(kmer.seq.find("*"), std::string::npos) << "kmer with '*' found: " << kmer.seq;
    counts[kmer.seq]++;
  });

  // Deletion is tandem repeat, so spanning kmers should appear multiple times
  ASSERT_GE(counts["TGGCCGCAGGGGC"], 2) << "expected kmer TGGCCGCAGGGGC count >= 2, got " << counts["TGGCCGCAGGGGC"];
}

