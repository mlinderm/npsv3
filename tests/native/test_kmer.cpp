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
##contig=<ID=chr1,length=248956422,md5=2648ae1bacce4ec4b6cf337dcae37816>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	Sample1
chr1	52277191	.	TCTATTGTTAGTAAAATAC	T	.	PASS	.	GT	0/1
)VCF");

  auto region = Range("chr1", 52277181, 52277219);
  Graph graph(HG38FastaPath_, vcf.file_path_, region);
  graph.ToGFA(std::cout);
  
  std::unordered_map<std::string, int> kmer_counts;
  std::vector<std::tuple<std::string, std::vector<handlegraph::handle_t>, uint64_t>> all_kmers;

  graph.Kmers(7, 5, [&](const std::string& kmer, const std::vector<handlegraph::handle_t>& handles, uint64_t offset) {
    ASSERT_EQ(kmer.find("*"), std::string::npos) << "kmer with '*' found: " << kmer;
    ASSERT_EQ(kmer.find("N"), std::string::npos) << "kmer with 'N' found: " << kmer;
    ASSERT_EQ(kmer.length(), 7) << "kmer length should be 7, got: " << kmer.length();
    ASSERT_FALSE(handles.empty()) << "handles vector should not be empty";

    kmer_counts[kmer]++;
    all_kmers.emplace_back(kmer, handles, offset);
  });

  // Verify we gor the correct number of k-mers (there should 38-k+1 from the REF path and k-1 from the ALT path)
  ASSERT_EQ(all_kmers.size(), 38) << "Should have generated 38 k-mers across both haplotypes";

  // Verify each k-mer has valid position information
  for (const auto& [kmer, handles, offset] : all_kmers) {
    // Verify we can reconstruct the k-mer from the graph using the handle and offset
    const auto& first_handle = handles[0];
    auto seq = graph.get_sequence(first_handle);

    if (handles.size() == 1) {
      // Single-handle k-mer: should be able to extract it directly
      ASSERT_LE(offset + 7, seq.length()) << "Offset " << offset << " + k-mer length should not exceed handle sequence length";
      ASSERT_EQ(seq.substr(offset, 7), kmer) << "K-mer should match sequence at offset " << offset << " in handle " << graph.get_id(first_handle);
    } else {
      // Multi-handle k-mer: starts at offset in first handle
      ASSERT_LT(offset, seq.length()) << "Offset should be within first handle sequence";
    }
  }
  // There should be 2 copies of the k-mer spanning the deletion breakpoint
  ASSERT_EQ(kmer_counts["GATTCTA"], 2) << "Expected k-mer GATTCTA count of 2, got " << kmer_counts["GATTCTA"];
}

// TEST_F(GraphConstructionTest, KmersFromSimpleDeletionSV) {
//   test::TestVCFFile vcf(R"VCF(##fileformat=VCFv4.2
// ##FILTER=<ID=PASS,Description="All filters passed">
// ##contig=<ID=12,length=133851895>
// ##INFO=<ID=END,Number=1,Type=Integer,Description="End position of the structural variant">
// ##INFO=<ID=SVTYPE,Number=1,Type=String,Description="Type of SV:DEL=Deletion, CON=Contraction, INS=Insertion, DUP=Duplication, INV=Inversion">
// ##INFO=<ID=SVLEN,Number=.,Type=Integer,Description="Difference in length between REF and ALT alleles">
// ##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
// #CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	S1
// 12	22129565	.	CAGGGGCATACTGTGAAGAACTTGACCTCTAATTAATAGCTAAGGCCGATCCTAAGAGAGCCAATTGTGGGAGATTGTCAGCTACTATATTCCTCATAGCTGGGTAGAAAGCCCTCTTGAAGGAAGATCTGAGCAGTACATCTTAGTGTCTGTCACAGACACACAGAGCTTGGATGACTCAAAAAAAGAAAAAGAGAAATAATTCTTCTGATTCTAAATATGTAACCCTCATTCCCTGAGGCGCAGTACTTCAAATTTAAGAACAAAGTTATAAAAACAACTAGTTAAGAAAAAAAGATCTGTAATCCTACTTACTCCTCAAGCAATATAACCCCCAGAAGTTCTTCTCGAGTAAATTTATGAATATCCAGTGGGTGTCTCACAAGAGTTCTAATAACATGCTGTTGACTACCATCGGGGATTCTACCAATTTTCCTATCTCCTAATCTAGATCACTGGATAATGTGTCTAATTGCTCCTAAGTTAAGAGTGGTAGCTATGCCAAACCATTGGCAGTTTCACTTCCCAGACACTACTCCTGAGGATGCTACATAGCCCAAGACTGAGGGTTCTGACTTCTATTCAGGGGTTCTGATGTTTTATATCCAGAGAATACAAGGCACTGAAATCAGCATTTTATCATTTTATCAATAACACAACTCATCAACATTGCTAACATTCTGTCCCTGTGTCATCAATGTCATCACTTCTAAGAGGACTCAATGTCTCATGAAGGTTATAGAACAACAGCTTTTTGAGATTTTACTTACTTTTTTGTTGCAGCTTTCTTGCTCTCAGATTGAGAATGGCTGGTCTAATTGAT	C	.	PASS	SVTYPE=DEL;SVLEN=-822;END=22130387	GT	1/1
// )VCF");

//   auto region = Range("12", 22129555, 22130397);
//   Graph graph(B37FastaPath_, vcf.file_path_, region);
//   graph.ToGFA(std::cout);
//   std::unordered_map<std::string, int> counts;
//   graph.Kmers(13, 5, [&counts](const odgi::kmer_t &kmer) {
//     ASSERT_EQ(kmer.seq.find("*"), std::string::npos) << "kmer with '*' found: " << kmer.seq;
//     counts[kmer.seq]++;
//   });

//   // Deletion is tandem repeat, so spanning kmers should appear multiple times
//   ASSERT_GE(counts["TGGCCGCAGGGGC"], 2) << "expected kmer TGGCCGCAGGGGC count >= 2, got " << counts["TGGCCGCAGGGGC"];
// }


