#include <fmt/format.h>
#include <gtest/gtest.h>
#include <odgi.hpp>
#include <algorithms/kmer.hpp>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>

#include "test_helpers.hpp"
#include "graph.hpp"
#include "kmer.hpp"

using namespace npsv3;
using npsv3::test::GraphConstructionTest;
namespace fs = std::filesystem;

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

TEST_F(GraphConstructionTest, KmersRespectsMaxEdges) {
  test::TestVCFFile vcf(R"VCF(##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##contig=<ID=chr1,length=248956422,md5=2648ae1bacce4ec4b6cf337dcae37816>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	Sample1
chr1	52277191	.	TCTATTGTTAGTAAAATAC	T	.	PASS	.	GT	0/1
)VCF");

  auto region = Range("chr1", 52277181, 52277219);
  Graph graph(HG38FastaPath_, vcf.file_path_, region);

  // Test with max_edges = 2: should not traverse more than 2 edges from any starting handle
  size_t max_edges = 2;
  std::vector<size_t> handle_counts;

  graph.Kmers(7, max_edges, [&]([[maybe_unused]] const std::string& kmer, const std::vector<handlegraph::handle_t>& handles, [[maybe_unused]] uint64_t offset) {
    handle_counts.push_back(handles.size());
    // The number of handles represents: 1 starting handle + number of edges traversed
    // So handles.size() - 1 = number of edges traversed
    ASSERT_LE(handles.size() - 1, max_edges) << "K-mer spans " << (handles.size() - 1) << " edges, exceeds max_edges=" << max_edges;
  });

  // Verify we got some k-mers
  ASSERT_GT(handle_counts.size(), 0) << "Should have generated at least one k-mer";

  // Test with max_edges = 0: should only generate k-mers within single handles
  graph.Kmers(7, 0, [&]([[maybe_unused]] const std::string& kmer, const std::vector<handlegraph::handle_t>& handles, [[maybe_unused]] uint64_t offset) {
    ASSERT_EQ(handles.size(), 1) << "With max_edges=0, all k-mers should be within single handles";
  });
}

// TEST_F(GraphConstructionTest, UniqueKmersAreGraphUnique) {
//   test::TestVCFFile vcf(R"VCF(##fileformat=VCFv4.2
// ##FILTER=<ID=PASS,Description="All filters passed">
// ##contig=<ID=chr1,length=248956422,md5=2648ae1bacce4ec4b6cf337dcae37816>
// ##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
// #CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	Sample1
// chr1	52277191	.	TCTATTGTTAGTAAAATAC	T	.	PASS	.	GT	0/1
// )VCF");
//   auto region = Range("chr1", 52277181, 52277219);
//   Graph graph(HG38FastaPath_, vcf.file_path_, region);

//   // Every k-mer returned by UniqueKmers must satisfy two invariants:
//   //   1. All Kmers() occurrences of that sequence share at least one common node.
//   //   2. Every k-mer with only a single Kmers() occurrence is included.

//   const size_t k = 7, max_edge = 5;
//   using Occurrence = std::vector<handlegraph::handle_t>;

//   // Collect all Kmers occurrences grouped by sequence.
//   std::unordered_map<std::string, std::vector<Occurrence>> all_occurrences;
//   graph.Kmers(k, max_edge,
//               [&](const std::string& seq, const std::vector<handlegraph::handle_t>& handles, uint64_t offset) {
//                 all_occurrences[seq].emplace_back(handles);
//               });

//   // Collect UniqueKmers output, checking there are no duplicates.
//   std::unordered_set<std::string> unique_kmers;
//   graph.UniqueKmers(k, max_edge, [&](const std::string& seq, const std::vector<handlegraph::handle_t>&, uint64_t) {
//     EXPECT_EQ(unique_kmers.count(seq), 0u) << "UniqueKmers reported duplicate: " << seq;
//     unique_kmers.insert(seq);
//   });
//   ASSERT_FALSE(unique_kmers.empty()) << "UniqueKmers should return at least one k-mer for this graph";

//   // Invariant 1: Every UniqueKmer must appear in Kmers output and all its occurrences
//   // must share a common node. This is weaker than the definition of graph-uniqueness, but
//   // it's a necessary condition that is easier to check.
//   for (const auto& seq : unique_kmers) {
//     ASSERT_GT(all_occurrences.count(seq), 0u) << "UniqueKmer '" << seq << "' was not also generated by Kmers";
    
//     auto & occurrences = all_occurrences[seq];
//     if (occurrences.size() <= 1) continue;  // k-mers with a single occurrence are trivially graph-unique
    
//     std::unordered_map<handlegraph::handle_t, int> handle_counts;
//     for (const auto& occurrence : occurrences) {
//       for (const auto& handle : occurrence) {
//         handle_counts[handle]++;
//       }
//     }
//     ASSERT_NE(std::find_if(handle_counts.begin(), handle_counts.end(),
//                            [&](const auto& pair) { return pair.second == occurrences.size(); }),
//               handle_counts.end())
//         << "UniqueKmer '" << seq << "' has occurrences with no common node";
//   }

//   // Invariant 2: Every Kmers k-mer with exactly one  occurrence must be included.
//   for (const auto& [seq, occ] : all_occurrences) {
//     if (occ.size() == 1) {
//       EXPECT_GT(unique_kmers.count(seq), 0u) << "Single-occurrence k-mer '" << seq << "' is missing from UniqueKmers";
//     }
//   }

//   // "GATTCTA" appears twice in Kmers (once per haplotype path at the deletion
//   // breakpoint) but both occurrences start in the same upstream node at the
//   // same offset, and so should be unique.
//   ASSERT_TRUE(all_occurrences["GATTCTA"].size() == 2 && unique_kmers.find("GATTCTA") != unique_kmers.end())
//       << "Expected GATTCTA to be included in UniqueKmers";

//   // "CTATTGT" appears in both the deletion and downstream, so should not be unique.
//   ASSERT_TRUE(all_occurrences["CTATTGT"].size() == 2 && unique_kmers.find("CTATTGT") == unique_kmers.end())
//       << "Expected CTATTGT to not be included in UniqueKmers";

//   // "TAAAATA" appears only the REF allele node, so should be unique.
//   ASSERT_TRUE(all_occurrences["TAAAATA"].size() == 1 && unique_kmers.find("TAAAATA") != unique_kmers.end())
//       << "Expected TAAAATA to be included in UniqueKmers";
// }

// TEST_F(GraphConstructionTest, UniqueKmersExcludeUniversal) {
//   test::TestVCFFile vcf(R"VCF(##fileformat=VCFv4.2
// ##FILTER=<ID=PASS,Description="All filters passed">
// ##contig=<ID=chr1,length=248956422,md5=2648ae1bacce4ec4b6cf337dcae37816>
// ##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
// #CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	Sample1
// chr1	52277191	.	TCTATTGTTAGTAAAATAC	T	.	PASS	.	GT	0/1
// )VCF");
//   auto region = Range("chr1", 52277181, 52277219);
//   Graph graph(HG38FastaPath_, vcf.file_path_, region);

//   const size_t k = 7, max_edge = 5;

//   std::unordered_set<std::string> all_unique, non_universal;
//   graph.UniqueKmers(k, max_edge, [&](const std::string& seq, const std::vector<handlegraph::handle_t>&, uint64_t) {
//     all_unique.insert(seq);
//   });
//   graph.UniqueKmers(k, max_edge, [&](const std::string& seq, const std::vector<handlegraph::handle_t>&, uint64_t) {
//     non_universal.insert(seq);
//   }, true /* exclude universal k-mers */);

//   // The filtered set must be a strict subset of the unfiltered set.
//   ASSERT_LT(non_universal.size(), all_unique.size()) << "exclude_universal should remove at least one k-mer";
//   for (const auto& seq : non_universal) {
//     EXPECT_GT(all_unique.count(seq), 0u) << "k-mer '" << seq << "' in non_universal but not in all_unique";
//   }

//   // "TAAAATA" is exclusively on the REF allele node — it must survive the filter.
//   EXPECT_GT(non_universal.count("TAAAATA"), 0u) << "TAAAATA (REF-allele-only) should be retained";

//   // "GATTCTA" appears in both haplotypes and thus is universal in this context
//   EXPECT_EQ(non_universal.count("GATTCTA"), 0u) << "GATTCTA (backbone-only coverage) should be excluded";
// }



// // Helper: write a FASTA file and run kmc to produce a KMC database.
// // Returns the database path prefix (without extension) or "" if kmc is absent.
// static std::string BuildKmcDatabase(const fs::path& dir,
//                                     const std::string& fasta_contents,
//                                     int k, int min_count = 1) {
//   const fs::path fasta = dir / "seqs.fa";
//   const fs::path db_prefix = dir / "kmers";
//   const fs::path tmp_dir = dir / "kmc_tmp";
//   fs::create_directories(tmp_dir);

//   { std::ofstream f(fasta); f << fasta_contents; }

//   std::string cmd = fmt::format(
//       "kmc -k{} -ci{} -fm -cs65535 -b {} {} {} >/dev/null 2>&1",
//       k, min_count, fasta.string(), db_prefix.string(), tmp_dir.string());
//   if (std::system(cmd.c_str()) != 0) {
//     throw std::runtime_error("kmc command failed: " + cmd);
//   }
//   return db_prefix.string();
// }

// class KmerCountsTest : public ::testing::Test {
//  protected:
//   void SetUp() override {
//     if (std::system("kmc --version >/dev/null 2>&1") != 0) {
//       GTEST_SKIP() << "kmc binary not available";
//     }
//   }
//   npsv3::test::TempDir dir_;
// };

// // Build a small database with known counts and verify ABSENT/HET/HOM classification.
// TEST_F(KmerCountsTest, ClassifiesKnownCounts) {
//   // We want a 7-mer that appears:
//   //   once  → count 1 (ABSENT at 30x with default absent_fraction=0.1 → threshold=3)
//   //   ~15x  → HETEROZYGOUS
//   //   ~30x  → HOMOZYGOUS
//   std::string fasta;
//   fasta += ">absent\nAAAAAAA\n";
//   for (int i = 0; i < 15; ++i) fasta += ">het_" + std::to_string(i) + "\nACGTACG\n";
//   for (int i = 0; i < 30; ++i) fasta += ">hom_" + std::to_string(i) + "\nACGACGA\n";

//   const int k = 7;
//   auto db_path = BuildKmcDatabase(dir_.path_, fasta, k, /*min_count=*/1);

//   // Provide coverage explicitly so we don't need auto-estimation.
//   npsv3::KmerCounts::Params params;
//   params.coverage = 30.0; // k-mer coverage

//   npsv3::KmerCounts counts(db_path, params);
//   EXPECT_DOUBLE_EQ(counts.coverage(), 30.0);

//   // AAAAAAA: count=1 < 3 (0.1 * 30) → ABSENT
//   EXPECT_EQ(counts.Classify("AAAAAAA"), npsv3::KmerZygosity::ABSENT);
//   // ACGTACG: count=15 in [3, 21) → HETEROZYGOUS
//   EXPECT_EQ(counts.Classify("ACGTACG"), npsv3::KmerZygosity::HETEROZYGOUS);
//   // ACGACGA: count=30 [21, 75) → HOMOZYGOUS
//   EXPECT_EQ(counts.Classify("ACGACGA"), npsv3::KmerZygosity::HOMOZYGOUS);
// }
