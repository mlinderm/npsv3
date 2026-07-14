#include <fmt/format.h>
#include <gtest/gtest.h>
#include <odgi.hpp>
#include <algorithms/kmer.hpp>
#include <queue>
#include <random>
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

TEST(UniqueKmersTest, WithinNodeRepeat) {
  test::TestFastaFile fasta(R"FASTA(>chr1
AAAAAAAAA)FASTA");
  
  test::TestVCFFile vcf(R"VCF(##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##contig=<ID=chr1>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	Sample1
chr1	5	.	A	AGATTACATTGATTACA	.	PASS	.	GT	0/1
)VCF");
  auto region = Range("chr1", 0, 10); // Range is 0-indexed, half-open
  Graph graph(fasta.file_path_, vcf.file_path_, region);

  UniqueKmersOverlay overlay(graph, 7, /*max_edges=*/5, /*exclude_universal=*/false);
  EXPECT_EQ(std::count(overlay.sequences().begin(), overlay.sequences().end(), "GATTACA"), 0u) 
    << "GATTACA is not graph unique because it appears at multiple offsets in the same node";

  EXPECT_GT(std::count(overlay.sequences().begin(), overlay.sequences().end(), "AGATTAC"), 0u) 
    << "AGATTAC is graph unique because it appears at multiple offsets in the same node";
}

TEST(UniqueKmerTest, SequentialOnSamePath) {
  test::TestFastaFile fasta(R"FASTA(>chr1
AAAAAAAAA)FASTA");
  
  test::TestVCFFile vcf(R"VCF(##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##contig=<ID=chr1>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	Sample1
chr1	4	.	A	T	.	PASS	.	GT	0/1
chr1	6	.	A	T	.	PASS	.	GT	0/1
)VCF");

  auto region = Range("chr1", 0, 10); // Range is 0-indexed, half-open
  Graph graph(fasta.file_path_, vcf.file_path_, region);

  UniqueKmersOverlay overlay(graph, /*k=*/2, /*max_edges=*/5, /*exclude_universal=*/false);
  EXPECT_EQ(std::count(overlay.sequences().begin(), overlay.sequences().end(), "AT"), 0u) 
    << "AT is not graph unique because it can appear twice the same path (both alt alleles)";
}

TEST(UniqueKmerTest, ParallelBranchesAreUnique) {
  test::TestFastaFile fasta(R"FASTA(>chr1
AAAAAGGGGG)FASTA");
  
  test::TestVCFFile vcf(R"VCF(##fileformat=VCFv4.2
##FILTER=<ID=PASS,Description="All filters passed">
##contig=<ID=chr1>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	Sample1
chr1	5	.	A	AG	.	PASS	.	GT	0/1
)VCF");

  auto region = Range("chr1", 0, 10); // Range is 0-indexed, half-open
  Graph graph(fasta.file_path_, vcf.file_path_, region);

  UniqueKmersOverlay overlay(graph, /*k=*/4, /*max_edges=*/5, /*exclude_universal=*/false);
  EXPECT_GT(std::count(overlay.sequences().begin(), overlay.sequences().end(), "AAGG"), 0u) 
    << "AAGG is graph unique because it only appears on exclusive paths (REF and ALT)";

  UniqueKmersOverlay overlay_without_universal(graph, /*k=*/4, /*max_edges=*/5, /*exclude_universal=*/true);
  EXPECT_EQ(std::count(overlay_without_universal.sequences().begin(), overlay_without_universal.sequences().end(), "AAGG"), 0u) 
    << "AAGG is not graph unique when universal kmers are excluded because it appears on both REF and ALT paths";
}

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

TEST_F(GraphConstructionTest, UniqueKmersExcludeUniversal) {
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

  UniqueKmersOverlay all_unique_overlay(graph, k, max_edge, false /* exclude_universal */);
  std::unordered_set<std::string> all_unique(all_unique_overlay.sequences().begin(), all_unique_overlay.sequences().end());

  UniqueKmersOverlay non_universal_overlay(graph, k, max_edge, true /* exclude_universal */);
  std::unordered_set<std::string> non_universal(non_universal_overlay.sequences().begin(), non_universal_overlay.sequences().end());

  // The filtered set must be a strict subset of the unfiltered set.
  ASSERT_LT(non_universal.size(), all_unique.size()) << "exclude_universal should remove at least one k-mer";
  for (const auto& seq : non_universal) {
    EXPECT_GT(all_unique.count(seq), 0u) << "k-mer '" << seq << "' in non_universal but not in all_unique";
  }

  // "TAAAATA" is exclusively on the REF allele node — it must survive the filter.
  EXPECT_GT(non_universal.count("TAAAATA"), 0u) << "TAAAATA (REF-allele-only) should be retained";

  // "TTGGATT" appears upstream of the deletion breakpoint on both haplotypes and is thus universal in this context.
  EXPECT_EQ(non_universal.count("TTGGATT"), 0u) << "TTGGATT (backbone-only coverage) should be excluded";

  // "GATTCTA" appears in both haplotypes and thus is universal in this context
  EXPECT_EQ(non_universal.count("GATTCTA"), 0u) << "GATTCTA (backbone-only coverage) should be excluded";
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

TEST_F(GraphConstructionTest, KmersStayWithinNodes) {
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
  graph.ToGFA(std::cout);

  graph.Kmers(7 /* k */, 5 /* max_edges */,
              [&](const std::string&, const std::vector<handlegraph::handle_t>& handles, uint64_t offset) {
                if (offset >= graph.get_length(handles[0])) {
                  // Need exception to jump out of helper function
                  throw std::runtime_error("Reported offset exceeds handle length");
                }
              });
}

TEST_F(GraphConstructionTest, UniqueKmersMatchEnumeratedPaths) {
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
  graph.ToGFA(std::cout);

  // Every k-mer returned by UniqueKmers must satisfy two invariants:
  //   1. No two distinct physical start positions (first_handle, offset) of that k-mer
  //      can co-appear on any single graph traversal (haplotype).
  //   2. Every k-mer whose Kmers() occurrences all share the same physical start position
  //      (i.e., the same graph location with different path continuations) must
  //      be included.

  // All possible haplotype paths
  std::vector<std::string> expected_haplotypes {
    "TTG" "C" "GGATT" "CTATTGTTAGTAAAATAC" "CTATTGTTAG",
    "TTG" "TT" "GGATT" "CTATTGTTAGTAAAATAC" "CTATTGTTAG",
    "TTG" "C" "GGATT" "CTATTGTTAG",
    "TTG" "TT" "GGATT" "CTATTGTTAG",
  };
  ASSERT_GT(expected_haplotypes.size(), 0u) << "Expected haplotypes should not be empty";

  const size_t k = 7, max_edges = 5;

  // We use std::map to ensure k-mers are sorted for comparison, since UniqueKmersOverlay returns them in sorted order`

  // Compute k-mer counts for each haplotype from the fully-enumerated paths
  std::vector<std::map<std::string, size_t>> haplotype_kmer_sets;
  for (const auto& haplotype : expected_haplotypes) {
    std::map<std::string, size_t> kmer_map;
    for (size_t i = 0; i + k <= haplotype.size(); ++i) {
      kmer_map[haplotype.substr(i, k)]++;
    }
    haplotype_kmer_sets.push_back(std::move(kmer_map));
  }

  // The unique k-mers (with universal kmers) are all singleton k-mers in each haplotype minus any non-singleton k-mers from any haplotpe.
  std::map<std::string, size_t> unique_kmers_with_universal;
  for (const auto& ks : haplotype_kmer_sets) {
    for (const auto& [kmer, count] : ks) {
      if (count == 1)
        unique_kmers_with_universal[kmer] += 1;
    }
  }
  for (const auto& ks : haplotype_kmer_sets) {
    for (const auto& [kmer, count] : ks) {
      if (count > 1)
        unique_kmers_with_universal.erase(kmer);
    }
  }
  
  UniqueKmersOverlay all_unique_overlay(graph, k, max_edges, false /* exclude_universal */);
  const auto & sequences_with_universal = all_unique_overlay.sequences();
    
  ASSERT_EQ(sequences_with_universal.size(), unique_kmers_with_universal.size());
  EXPECT_TRUE(std::equal(
    sequences_with_universal.begin(), sequences_with_universal.end(), unique_kmers_with_universal.begin(), 
    [](const std::string& a, const std::pair<const std::string, size_t>& b) {
      return a == b.first;
    }
  )) << "UniqueKmersOverlay with universal k-mers does not match enumerated haplotype k-mers";

  std::map<std::string, size_t> unique_kmers_without_universal;
  for (const auto & [kmer, count] : unique_kmers_with_universal) {
    // A k-mer is universal if it appears in every haplotype
    if (count < haplotype_kmer_sets.size()) {
      unique_kmers_without_universal[kmer] = count;
    }
  }
  UniqueKmersOverlay all_unique_overlay_without_universal(graph, k, max_edges, true /* exclude_universal */);
  const auto & sequences_without_universal = all_unique_overlay_without_universal.sequences();

  ASSERT_EQ(sequences_without_universal.size(), unique_kmers_without_universal.size());
  EXPECT_TRUE(std::equal(
    sequences_without_universal.begin(), sequences_without_universal.end(), unique_kmers_without_universal.begin(), 
    [](const std::string& a, const std::pair<const std::string, size_t>& b) {
      return a == b.first;
    }
  )) << "UniqueKmersOverlay without universal k-mers does not match enumerated haplotype k-mers";

  for (const auto& [kmer, count] : unique_kmers_without_universal) {
    EXPECT_GT(unique_kmers_with_universal.count(kmer), 0) << "'Without universal' k-mers should be a subset of 'with universal' k-mers";
  }
}

TEST_F(GraphConstructionTest, UniqueKmersSerializationRoundtrip) {
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
  UniqueKmersOverlay original(graph, k, max_edges);

  test::TempDir dir;
  auto bin_path = (dir.path_ / "ukmer.bin").string();
  original.Save(bin_path);

  // Load into a heap-allocated UniqueKmersOverlay via placement new.
  // UniqueKmersOverlay holds a const reference so it is neither copyable nor movable;
  // a custom-deleter unique_ptr manages the lifetime cleanly.
  auto* raw = static_cast<UniqueKmersOverlay*>(::operator new(sizeof(UniqueKmersOverlay)));
  UniqueKmersOverlay::Load(raw, graph, bin_path);
  std::unique_ptr<UniqueKmersOverlay, void(*)(UniqueKmersOverlay*)> loaded_owner(
      raw, [](UniqueKmersOverlay* p) { p->~UniqueKmersOverlay(); ::operator delete(p); });
  UniqueKmersOverlay* loaded_ptr = loaded_owner.get();

  EXPECT_EQ(loaded_ptr->size(), original.size());
  EXPECT_EQ(loaded_ptr->sequences(), original.sequences());

  // Location handle and offset must match element-by-element.
  const auto& orig_locations = original.locations();
  const auto& loaded_locations = loaded_ptr->locations();
  EXPECT_EQ(loaded_locations, orig_locations) << "Locations mismatch";
}

// Deterministic pseudo-random ACGT sequence of the given length.
static std::string RandomSequence(size_t length, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> base(0, 3);
  static const char kBases[] = "ACGT";
  std::string seq(length, 'A');
  for (auto& c : seq) c = kBases[base(rng)];
  return seq;
}

TEST(MMapKMCFileTest, MatchesUpstreamCKMCFile) {
  test::TempDir dir;
  const uint32_t k = 25;

  // A repeated/overlapping sequence, so the resulting database has both count == 1 and
  // count > 1 k-mers, and is large enough to produce a real (KMC2, signature-binned) database
  // rather than a degenerate one.
  std::string seq = RandomSequence(2000, /*seed=*/42);
  seq += seq.substr(0, 500);  // Repeat a prefix to create some count > 1 k-mers

  auto fasta_path = (dir / "kmers.fa").string();
  {
    std::ofstream fasta(fasta_path);
    fasta << ">1\n" << seq << "\n";
  }

  auto kmc_prefix = (dir / "kmers").string();
  std::string cmd = fmt::format("kmc -t1 -k{} -ci1 -fa {} {} {} > /dev/null 2>&1", k, fasta_path, kmc_prefix,
                                 dir.path_.string());
  ASSERT_EQ(std::system(cmd.c_str()), 0) << "kmc command failed: " << cmd;

  // The upstream reader (CKMCFile::OpenForRA, unmodified) reads *.kmc_suf entirely into a
  // private heap buffer; MMapKMCFile overrides only that step to mmap it read-only instead.
  // Both must therefore report identical results for every query.
  CKMCFile upstream;
  ASSERT_TRUE(upstream.OpenForRA(kmc_prefix)) << "Failed to open upstream CKMCFile";

  npsv3::detail::MMapKMCFile mmapped;
  ASSERT_TRUE(mmapped.OpenForRA(kmc_prefix)) << "Failed to open MMapKMCFile";

  ASSERT_EQ(upstream.KmerLength(), mmapped.KmerLength());
  ASSERT_EQ(upstream.KmerCount(), mmapped.KmerCount());

  // Query every k-mer actually present in the sequence, plus several that should be absent.
  std::unordered_set<std::string> query_kmers;
  for (size_t i = 0; i + k <= seq.size(); ++i) {
    query_kmers.insert(seq.substr(i, k));
  }
  const size_t present_count = query_kmers.size();
  ASSERT_GT(present_count, 100u) << "Test setup should produce a nontrivial number of distinct k-mers";
  for (uint32_t seed = 1000; seed < 1020; ++seed) {
    query_kmers.insert(RandomSequence(k, seed));  // May coincidentally already be present; still a valid comparison.
  }

  for (const auto& kmer_str : query_kmers) {
    CKmerAPI kmer(k);
    ASSERT_TRUE(kmer.from_string(kmer_str));
    
    uint64_t upstream_count = 0;
    const bool upstream_found = upstream.CheckKmer(kmer, upstream_count);

    uint64_t mmapped_count = 0;
    const bool mmapped_found = mmapped.CheckKmer(kmer, mmapped_count);

    ASSERT_EQ(upstream_found, mmapped_found) << "Found mismatch for k-mer " << kmer_str;
    if (upstream_found) {
      ASSERT_EQ(upstream_count, mmapped_count) << "Count mismatch for k-mer " << kmer_str;
    }
  }

  upstream.Close();
  mmapped.Close();
}
