#include <gtest/gtest.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "kmer_evidence.hpp"

using namespace npsv3;

namespace {

// Two 12bp haplotypes sharing flanks but differing across a 4bp core (k=4):
//   ref: TTTT AAAA TTTT
//   alt: TTTT CCCC TTTT
// The shared "TTTT" k-mer (at both ends) is pruned as uninformative; every other 4-mer is unique to one
// haplotype, so a read matching one haplotype exactly should show a symmetric tent of apex k=4 on that
// haplotype's vote track and all zeros on the other.
constexpr const char* kRef = "TTTTAAAATTTT";
constexpr const char* kAlt = "TTTTCCCCTTTT";

}  // namespace

TEST(KmerEvidenceIndexTest, TentShapeOnExactAlleleMatch) {
  KmerEvidenceIndex index({kRef, kAlt}, /*k=*/4);

  // 7 unique 4-mers per haplotype (9 windows - 2 that equal the shared "TTTT" flanks).
  EXPECT_EQ(index.size(), 14u);

  auto ref_scan = index.Scan(kRef);
  const std::vector<float> expected_ref_votes = {0, 1, 2, 3, 4, 4, 4, 4, 3, 2, 1, 0};
  EXPECT_EQ(ref_scan.ref_votes, expected_ref_votes);
  EXPECT_EQ(ref_scan.alt_votes, std::vector<float>(12, 0.0f));

  auto alt_scan = index.Scan(kAlt);
  EXPECT_EQ(alt_scan.ref_votes, std::vector<float>(12, 0.0f));
  const std::vector<float> expected_alt_votes = {0, 1, 2, 3, 4, 4, 4, 4, 3, 2, 1, 0};
  EXPECT_EQ(alt_scan.alt_votes, expected_alt_votes);
}

TEST(KmerEvidenceIndexTest, SharedRegionCarriesNoVotes) {
  // A read drawn entirely from the shared flank has no discriminating k-mer anywhere.
  KmerEvidenceIndex index({kRef, kAlt}, /*k=*/4);
  auto scan = index.Scan("TTTT");
  EXPECT_EQ(scan.ref_votes, std::vector<float>(4, 0.0f));
  EXPECT_EQ(scan.alt_votes, std::vector<float>(4, 0.0f));
}

TEST(KmerEvidenceIndexTest, NBaseBreaksWindowsButDoesNotCrash) {
  KmerEvidenceIndex index({kRef, kAlt}, /*k=*/4);
  std::string read = kRef;
  read[5] = 'N';  // Inside the discriminating core; breaks every window touching it.
  auto scan = index.Scan(read);
  ASSERT_EQ(scan.ref_votes.size(), read.size());
  // No window can straddle the N, so no full run of k=4 informative bases remains around it, but
  // untouched flank windows on either side of the N should still vote (this mainly checks the scanner
  // doesn't crash or wrap incorrectly across the invalid base).
  EXPECT_EQ(scan.alt_votes, std::vector<float>(read.size(), 0.0f));
}

TEST(KmerEvidenceIndexTest, ExcludeLocalMultiDropsRepeatedKmer) {
  // "AAAA" repeats within the ref haplotype's core here; with exclude_local_multi the whole key is
  // dropped rather than retained with a multiplicity-inflated (but still allele-exclusive) mask.
  const std::string ref_with_repeat = "GGGGAAAAAAAAGGGG";  // "AAAA" occurs at offsets 4,5,6,7,8 (multi)
  const std::string alt = "GGGGCCCCCCCCGGGG";

  KmerEvidenceIndex all_index({ref_with_repeat, alt}, /*k=*/4, /*exclude_local_multi=*/false);
  KmerEvidenceIndex local_unique_index({ref_with_repeat, alt}, /*k=*/4, /*exclude_local_multi=*/true);

  EXPECT_GT(all_index.size(), local_unique_index.size());

  auto all_scan = all_index.Scan(ref_with_repeat);
  auto unique_scan = local_unique_index.Scan(ref_with_repeat);

  // "AAAA" is repeated, so local_unique's ref votes over that repeated stretch must be strictly less
  // than (never more than) the unweighted "all" index's votes at the same position.
  bool found_reduced_position = false;
  for (size_t j = 0; j < all_scan.ref_votes.size(); ++j) {
    EXPECT_LE(unique_scan.ref_votes[j], all_scan.ref_votes[j]);
    if (unique_scan.ref_votes[j] < all_scan.ref_votes[j]) found_reduced_position = true;
  }
  EXPECT_TRUE(found_reduced_position);
}

TEST(KmerEvidenceIndexTest, MultiOccurrenceHitsReportedUnderAllNotLocalUnique) {
  const std::string ref_with_repeat = "GGGGAAAAAAAAGGGG";  // "AAAA" occurs 5x within this haplotype
  const std::string alt = "GGGGCCCCCCCCGGGG";

  KmerEvidenceIndex all_index({ref_with_repeat, alt}, /*k=*/4, /*exclude_local_multi=*/false);
  auto all_scan = all_index.Scan(ref_with_repeat);
  EXPECT_GT(all_scan.multi_occurrence_hits, 0u) << "Scanning the repeat-containing haplotype under `all` "
                                                    "should hit the flagged multi-occurrence \"AAAA\" key.";

  KmerEvidenceIndex local_unique_index({ref_with_repeat, alt}, /*k=*/4, /*exclude_local_multi=*/true);
  auto unique_scan = local_unique_index.Scan(ref_with_repeat);
  EXPECT_EQ(unique_scan.multi_occurrence_hits, 0u) << "local_unique drops the key entirely, so a scan "
                                                       "against it can never report a multi-occurrence hit.";

  // A read drawn entirely from a non-repeated context should report no multi-occurrence hits either way.
  KmerEvidenceIndex index({kRef, kAlt}, /*k=*/4);
  EXPECT_EQ(index.Scan(kRef).multi_occurrence_hits, 0u);
}

TEST(KmerEvidenceIndexTest, GraphUniqueStyleConstructorMatchesHaplotypeConstructor) {
  // Build the same index two ways: directly from haplotype sequences, and from a pre-enumerated
  // kmer_sequences/masks pair transposed the way HaplotypeSamplerOverlay::kmers_on_path would produce
  // (plan constraint 2's `graph_unique` setting).
  KmerEvidenceIndex direct_index({kRef, kAlt}, /*k=*/4);

  std::vector<std::string> all_4mers;
  std::vector<uint16_t> masks;
  auto add_haplotype_kmers = [&](const std::string& seq, uint16_t allele_bit) {
    for (size_t i = 0; i + 4 <= seq.size(); ++i) {
      std::string kmer = seq.substr(i, 4);
      bool found = false;
      for (size_t j = 0; j < all_4mers.size(); ++j) {
        if (all_4mers[j] == kmer) {
          masks[j] |= allele_bit;
          found = true;
          break;
        }
      }
      if (!found) {
        all_4mers.push_back(kmer);
        masks.push_back(allele_bit);
      }
    }
  };
  add_haplotype_kmers(kRef, 1u << 0);
  add_haplotype_kmers(kAlt, 1u << 1);

  KmerEvidenceIndex transposed_index(all_4mers, masks, /*k=*/4, /*num_alleles=*/2);
  EXPECT_EQ(transposed_index.size(), direct_index.size());

  auto direct_scan = direct_index.Scan(kRef);
  auto transposed_scan = transposed_index.Scan(kRef);
  EXPECT_EQ(direct_scan.ref_votes, transposed_scan.ref_votes);
  EXPECT_EQ(direct_scan.alt_votes, transposed_scan.alt_votes);
}

TEST(KmerEvidenceIndexTest, BaseQualityGateSuppressesLowQualityWindows) {
  KmerEvidenceIndex index({kRef, kAlt}, /*k=*/4);
  std::string read = kRef;
  std::vector<uint8_t> quals(read.size(), 40);
  quals[5] = 2;  // A single low-quality base inside the discriminating core.

  auto ungated = index.Scan(read, &quals, /*min_baseq=*/0);
  auto gated = index.Scan(read, &quals, /*min_baseq=*/20);

  // Gating can only remove support, never add it.
  for (size_t j = 0; j < ungated.ref_votes.size(); ++j) {
    EXPECT_LE(gated.ref_votes[j], ungated.ref_votes[j]);
  }
  // At least the windows overlapping position 5 must have lost support.
  EXPECT_LT(gated.ref_votes[5], ungated.ref_votes[5]);
}

TEST(KmerEvidenceIndexTest, RejectsOutOfRangeK) {
  EXPECT_THROW(KmerEvidenceIndex({kRef, kAlt}, /*k=*/0), std::invalid_argument);
  EXPECT_THROW(KmerEvidenceIndex({kRef, kAlt}, /*k=*/33), std::invalid_argument);
}

TEST(KmerEvidenceIndexTest, RejectsTooManyAlleles) {
  std::vector<std::string> haplotypes(17, kRef);
  EXPECT_THROW(KmerEvidenceIndex(haplotypes, /*k=*/4), std::invalid_argument);
}

TEST(KmerEvidenceIndexTest, RejectsMismatchedKmerSequencesAndMasks) {
  std::vector<std::string> kmers = {"AAAA", "CCCC"};
  std::vector<uint16_t> masks = {1};
  EXPECT_THROW(KmerEvidenceIndex(kmers, masks, /*k=*/4, /*num_alleles=*/2), std::invalid_argument);
}

TEST(KmerEvidenceIndexTest, ShortReadProducesNoVotes) {
  KmerEvidenceIndex index({kRef, kAlt}, /*k=*/4);
  auto scan = index.Scan("TT");  // Shorter than k
  EXPECT_EQ(scan.ref_votes, std::vector<float>(2, 0.0f));
  EXPECT_EQ(scan.alt_votes, std::vector<float>(2, 0.0f));
}
