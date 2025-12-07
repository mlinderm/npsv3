#include <gtest/gtest.h>

#include "range.hpp"

using namespace npsv3;

TEST(RangeTest, ExpandRange) {
  Range r("chr1", 100, 200);
  auto expanded = r.Expand(50, 75);
  EXPECT_EQ(expanded.contig(), ContigName("chr1"));
  EXPECT_EQ(expanded.start(), 50u);
  EXPECT_EQ(expanded.end(), 275u);
}

TEST(RangeTest, ExpandUnderflow) {
  Range r("chrX", 10, 20);
  auto expanded = r.Expand(15, 5);  // left_flank > start -> new_start should be 0
  EXPECT_EQ(expanded.contig(), ContigName("chrX"));
  EXPECT_EQ(expanded.start(), 0u);
  EXPECT_EQ(expanded.end(), 25u);
}


TEST(RangeTest, UnionAndUnionWithSameContig) {
  Range a("chr1", 100, 200);
  Range b("chr1", 50, 250);
  auto u = a.Union(b);
  EXPECT_EQ(u.contig(), ContigName("chr1"));
  EXPECT_EQ(u.start(), 50u);
  EXPECT_EQ(u.end(), 250u);

  Range c("chr1", 120, 180);
  Range d = c;  // copy for testing UnionWith
  d.UnionWith(Range("chr1", 90, 190));
  EXPECT_EQ(d.start(), 90u);
  EXPECT_EQ(d.end(), 190u);
}

TEST(RangeTest, UnionThrowsOnDifferentContigs) {
  Range a("chr1", 0, 10);
  Range b("chr2", 0, 10);
  EXPECT_THROW(a.Union(b), std::runtime_error);
  EXPECT_THROW(a.UnionWith(b), std::runtime_error);
}

TEST(RangeTest, OverlapsBasicScenarios) {
  Range a("chr1", 100, 200);
  Range b("chr1", 150, 250);
  EXPECT_TRUE(a.Overlaps(b));
  EXPECT_TRUE(b.Overlaps(a));

  // Touching at boundary should not count as overlap
  Range c("chr1", 200, 300);
  EXPECT_FALSE(a.Overlaps(c));
  EXPECT_FALSE(c.Overlaps(a));

  // Completely separate
  Range d("chr1", 300, 400);
  EXPECT_FALSE(a.Overlaps(d));

  // Different contigs never overlap
  Range e("chr2", 150, 250);
  EXPECT_FALSE(a.Overlaps(e));
}

TEST(RangeTest, OverlapsWithContainmentAndZeroLength) {
  Range outer("chr1", 0, 100);
  Range inner("chr1", 10, 20);
  EXPECT_TRUE(outer.Overlaps(inner));
  EXPECT_TRUE(inner.Overlaps(outer));

  // Identical ranges overlap
  Range same("chr1", 0, 100);
  EXPECT_TRUE(outer.Overlaps(same));

  // Zero-length interval should not overlap unless both have positive overlap range
  Range zero("chr1", 100, 100);
  Range touch("chr1", 99, 100);
  EXPECT_FALSE(zero.Overlaps(touch));
  EXPECT_FALSE(touch.Overlaps(zero));
}

TEST(RangeTest, ContainmentOperatorLe) {
  Range outer("chr1", 0, 100);
  Range inner("chr1", 20, 80);
  EXPECT_TRUE(inner <= outer);   // inner is inside outer
  EXPECT_FALSE(outer <= inner);  // outer is not inside inner
  EXPECT_TRUE(inner <= inner);   // reflexive
}

TEST(RangeTest, StartPointComparisonOperatorLt) {
  Range r("chr1", 50, 150);
  EXPECT_TRUE(r < 100);   // start (50) < 100
  EXPECT_FALSE(r < 50);   // start (50) < 50 is false
  EXPECT_TRUE(r < 200);
}