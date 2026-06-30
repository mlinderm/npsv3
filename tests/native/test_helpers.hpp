#pragma once

#include <cstdlib>
#include <filesystem>
#include <memory>
#include <algorithm>
#include <fstream>

#include <htslib/bgzf.h>
#include <htslib/tbx.h>
#include <htslib/faidx.h>

#include <gtest/gtest.h>

#include "variant.hpp"

namespace npsv3 {
namespace test {

namespace fs = std::filesystem;

class TempDir {
public:
  fs::path path_;

  TempDir() {
    std::string temp_template = (fs::temp_directory_path() / "test.XXXXXX").string();
    path_ = fs::path(::mkdtemp(temp_template.data()));
  }
  ~TempDir() {
    std::error_code ec;
    std::filesystem::remove_all(path_, ec);
  }

  fs::path operator/(const fs::path& rhs) {
    return path_ / rhs;
  }
};

class TestVCFFile {
public:
  explicit TestVCFFile(const std::string& contents) {
    file_path_ = dir_ / "variant.vcf.gz";
    
    { // Write BGZF-compressed VCF (closing when done or on error)
      std::unique_ptr<BGZF, npsv3::detail::bgzf_deleter> fp(bgzf_open(file_path_.c_str(), "w"));
      if (!fp) {
        throw std::runtime_error("failed to open bgzf for writing: " + file_path_.native());
      }

      int wrote = bgzf_write(fp.get(), contents.data(), contents.size());
      if (wrote < 0) {
        throw std::runtime_error("bgzf_write failed");
      }
    }

    // Build tabix index (.tbi)
    if (tbx_index_build(file_path_.c_str(), 0, &tbx_conf_vcf) != 0) {
      throw std::runtime_error("failed to build tabix index for: " + file_path_.native());
    }
  }

  TempDir dir_;
  fs::path file_path_;
};

class TestFastaFile {
public:
  explicit TestFastaFile(const std::string& contents) {
    file_path_ = dir_ / "reference.fasta";
    std::ofstream ofs(file_path_);
    if (!ofs) {
      throw std::runtime_error("failed to open fasta for writing: " + file_path_.native());
    }
    ofs << contents;
    ofs.close();
  
    // Build index (.fai)
    if (fai_build(file_path_.c_str()) != 0) {
      throw std::runtime_error("failed to build fasta index for: " + file_path_.native());
    }
  }

  TempDir dir_;
  fs::path file_path_;
};


// Shared FASTA path constants for native tests.
extern const std::vector<fs::path> kB37FastaPaths; 
extern const std::vector<fs::path> kHG38FastaPaths;

class GraphConstructionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    {
      auto it = std::find_if(kB37FastaPaths.begin(), kB37FastaPaths.end(), [](const fs::path& path) { return fs::exists(path); });
      if (it == kB37FastaPaths.end())
        GTEST_SKIP() << "B37 Reference FASTA is not available";
      else
        B37FastaPath_ = *it;
    }

    {
      auto it = std::find_if(kHG38FastaPaths.begin(), kHG38FastaPaths.end(), [](const fs::path& path) { return fs::exists(path); });
      if (it == kHG38FastaPaths.end())
        GTEST_SKIP() << "HG38 Reference FASTA is not available";
      else
        HG38FastaPath_ = *it;
    }
  }

  fs::path B37FastaPath_;
  fs::path HG38FastaPath_;
};

}  // namespace test
}  // namespace npsv3