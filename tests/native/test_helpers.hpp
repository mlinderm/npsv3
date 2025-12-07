#pragma once

#include <cstdlib>
#include <filesystem>
#include <memory>

#include <htslib/bgzf.h>
#include <htslib/tbx.h>

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

}  // namespace test
}  // namespace npsv3