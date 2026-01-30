#include <filesystem>

#include "test_helpers.hpp"

namespace fs = std::filesystem;

namespace npsv3 {
namespace test {

const std::vector<fs::path> kB37FastaPaths = { fs::path("/data/human_g1k_v37.fasta"), fs::path("/storage/mlinderman/projects/sv/npsv3-experiments/resources/human_g1k_v37.fasta") };
const std::vector<fs::path> kHG38FastaPaths = { fs::path("/data/Homo_sapiens_assembly38.fasta"), fs::path("/storage/mlinderman/projects/sv/npsv3-experiments/resources/Homo_sapiens_assembly38.fasta") };

} // namespace test
} // namespace npsv3
