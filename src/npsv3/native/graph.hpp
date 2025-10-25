#pragma once

#include <vector>
#include <odgi.hpp>

#include "range.hpp"

namespace npsv3 {
class Graph {
public:
    Graph(const std::string& reference_fasta_path, const std::string& vcf_path, const Range& region);

    size_t NodeCount() const { return graph_.get_node_count(); }
    odgi::nid_t NodeId(handlegraph::handle_t handle) const { return graph_.get_id(handle); }

    std::vector<handlegraph::handle_t> PathHandles(const std::string& path_name) const;
    
private:
    odgi::graph_t graph_;
};

namespace test {
void TestCreateGraph(const std::string& reference_fasta_path, const std::string& vcf_path);
}

} // namespace npsv3
