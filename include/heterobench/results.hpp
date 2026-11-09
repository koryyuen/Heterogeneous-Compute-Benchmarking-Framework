#pragma once

#include "heterobench/benchmark.hpp"
#include <map>
#include <string>
#include <vector>

namespace heterobench {

using Metadata = std::map<std::string, std::string>;

struct ExperimentResult {
    std::string backend;
    std::string device;
    std::size_t dimension = 0;
    std::size_t repetition = 0;
    BenchmarkConfig config;
    std::vector<TimingSample> samples;
};

Metadata collect_metadata();
// Creates PREFIX.samples.csv and PREFIX.summary.json; refuses existing files.
void write_results(const std::string& prefix, const Metadata& metadata,
                   const std::vector<ExperimentResult>& results);

} // namespace heterobench
