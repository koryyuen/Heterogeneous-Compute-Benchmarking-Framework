#pragma once

#include "heterobench/backend.hpp"
#include <cstddef>
#include <optional>
#include <vector>

namespace heterobench {

struct BenchmarkConfig {
    std::size_t warmup_iterations{5};
    std::size_t measurement_iterations{20};
};

struct TimingSample {
    double host_ms = 0.0;
    std::optional<double> gpu_ms;
};

struct LatencyStats {
    double min_ms = 0.0;
    double median_ms = 0.0;
    double mean_ms = 0.0;
    double p95_ms = 0.0;
    double variance_ms2 = 0.0;
    double stddev_ms = 0.0;
};

// Requires a prepared backend. Raw samples remain in execution order.
std::vector<TimingSample> run_benchmark_samples(ComputeBackend& backend, const BenchmarkConfig& config);

// Compatible Week 1/2 interface: returns host-observed latency in milliseconds.
std::vector<double> run_benchmark(ComputeBackend& backend, const BenchmarkConfig& config);

// P95 uses nearest rank; variance uses the population denominator N.
LatencyStats summarize_latencies(const std::vector<double>& samples);

} // namespace heterobench
