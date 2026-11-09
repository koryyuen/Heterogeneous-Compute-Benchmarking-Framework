#include "heterobench/benchmark.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>

namespace heterobench {

std::vector<TimingSample> run_benchmark_samples(ComputeBackend& backend, const BenchmarkConfig& config) {
    if (config.measurement_iterations == 0) {
        throw std::invalid_argument("Measurement iteration count must be nonzero");
    }
    std::vector<TimingSample> samples;
    samples.reserve(config.measurement_iterations);

    for (std::size_t i = 0; i < config.warmup_iterations; ++i) {
        backend.execute();
        backend.synchronize();
    }
    for (std::size_t i = 0; i < config.measurement_iterations; ++i) {
        const auto start = std::chrono::steady_clock::now();
        backend.execute();
        backend.synchronize();
        const auto end = std::chrono::steady_clock::now();

        const double host_ms = std::chrono::duration<double, std::milli>(end - start).count();
        auto gpu_ms = backend.last_gpu_time_ms();
        if (gpu_ms && (!std::isfinite(*gpu_ms) || *gpu_ms <= 0.0)) {
            gpu_ms.reset();
        }
        samples.push_back({host_ms, gpu_ms});
    }
    return samples;
}

std::vector<double> run_benchmark(ComputeBackend& backend, const BenchmarkConfig& config) {
    const auto samples = run_benchmark_samples(backend, config);
    std::vector<double> host;
    host.reserve(samples.size());
    for (const auto& sample : samples) host.push_back(sample.host_ms);
    return host;
}

LatencyStats summarize_latencies(const std::vector<double>& samples) {
    if (samples.empty()) throw std::invalid_argument("Cannot summarize empty latency samples");

    // Welford's update avoids subtracting large, nearly equal sums for variance.
    LatencyStats stats;
    double squared_deviation = 0.0;
    std::size_t count = 0;
    for (double value : samples) {
        if (!std::isfinite(value) || value < 0.0) {
            throw std::invalid_argument("Latencies must be finite and nonnegative");
        }
        ++count;
        const double difference = value - stats.mean_ms;
        stats.mean_ms += difference / static_cast<double>(count);
        squared_deviation += difference * (value - stats.mean_ms);
    }
    stats.variance_ms2 = std::max(0.0, squared_deviation / static_cast<double>(count));
    stats.stddev_ms = std::sqrt(stats.variance_ms2);

    auto sorted = samples;
    std::sort(sorted.begin(), sorted.end());
    stats.min_ms = sorted.front();
    const std::size_t middle = sorted.size() / 2;
    stats.median_ms = sorted[middle];
    if (sorted.size() % 2 == 0) {
        stats.median_ms = sorted[middle - 1] / 2.0 + sorted[middle] / 2.0;
    }
    const auto rank = static_cast<std::size_t>(std::ceil(0.95 * static_cast<double>(sorted.size())));
    stats.p95_ms = sorted[rank - 1];
    return stats;
}

} // namespace heterobench
