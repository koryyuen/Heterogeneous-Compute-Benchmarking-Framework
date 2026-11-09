#include "heterobench/benchmark.hpp"
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
class CountingBackend : public heterobench::ComputeBackend {
public:
    int executions = 0;
    int completions = 0;
    void prepare(const heterobench::WorkloadConfig&, heterobench::MatrixData&) override {}
    void execute() override {
        if (executions != completions) throw std::runtime_error("execute/sync order changed");
        ++executions;
    }
    void synchronize() override { ++completions; }
    std::optional<double> last_gpu_time_ms() const override { return 0.25; }
};

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
}

int main() {
    try {
        const std::vector<double> original{8, 2, 6, 4};
        const auto stats = heterobench::summarize_latencies(original);
        require(stats.min_ms == 2 && stats.median_ms == 5 && stats.mean_ms == 5, "even statistics");
        require(stats.p95_ms == 8 && stats.variance_ms2 == 5, "P95 and population variance");
        require(std::abs(stats.stddev_ms - std::sqrt(5.0)) < 1e-12, "standard deviation");
        require(original == std::vector<double>({8, 2, 6, 4}), "sample order changed");
        const auto odd = heterobench::summarize_latencies({9, 1, 5});
        require(odd.min_ms == 1 && odd.median_ms == 5 && odd.mean_ms == 5, "odd statistics");
        const auto single = heterobench::summarize_latencies({0});
        require(single.median_ms == 0 && single.p95_ms == 0 && single.variance_ms2 == 0, "singleton");
        std::vector<double> percentile;
        for (int i = 1; i <= 100; ++i) percentile.push_back(i);
        require(heterobench::summarize_latencies(percentile).p95_ms == 95, "nearest-rank P95");
        for (const auto& invalid : std::vector<std::vector<double>>{
                {}, {-1}, {std::numeric_limits<double>::quiet_NaN()},
                {std::numeric_limits<double>::infinity()}}) {
            bool rejected = false;
            try { heterobench::summarize_latencies(invalid); }
            catch (const std::invalid_argument&) { rejected = true; }
            require(rejected, "invalid statistics input accepted");
        }
        CountingBackend backend;
        const auto samples = heterobench::run_benchmark_samples(backend, {2, 3});
        require(samples.size() == 3 && backend.executions == 5 && backend.completions == 5, "iteration counts");
        for (const auto& sample : samples) {
            require(sample.host_ms >= 0 && sample.gpu_ms == 0.25, "paired timing sample");
        }
        require(heterobench::run_benchmark(backend, {0, 1}).size() == 1, "compatible host API");
        bool rejected = false;
        try { heterobench::run_benchmark_samples(backend, {0, 0}); }
        catch (const std::invalid_argument&) { rejected = true; }
        require(rejected && backend.executions == 6, "zero measurements executed work");
        std::cout << "PASS: stats, P95, variance, raw order, paired timings, and iteration lifecycle\n";
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
