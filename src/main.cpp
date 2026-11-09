#include "heterobench/benchmark.hpp"
#include "heterobench/cpu_backend.hpp"
#include "heterobench/results.hpp"
#include "heterobench/validation.hpp"
#ifdef HETEROBENCH_HAS_METAL
#include "heterobench/metal_backend.hpp"
#endif
#ifdef HETEROBENCH_HAS_ACCELERATE
#include "heterobench/accelerate_backend.hpp"
#endif

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Options {
    std::string backend = "cpu";
    std::string reference = "cpu";
    std::vector<std::size_t> dimensions{2};
    heterobench::BenchmarkConfig benchmark;
    std::size_t runs = 1;
    std::string output;
};

std::size_t parse_integer(std::string_view text, bool allow_zero = false) {
    std::size_t value = 0;
    const auto conversion = std::from_chars(text.data(), text.data() + text.size(), value);
    if (text.empty() || conversion.ec != std::errc{} || conversion.ptr != text.data() + text.size() ||
        (!allow_zero && value == 0)) {
        throw std::invalid_argument("Invalid nonnegative integer: " + std::string(text));
    }
    return value;
}

Options parse_options(int argc, char* argv[]) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument = argv[index];
        if (index + 1 >= argc) throw std::invalid_argument("Missing value for " + std::string(argument));
        const std::string_view value = argv[++index];
        if (argument == "--backend") options.backend = value;
        else if (argument == "--reference") options.reference = value;
        else if (argument == "--size") options.dimensions = {parse_integer(value)};
        else if (argument == "--sizes") {
            options.dimensions.clear();
            std::size_t start = 0;
            do {
                const auto end = value.find(',', start);
                options.dimensions.push_back(parse_integer(value.substr(start, end - start)));
                if (end == std::string_view::npos) break;
                start = end + 1;
            } while (true);
        } else if (argument == "--warmup") options.benchmark.warmup_iterations = parse_integer(value, true);
        else if (argument == "--iterations") options.benchmark.measurement_iterations = parse_integer(value);
        else if (argument == "--runs") options.runs = parse_integer(value);
        else if (argument == "--output") {
            if (value.empty()) throw std::invalid_argument("Output prefix must be nonempty");
            options.output = value;
        } else throw std::invalid_argument("Unknown argument: " + std::string(argument));
    }
    if (options.reference != "cpu" && options.reference != "accelerate") {
        throw std::invalid_argument("--reference must be cpu or accelerate");
    }
#ifndef HETEROBENCH_HAS_ACCELERATE
    if (options.reference == "accelerate") {
        throw std::invalid_argument("This build has no Accelerate validation reference");
    }
#endif
    auto sorted_dimensions = options.dimensions;
    std::sort(sorted_dimensions.begin(), sorted_dimensions.end());
    if (std::adjacent_find(sorted_dimensions.begin(), sorted_dimensions.end()) != sorted_dimensions.end()) {
        throw std::invalid_argument("Matrix dimensions in a sweep must be unique");
    }
    return options;
}

std::vector<std::string> selected_backends(const std::string& requested) {
    std::vector<std::string> available{"cpu"};
#ifdef HETEROBENCH_HAS_ACCELERATE
    available.push_back("accelerate");
#endif
#ifdef HETEROBENCH_HAS_METAL
    available.insert(available.end(), {"metal", "metal-tiled", "mps"});
#endif
    if (requested == "all") return available;
    if (requested == "both") {
        if (std::find(available.begin(), available.end(), "metal") == available.end()) {
            throw std::invalid_argument("This build has no Metal support");
        }
        return {"cpu", "metal"};
    }
    std::vector<std::string> selected;
    std::size_t start = 0;
    do {
        const auto end = requested.find(',', start);
        const auto name = requested.substr(start, end - start);
        if (std::find(available.begin(), available.end(), name) == available.end()) {
            throw std::invalid_argument("Unknown or unavailable backend: " + name);
        }
        if (std::find(selected.begin(), selected.end(), name) != selected.end()) {
            throw std::invalid_argument("Duplicate backend: " + name);
        }
        selected.push_back(name);
        if (end == std::string::npos) break;
        start = end + 1;
    } while (true);
    return selected;
}

std::unique_ptr<heterobench::ComputeBackend> make_backend(const std::string& name) {
    if (name == "cpu") return std::make_unique<heterobench::CpuBackend>();
#ifdef HETEROBENCH_HAS_ACCELERATE
    if (name == "accelerate") return std::make_unique<heterobench::AccelerateBackend>();
#endif
#ifdef HETEROBENCH_HAS_METAL
    auto kernel = heterobench::MetalKernel::Naive;
    if (name == "metal-tiled") kernel = heterobench::MetalKernel::Tiled;
    if (name == "mps") kernel = heterobench::MetalKernel::Mps;
    return std::make_unique<heterobench::MetalBackend>(kernel);
#else
    throw std::invalid_argument("Unavailable backend: " + name);
#endif
}

heterobench::MatrixData make_inputs(std::size_t dimension) {
    const auto maximum = std::numeric_limits<std::size_t>::max();
    if (dimension == 0 || dimension > maximum / dimension) throw std::invalid_argument("Matrix count overflows");
    const auto count = dimension * dimension;
    if (count > maximum / sizeof(float) || count > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("Matrix exceeds the supported indexing range");
    }
    heterobench::MatrixData data;
    data.a.resize(count);
    data.b.resize(count);
    data.c.resize(count);
    if (dimension == 2) {
        data.a = {1, 2, 3, 4};
        data.b = {5, 6, 7, 8};
    } else {
        for (std::size_t index = 0; index < count; ++index) {
            data.a[index] = static_cast<float>(static_cast<int>(index % 13) - 6) * 0.125f;
            data.b[index] = static_cast<float>(static_cast<int>(index % 11) - 5) * 0.2f;
        }
    }
    return data;
}

} // namespace

int main(int argc, char* argv[]) {
    if (argc == 2 && std::string_view(argv[1]) == "--help") {
        std::cout << "Usage: heterobench [--backend cpu|accelerate|metal|metal-tiled|mps|both|all]\n"
                  << "  [--size N | --sizes 64,128,256] [--warmup 5] [--iterations 20]\n"
                  << "  [--runs 1] [--output PREFIX] [--reference cpu|accelerate]\n"
                  << "  --backend also accepts a comma-separated subset.\n"
                  << "Default: CPU, size 2. Output creates PREFIX.samples.csv and PREFIX.summary.json.\n";
        return 0;
    }
    try {
        const Options options = parse_options(argc, argv);
        const auto backends = selected_backends(options.backend);
        auto metadata = heterobench::collect_metadata();
        std::string command;
        for (int i = 0; i < argc; ++i) {
            if (i != 0) command += ' ';
            command += argv[i];
        }
        metadata["command"] = command;
        metadata["validation_reference"] = options.reference;
        metadata["reference_spot_checks"] = "16 deterministic output positions accumulated in double";
        if (metadata.at("build_type") != "Release" && metadata.at("build_type") != "RelWithDebInfo") {
            std::cerr << "Performance measurements should use an optimized build.\n";
        }
        std::vector<heterobench::ExperimentResult> results;
        for (std::size_t dimension : options.dimensions) {
            const heterobench::WorkloadConfig workload{dimension};
            const auto inputs = make_inputs(dimension);
            auto reference_data = inputs;
            auto reference = make_backend(options.reference);
            reference->prepare(workload, reference_data);
            reference->execute();
            reference->synchronize();

            // Independently verify selected reference outputs with double accumulation.
            // This is O(16 * N), so large validation need not run scalar O(N^3) GEMM.
            for (std::size_t check = 0; check < 16; ++check) {
                const std::size_t row = (check * 37) % dimension;
                const std::size_t column = (check * 53) % dimension;
                double expected = 0.0;
                for (std::size_t k = 0; k < dimension; ++k) {
                    expected += static_cast<double>(inputs.a[row * dimension + k]) *
                                static_cast<double>(inputs.b[k * dimension + column]);
                }
                heterobench::validate_result(
                    {reference_data.c[row * dimension + column]},
                    {static_cast<float>(expected)}
                );
            }
            if (dimension == 2) heterobench::validate_result(reference_data.c, {19, 22, 43, 50});

            for (std::size_t repetition = 0; repetition < options.runs; ++repetition) {
                // Rotate order across repetitions to reduce a fixed-order confound.
                for (std::size_t slot = 0; slot < backends.size(); ++slot) {
                    const std::string& name = backends[(slot + repetition) % backends.size()];
                    auto data = inputs;
                    auto backend = make_backend(name);
                    std::string device = metadata.count("cpu") ? metadata.at("cpu") : "host";
#ifdef HETEROBENCH_HAS_METAL
                    if (auto* metal = dynamic_cast<heterobench::MetalBackend*>(backend.get())) {
                        device = metal->device_name();
                    }
#endif
                    backend->prepare(workload, data);
                    backend->execute();
                    backend->synchronize();
                    heterobench::validate_result(data.c, reference_data.c);
                    const auto samples = heterobench::run_benchmark_samples(*backend, options.benchmark);
                    heterobench::validate_result(data.c, reference_data.c);
                    std::vector<double> host, gpu;
                    for (const auto& sample : samples) {
                        host.push_back(sample.host_ms);
                        if (sample.gpu_ms) gpu.push_back(*sample.gpu_ms);
                    }
                    const auto host_stats = heterobench::summarize_latencies(host);
                    std::cout << name << " N=" << dimension << " run=" << repetition
                              << " validation=passed samples=" << samples.size()
                              << " host_median_ms=" << host_stats.median_ms
                              << " host_p95_ms=" << host_stats.p95_ms;
                    if (!gpu.empty()) {
                        std::cout << " gpu_command_buffer_median_ms="
                                  << heterobench::summarize_latencies(gpu).median_ms;
                    } else std::cout << " gpu_ms=unavailable";
                    std::cout << '\n';
                    results.push_back({name, device, dimension, repetition, options.benchmark, samples});
                }
            }
        }
        if (!options.output.empty()) heterobench::write_results(options.output, metadata, results);
    } catch (const std::exception& error) {
        std::cerr << "HeteroBench failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
