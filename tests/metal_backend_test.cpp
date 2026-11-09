#include "heterobench/benchmark.hpp"
#include "heterobench/cpu_backend.hpp"
#include "heterobench/metal_backend.hpp"
#include "heterobench/validation.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

template<typename Exception, typename Function>
void expect_exception(Function action, const std::string& name) {
    try {
        action();
    } catch (const Exception&) {
        return;
    }
    throw std::runtime_error(name + ": expected exception was not thrown");
}

heterobench::MatrixData fractional_inputs(std::size_t dimension) {
    heterobench::MatrixData data;
    const std::size_t count = dimension * dimension;
    data.a.resize(count);
    data.b.resize(count);
    data.c.assign(count, 99.0f);
    for (std::size_t index = 0; index < count; ++index) {
        data.a[index] = static_cast<float>(static_cast<int>(index % 13) - 6) * 0.125f;
        data.b[index] = static_cast<float>(static_cast<int>(index % 11) - 5) * 0.2f;
    }
    return data;
}

void check_product(
    std::size_t dimension,
    heterobench::MatrixData data,
    const std::string& name
) {
    auto reference_data = data;
    const heterobench::WorkloadConfig config{dimension};
    heterobench::CpuBackend cpu;
    cpu.prepare(config, reference_data);
    cpu.execute();
    cpu.synchronize();

    for (const auto kernel : {heterobench::MetalKernel::Naive,
                               heterobench::MetalKernel::Tiled,
                               heterobench::MetalKernel::Mps}) {
        heterobench::MetalBackend metal(kernel);
        metal.prepare(config, data);
        if (metal.last_gpu_time_ms()) throw std::runtime_error("Stale GPU timestamp after prepare");
        metal.execute();
        metal.synchronize();
        heterobench::validate_result(data.c, reference_data.c);
        const auto gpu_ms = metal.last_gpu_time_ms();
        if (gpu_ms && (!std::isfinite(*gpu_ms) || *gpu_ms <= 0)) {
            throw std::runtime_error("Invalid GPU command-buffer timestamp");
        }
        std::fill(data.c.begin(), data.c.end(), 99.0f);
        metal.execute();
        if (metal.last_gpu_time_ms()) throw std::runtime_error("Stale GPU timestamp during execution");
        metal.synchronize();
        heterobench::validate_result(data.c, reference_data.c);
    }
    std::cout << "PASS: naive/tiled/MPS " << name << " and repeated execution\n";

}

} // namespace

int main() {
    if (!heterobench::MetalBackend::is_available()) {
        std::cout << "SKIP: no Metal device is available; GPU correctness was not tested\n";
        return 77;
    }

    try {
        // Caller-owned data is declared before the backend that borrows it.
        heterobench::MatrixData data{{1, 2, 3, 4}, {5, 6, 7, 8}, {99, 99, 99, 99}};
        heterobench::MetalBackend metal;
        std::cout << "Metal device: " << metal.device_name() << '\n';

        expect_exception<std::logic_error>([&] { metal.execute(); }, "execute before prepare");
        expect_exception<std::logic_error>([&] { metal.synchronize(); }, "sync without execution");
        expect_exception<std::invalid_argument>([&] { metal.prepare({0}, data); }, "zero dimension");
        expect_exception<std::invalid_argument>([&] {
            metal.prepare({std::numeric_limits<std::size_t>::max()}, data);
        }, "dimension overflow");

        const std::size_t byte_overflow_dimension =
            std::size_t{1} << (std::numeric_limits<std::size_t>::digits / 2 - 1);
        expect_exception<std::invalid_argument>([&] {
            metal.prepare({byte_overflow_dimension}, data);
        }, "byte count overflow");
        expect_exception<std::invalid_argument>([&] {
            metal.prepare({65536}, data);
        }, "shader index overflow");

        for (int buffer = 0; buffer < 3; ++buffer) {
            auto invalid = data;
            if (buffer == 0) invalid.a.pop_back();
            if (buffer == 1) invalid.b.pop_back();
            if (buffer == 2) invalid.c.pop_back();
            expect_exception<std::invalid_argument>([&] {
                metal.prepare({2}, invalid);
            }, "incorrect buffer size");
        }
        std::cout << "PASS: invalid dimensions, sizes, overflow, and unprepared calls\n";

        metal.prepare({2}, data);
        metal.execute();
        expect_exception<std::logic_error>([&] { metal.execute(); }, "double submission");
        expect_exception<std::logic_error>([&] { metal.prepare({2}, data); }, "prepare while pending");
        metal.synchronize();
        heterobench::validate_result(data.c, {19, 22, 43, 50});
        expect_exception<std::logic_error>([&] { metal.synchronize(); }, "double synchronization");

        // A failed re-prepare must preserve the previous usable state.
        expect_exception<std::invalid_argument>([&] { metal.prepare({0}, data); }, "failed re-prepare");
        metal.execute();
        metal.synchronize();
        heterobench::validate_result(data.c, {19, 22, 43, 50});

        // Inputs are snapshots, uploaded once during preparation.
        data.a = {2, 0, 0, 2};
        metal.execute();
        metal.synchronize();
        heterobench::validate_result(data.c, {19, 22, 43, 50});
        metal.prepare({2}, data);
        metal.execute();
        metal.synchronize();
        heterobench::validate_result(data.c, {10, 12, 14, 16});

        data.c.pop_back();
        expect_exception<std::logic_error>([&] { metal.execute(); }, "resized output");
        data.c.resize(4);
        metal.execute();
        data.c.pop_back();
        expect_exception<std::logic_error>([&] { metal.synchronize(); }, "resized readback");
        data.c.resize(4);
        std::cout << "PASS: pending-work guards, input snapshots, and re-preparation\n";

        check_product(1, {{3}, {4}, {99}}, "1 x 1");
        check_product(2, {{1, 2, 3, 4}, {5, 6, 7, 8}, {99, 99, 99, 99}}, "known 2 x 2");
        check_product(2, {{2, -3, 4, 5}, {1, 0, 0, 1}, {99, 99, 99, 99}}, "identity with negatives");
        check_product(3, fractional_inputs(3), "fractional 3 x 3 padded grid");
        check_product(17, fractional_inputs(17), "fractional 17 x 17 padded grid");
        check_product(31, fractional_inputs(31), "fractional 31 x 31 padded grid");
        check_product(65, fractional_inputs(65), "fractional 65 x 65 tile edges");
        check_product(32, fractional_inputs(32), "fractional 32 x 32 aligned grid");

        // The same benchmark loop drives Metal through ComputeBackend&.
        metal.prepare({2}, data);
        const auto samples = heterobench::run_benchmark(metal, {2, 3});
        if (samples.size() != 3) {
            throw std::runtime_error("Incorrect number of Metal latency samples");
        }
        for (double sample : samples) {
            if (!std::isfinite(sample) || sample < 0.0) {
                throw std::runtime_error("Invalid Metal latency sample");
            }
        }
        heterobench::validate_result(data.c, {10, 12, 14, 16});
        const auto no_warmup = heterobench::run_benchmark(metal, {0, 1});
        if (no_warmup.size() != 1) {
            throw std::runtime_error("Zero-warmup benchmark failed");
        }
        expect_exception<std::invalid_argument>([&] {
            heterobench::run_benchmark(metal, {1, 0});
        }, "zero measurements");

        expect_exception<std::runtime_error>([] {
            heterobench::validate_result({std::numeric_limits<float>::quiet_NaN()}, {1});
        }, "NaN output");
        expect_exception<std::runtime_error>([] {
            heterobench::validate_result({std::numeric_limits<float>::infinity()}, {1});
        }, "infinite output");
        expect_exception<std::runtime_error>([] {
            heterobench::validate_result({2}, {1});
        }, "incorrect finite output");
        std::cout << "PASS: shared benchmark loop, sample counts, and nonfinite rejection\n";
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
