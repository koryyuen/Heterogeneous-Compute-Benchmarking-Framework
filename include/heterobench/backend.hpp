#pragma once
#include "heterobench/workload.hpp"
#include <optional>

//prepare - validates configuration and prepares the backend for execution
namespace heterobench {
    class ComputeBackend {
    public:
        virtual ~ComputeBackend() = default;
        virtual void prepare(const WorkloadConfig& config, MatrixData& data) = 0;
        virtual void execute() = 0;
        virtual void synchronize() = 0;

        // Supplementary device timing for the last completed operation, if available.
        virtual std::optional<double> last_gpu_time_ms() const { return std::nullopt; }
    };
} // namespace heterobench