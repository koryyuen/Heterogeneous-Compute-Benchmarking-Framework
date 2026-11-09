#pragma once
#include "heterobench/backend.hpp"

namespace heterobench {
    class CpuBackend : public ComputeBackend {
    public:
        virtual ~CpuBackend() override = default;
        void prepare(const WorkloadConfig& config, MatrixData& data) override;
        void execute() override;
        void synchronize() override;
    private:
        WorkloadConfig config_;
        MatrixData* data_{nullptr};
    };
} // namespace heterobench