#pragma once
#include "heterobench/backend.hpp"

namespace heterobench {

// An optimized library reference. Do not assume Accelerate's internal hardware
// resources or threading strategy from this interface.
class AccelerateBackend : public ComputeBackend {
public:
    void prepare(const WorkloadConfig& config, MatrixData& data) override;
    void execute() override;
    void synchronize() override;
private:
    MatrixData* data_ = nullptr;
    int dimension_ = 0;
};

} // namespace heterobench
