#pragma once
#include "heterobench/backend.hpp"
#include <memory>
#include <string>

namespace heterobench {

enum class MetalKernel { Naive, Tiled, Mps };

class MetalBackend : public ComputeBackend {
public:
    explicit MetalBackend(MetalKernel kernel = MetalKernel::Naive);
    ~MetalBackend() override;

    static bool is_available();
    std::string device_name() const;

    // Inputs are copied during preparation. The caller owns data and must
    // keep it alive, with unchanged vector sizes, until execution completes.
    void prepare(const WorkloadConfig& config, MatrixData& data) override;

    // One execution may be pending. Synchronize before executing or preparing again.
    void execute() override;
    void synchronize() override;
    std::optional<double> last_gpu_time_ms() const override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace heterobench
