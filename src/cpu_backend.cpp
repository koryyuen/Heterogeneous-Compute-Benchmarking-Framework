#include "heterobench/cpu_backend.hpp"

#include <limits>
#include <stdexcept>

namespace heterobench {

void CpuBackend::prepare(const WorkloadConfig& config, MatrixData& data) {
    const std::size_t n = config.matrix_size;
    if (n == 0) {
        throw std::invalid_argument("Matrix dimension must be nonzero");
    }
    if (n > std::numeric_limits<std::size_t>::max() / n) {
        throw std::invalid_argument("Matrix element count overflows std::size_t");
    }

    const std::size_t element_count = n * n;
    if (data.a.size() != element_count ||
        data.b.size() != element_count ||
        data.c.size() != element_count) {
        throw std::invalid_argument("Each matrix must contain matrix_size * matrix_size elements");
    }

    config_ = config;
    data_ = &data;
}

void CpuBackend::execute() {
    if (!data_) {
        throw std::runtime_error("CpuBackend::execute called before prepare");
    }
    const std::size_t n = config_.matrix_size;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (std::size_t k = 0; k < n; ++k) {
                sum += data_->a[i * n + k] * data_->b[k * n + j];
            }
            data_->c[i * n + j] = sum;
        }
    }
}

void CpuBackend::synchronize() {
    // No-op for CPU backend
}

} // namespace heterobench
