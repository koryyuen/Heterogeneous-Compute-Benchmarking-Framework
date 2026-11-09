#include "heterobench/accelerate_backend.hpp"
#include <Accelerate/Accelerate.h>
#include <limits>
#include <stdexcept>

namespace heterobench {

void AccelerateBackend::prepare(const WorkloadConfig& config, MatrixData& data) {
    const auto n = config.matrix_size;
    if (n == 0 || n > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        n > std::numeric_limits<std::size_t>::max() / n) {
        throw std::invalid_argument("Invalid Accelerate matrix dimension");
    }
    if (data.a.size() != n * n || data.b.size() != n * n || data.c.size() != n * n) {
        throw std::invalid_argument("Invalid Accelerate matrix buffer sizes");
    }
    dimension_ = static_cast<int>(n);
    data_ = &data;
}

void AccelerateBackend::execute() {
    if (data_ == nullptr) throw std::logic_error("Accelerate execution requires preparation");
    const auto count = static_cast<std::size_t>(dimension_) * dimension_;
    if (data_->a.size() != count || data_->b.size() != count || data_->c.size() != count) {
        throw std::logic_error("Matrix sizes changed after Accelerate preparation");
    }
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                dimension_, dimension_, dimension_, 1.0f,
                data_->a.data(), dimension_, data_->b.data(), dimension_,
                0.0f, data_->c.data(), dimension_);
}

void AccelerateBackend::synchronize() {
    // cblas_sgemm returns after the result is available to the host.
}

} // namespace heterobench
