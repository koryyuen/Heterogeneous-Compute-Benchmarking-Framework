#pragma once

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace heterobench {

// Use absolute plus relative tolerance for these FP32 workloads.
// Reject nonfinite values explicitly because comparisons with NaN are unsafe.
inline void validate_result(
    const std::vector<float>& actual,
    const std::vector<float>& expected
) {
    if (actual.size() != expected.size()) {
        throw std::runtime_error("Result vector sizes do not match");
    }

    constexpr float absolute_tolerance = 1.0e-4f;
    constexpr float relative_tolerance = 1.0e-4f;

    for (std::size_t index = 0; index < expected.size(); ++index) {
        const float allowed_error =
            absolute_tolerance + relative_tolerance * std::abs(expected[index]);

        if (!std::isfinite(actual[index]) || !std::isfinite(expected[index]) ||
            std::abs(actual[index] - expected[index]) > allowed_error) {
            throw std::runtime_error(
                "Result mismatch at index " + std::to_string(index) +
                ": expected " + std::to_string(expected[index]) +
                ", got " + std::to_string(actual[index])
            );
        }
    }
}

} // namespace heterobench
