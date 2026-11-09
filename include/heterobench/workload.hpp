#pragma once

#include <cstddef>
#include <vector>

namespace heterobench {

struct WorkloadConfig {
    std::size_t matrix_size{0};
};

struct MatrixData{
    std::vector<float> a;
    std::vector<float> b;
    std::vector<float> c;
};



} // namespace heterobench
