#include <iostream>
#include <stdexcept>
#include <heterobench/workload.hpp>
#include <heterobench/cpu_backend.hpp>

int main() {
    heterobench::CpuBackend backend;
    heterobench::WorkloadConfig config{0};
    heterobench::MatrixData data;

    //Invalid Input Test
    try {
        backend.prepare(config, data);
        std::cerr << "FAIL: invalid input was accepted\n";
        return 1;
    } catch (const std::invalid_argument& e) {
        std::cout << "PASS: invalid input rejected: " << e.what() << '\n';
    }

    //Invalid Input Test
    config.matrix_size = 2;
    try {
        backend.prepare(config, data);
        std::cerr << "FAIL: invalid input was accepted\n";
        return 1;
    } catch (const std::invalid_argument& e) {
        std::cout << "PASS: invalid input rejected: " << e.what() << '\n';
    }

    //1x1 Test
    config.matrix_size = 1;
    data.a.resize(config.matrix_size * config.matrix_size);
    data.b.resize(config.matrix_size * config.matrix_size);
    data.c.resize(config.matrix_size * config.matrix_size);
    data.a[0] = 3.0f;
    data.b[0] = 4.0f;
    data.c[0] = 0.0f;
    backend.prepare(config, data);
    backend.execute();
    backend.synchronize();
    if (data.c[0] != 12.0f) {
        std::cerr << "FAIL: c = " << data.c[0] << ", expected 12.0f\n";
        return 1;
    }

    //2x2 Test
    config.matrix_size = 2;
    data.a.resize(config.matrix_size * config.matrix_size);
    data.b.resize(config.matrix_size * config.matrix_size);
    data.c.resize(config.matrix_size * config.matrix_size);
    data.a[0] = 2.0f;
    data.a[1] = -3.0f;
    data.a[2] = 4.0f;
    data.a[3] = 5.0f;
    data.b[0] = 1.0f;
    data.b[1] = 0.0f;
    data.b[2] = 0.0f;
    data.b[3] = 1.0f;
    data.c.assign(config.matrix_size * config.matrix_size, 99.0f);
    backend.prepare(config, data);
    backend.execute();
    backend.synchronize();
    float expected_c[4] = {2.0f, -3.0f, 4.0f, 5.0f};
    for (int i = 0; i < 4; ++i) {
        if (data.c[i] != expected_c[i]) {
            std::cerr << "FAIL: c[" << i << "] = " << data.c[i] << ", expected " << expected_c[i] << '\n';
            return 1;
        }
    }
    backend.execute();
    backend.synchronize();
    for (int i = 0; i < 4; ++i) {
        if (data.c[i] != expected_c[i]) {
            std::cerr << "FAIL: c[" << i << "] = " << data.c[i] << ", expected " << expected_c[i] << '\n';
            return 1;
        }
    }
    
    return 0;
}