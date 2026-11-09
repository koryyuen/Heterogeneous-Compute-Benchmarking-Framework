#include "heterobench/metal_backend.hpp"
#include "heterobench/validation.hpp"

#include <exception>
#include <iostream>

int main() {
    @autoreleasepool {
        if (!heterobench::MetalBackend::is_available()) {
            std::cout << "SKIP: no Metal device is available\n";
            return 77;
        }

        try {
            const heterobench::WorkloadConfig config{2};
            heterobench::MatrixData data{
                {1.0f, 2.0f, 3.0f, 4.0f},
                {5.0f, 6.0f, 7.0f, 8.0f},
                {99.0f, 99.0f, 99.0f, 99.0f}
            };
            heterobench::MetalBackend backend;
            std::cout << "Metal device: " << backend.device_name() << '\n';
            backend.prepare(config, data);
            backend.execute();
            backend.synchronize();
            heterobench::validate_result(data.c, {19.0f, 22.0f, 43.0f, 50.0f});
            std::cout << "PASS: Metal result is [19, 22, 43, 50]\n";
        } catch (const std::exception& error) {
            std::cerr << "Metal smoke check failed: " << error.what() << '\n';
            return 1;
        }
    }
    return 0;
}
