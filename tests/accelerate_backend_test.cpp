#include "heterobench/accelerate_backend.hpp"
#include "heterobench/cpu_backend.hpp"
#include "heterobench/validation.hpp"
#include <iostream>
#include <stdexcept>

int main() {
    try {
        heterobench::AccelerateBackend unprepared;
        bool rejected = false;
        try { unprepared.execute(); } catch (const std::logic_error&) { rejected = true; }
        if (!rejected) throw std::runtime_error("Unprepared Accelerate execution accepted");
        for (std::size_t dimension : {1, 2, 3, 17, 32}) {
            heterobench::MatrixData data;
            data.a.resize(dimension * dimension);
            data.b.resize(dimension * dimension);
            data.c.assign(dimension * dimension, 99);
            for (std::size_t i = 0; i < data.a.size(); ++i) {
                data.a[i] = static_cast<float>(static_cast<int>(i % 13) - 6) * 0.125f;
                data.b[i] = static_cast<float>(static_cast<int>(i % 11) - 5) * 0.2f;
            }
            auto reference = data;
            heterobench::CpuBackend cpu;
            cpu.prepare({dimension}, reference);
            cpu.execute();
            heterobench::AccelerateBackend accelerated;
            accelerated.prepare({dimension}, data);
            accelerated.execute();
            accelerated.synchronize();
            heterobench::validate_result(data.c, reference.c);
            accelerated.execute();
            heterobench::validate_result(data.c, reference.c);
            data.c.pop_back();
            rejected = false;
            try { accelerated.execute(); } catch (const std::logic_error&) { rejected = true; }
            if (!rejected) throw std::runtime_error("Resized Accelerate output accepted");
        }
        std::cout << "PASS: Accelerate matches CPU, overwrites output, and rejects invalid lifecycle\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
