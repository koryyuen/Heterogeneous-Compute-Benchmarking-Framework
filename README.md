# HeteroBench

**A C++20 benchmarking and performance analysis framework for heterogeneous computing on Apple Silicon.**

HeteroBench evaluates matrix multiplication performance across CPU and GPU implementations, including custom Metal compute kernels and Apple's optimized Accelerate and Metal Performance Shaders (MPS) libraries.

The project investigates how workload size, GPU memory access patterns, threadgroup tiling, and execution overhead affect computational performance.

## Key Results

Custom Metal GPU matrix multiplication achieved:

- **Up to 2.49× faster GPU command-buffer execution** using 16×16 threadgroup tiling compared with a naive Metal kernel.
- **1.81× GPU execution speedup at 4096×4096**, demonstrating that the optimization remains beneficial for larger workloads.
- Profiling evidence showing that host-side execution overhead has a much greater relative impact on small workloads than large ones.

Benchmarks were performed on an **Apple M2 Pro** using deterministic FP32 matrix multiplication workloads.

### Large-Workload Performance

Host-observed median execution latency, in milliseconds (lower is better):

| Matrix Size | Accelerate | Naive Metal | Tiled Metal | MPS |
|---|---:|---:|---:|---:|
| 1024×1024 | 0.794 | 6.430 | 3.648 | **0.718** |
| 2048×2048 | 7.524 | 45.049 | 27.679 | **4.311** |
| 4096×4096 | 63.247 | 395.822 | 220.454 | **32.999** |

**Key observation:** Although tiling substantially improves custom Metal performance, Apple's optimized MPS implementation remains faster at larger matrix dimensions. At 4096×4096, MPS achieved approximately 6.68× lower host latency than the custom tiled kernel.

The smaller-workload experiments also showed that Accelerate could outperform the GPU implementations in host-observed latency, illustrating the importance of workload size and runtime overhead.

These are results from one system and specific implementations, not general hardware performance rankings.

See [Detailed Benchmark Results](docs/RESULTS.md) and [Large-Workload Results](docs/LARGE_WORKLOAD_RESULTS.md).

## Features

### Heterogeneous Execution Backends

Five FP32 matrix multiplication implementations:

| Backend | Description |
|---|---|
| CPU Reference | Single-threaded C++ matrix multiplication |
| Accelerate | Optimized BLAS `cblas_sgemm` |
| Metal | Custom naive GPU compute shader |
| Metal Tiled | Custom 16×16 tiled GPU compute shader |
| MPS | Apple's `MPSMatrixMultiplication` |

### Performance Measurement

- Host-observed completed-operation latency
- Metal GPU command-buffer execution duration
- Minimum, mean, median, and P95 latency
- Variance and standard deviation
- Achieved GFLOP/s
- Configurable warmup and measurement iterations
- Multiple independent experiment configurations and repeated runs
- Raw CSV and summarized JSON output
- Experiment metadata and source/binary provenance
- Automated performance comparisons and plots

### Correctness and Validation

- CPU reference comparisons
- Absolute and relative floating-point tolerance checks
- Input validation and matrix dimension overflow protection
- Edge-dimension testing for GPU dispatch
- Backend lifecycle and synchronization testing
- CTest integration

## Architecture

HeteroBench separates workload configuration, backend execution, measurement, and result export.

```text
                   HeteroBench CLI
                          |
                 Workload Configuration
                          |
                   Benchmark Runner
                          |
                 ComputeBackend Interface
                          |
          +---------------+----------------+
          |               |                |
        CPU          Accelerate        Metal / MPS
          |               |                |
      C++ GEMM         BLAS GEMM        GPU Execution
          |               |                |
          +---------------+----------------+
                          |
                Correctness Validation
                          |
                Latency and Statistics
                          |
                    CSV / JSON
                          |
                 Python Visualization
```

The core framework uses **C++20**, while Apple-specific Metal integration is isolated using **Objective-C++**.

The backend interface defines a common execution lifecycle:

```cpp
prepare(config, data);
execute();
synchronize();
```

This allows the benchmark runner to measure different implementations without embedding hardware-specific logic.

The synchronous interface prioritizes reproducible completed-operation latency over maximizing asynchronous GPU throughput.

## GPU Kernel Optimization

### Naive Matrix Multiplication

The baseline Metal kernel assigns one output element to each GPU thread.

Each thread computes the dot product between a row of matrix A and a column of matrix B.

Although straightforward, neighboring threads can repeatedly request overlapping input values.

### Tiled Matrix Multiplication

The optimized kernel uses **16×16 threadgroup tiling**.

Threads cooperate to:

1. Load blocks of A and B into threadgroup memory.
2. Synchronize after shared data is loaded.
3. Reuse tile elements across multiple multiply-accumulate operations.
4. Synchronize before overwriting shared tiles.
5. Repeat until the output element is complete.

The implementation handles matrix dimensions that are not multiples of 16 using boundary-safe loads and output writes.

Tiling reduces redundant load requests and improves data reuse, but additional synchronization and indexing introduce overhead. The measured benefit depends on matrix size and execution conditions.

### GPU Execution Speedup

| Dimension | Naive GPU (ms) | Tiled GPU (ms) | Speedup |
|---|---:|---:|---:|
| 64 | 0.046 | 0.019 | 2.44× |
| 128 | 0.088 | 0.035 | **2.49×** |
| 256 | 0.317 | 0.185 | 1.72× |
| 512 | 0.727 | 0.429 | 1.69× |
| 1024 | 6.120 | 3.354 | 1.82× |
| 2048 | 44.344 | 26.975 | 1.64× |
| 4096 | 393.845 | 218.142 | 1.81× |

Values are medians of repeated-run medians, rounded for presentation. The GPU metric measures command-buffer execution duration, not total application latency.

## Profiling and Performance Analysis

HeteroBench uses **Xcode Instruments** to investigate Metal execution behavior.

Profiling and uninstrumented benchmark measurements revealed:

- A substantial host-versus-GPU timing difference for small matrix workloads.
- Faster GPU execution following tiling optimization.
- Persistent gaps between successive GPU compute intervals.
- A serialized submit–wait–readback execution model that motivates future investigation into command submission overhead and batching.

Host-observed latency includes command encoding, submission, waiting, and output readback. GPU command-buffer duration is reported separately.

These observations support investigating host-side overhead, but do not establish exact dispatch overhead, memory bandwidth saturation, or GPU occupancy without additional measurements.

## Building

### Requirements

- Apple Silicon Mac
- macOS with Xcode and Metal development tools
- CMake 3.20 or newer
- C++20-compatible compiler
- Python 3 for experiment automation
- Matplotlib for plotting results

### Build and Test

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/HeteroBench.git
cd HeteroBench
```

Configure and build:

```bash
cmake -S . -B build/release -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build/release
```

Run the test suite:

```bash
ctest --test-dir build/release --output-on-failure
```

The Apple build supports the CPU reference, Accelerate, Metal, tiled Metal, and MPS backends.

GPU validation requires an accessible Metal device.

## Running Benchmarks

### Single Workload

Run CPU matrix multiplication:

```bash
./build/release/heterobench --backend cpu --size 512
```

Run naive Metal:

```bash
./build/release/heterobench --backend metal --size 512
```

Run tiled Metal:

```bash
./build/release/heterobench --backend metal-tiled --size 512
```

Compare all compiled backends:

```bash
./build/release/heterobench --backend all --sizes 64,128,256,512 --runs 3 --iterations 20 --warmup 5 --output results/comparison
```

### Larger Workloads

```bash
python3 scripts/run_experiments.py \
  --binary build/release/heterobench \
  --backend accelerate,metal,metal-tiled,mps \
  --reference accelerate \
  --sizes 1024,2048,4096 \
  --runs 3 \
  --iterations 20 \
  --warmup 5 \
  --output results/large-run
```

The large-workload configuration uses Accelerate as the full-output validation reference, with independent double-precision spot checks. The scalar CPU implementation remains available for smaller reference tests.

### Generate Performance Plots

```bash
python3 scripts/analyze_results.py \
  results/large-run.summary.json \
  --output results/large-run-plots
```

The analysis script produces latency comparisons, throughput plots, and aggregate results.

## Benchmark Methodology

To improve measurement reliability:

- Inputs are deterministic and equivalent across backends.
- Warmup iterations are excluded from measured samples.
- Backends are validated before and after benchmark runs.
- GPU execution is synchronized before host timing stops.
- Allocation, input upload, pipeline creation, and validation are excluded from timed execution.
- Raw samples and summary statistics are preserved.
- Backend order rotates across repetitions to reduce fixed-order bias.
- System configuration and experiment metadata are recorded.

### Limitations

Experiments were conducted on a single Apple M2 Pro system. Thermal conditions, processor frequencies, and background system activity were not controlled.

The workload is square FP32 GEMM rather than complete neural network inference.

Results do not establish hardware peak throughput, power efficiency, or isolated kernel performance for every backend. GPU command-buffer timing and host-observed latency have different measurement boundaries.

## Project Structure

```text
HeteroBench/
├── CMakeLists.txt
├── include/heterobench/
│   ├── backend.hpp
│   ├── workload.hpp
│   ├── benchmark.hpp
│   └── ...
├── src/
│   ├── main.cpp
│   ├── benchmark.cpp
│   ├── cpu_backend.cpp
│   ├── metal_backend.mm
│   ├── accelerate_backend.cpp
│   └── ...
├── shaders/
│   └── matmul.metal
├── tests/
├── scripts/
│   ├── run_experiments.py
│   ├── analyze_results.py
│   └── profile_metal.py
├── experiments/
│   ├── final.samples.csv
│   ├── final.summary.json
│   ├── large.samples.csv
│   ├── large.summary.json
│   └── ...
└── docs/
    ├── RESULTS.md
    ├── LARGE_WORKLOAD_RESULTS.md
    └── WEEK1_WALKTHROUGH.md ... WEEK6_WALKTHROUGH.md
```

The structure above highlights important components rather than listing every source and generated file.

## Future Work

Potential extensions include:

- FP16 matrix multiplication
- Register blocking and additional Metal kernel optimizations
- GPU command batching and asynchronous throughput measurements
- Roofline performance analysis
- More comprehensive hardware profiling
- Core ML inference workloads
- CUDA or ROCm execution backends
- Defensible accelerator energy measurements

## Project Motivation

HeteroBench was developed to explore the relationship between hardware architecture, software implementation, and measured computational performance.

The project emphasizes systems programming, GPU execution, benchmarking methodology, numerical correctness, and evidence-based optimization.

Rather than simply comparing benchmark scores, the goal is to understand **why different execution backends exhibit different performance characteristics and how software design can improve hardware utilization.**
