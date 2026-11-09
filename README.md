# HeteroBench

A C++20 FP32 matrix-multiplication benchmark and performance-engineering learning project for Apple Silicon. The initial six-week implementation includes scalar CPU, Accelerate, naive Metal, tiled Metal, and MPS backends; separate host and GPU command-buffer timing; CSV/JSON export; repeatable sweeps; analysis plots; tests; and Instruments workflows.

Start with the [measured findings](docs/RESULTS.md). Results describe this machine and experiment, not a universal speedup or full AI inference system.

## Build and test

Install full Xcode and the Metal toolchain. From the repository root:

```sh
cmake -S . -B build/final -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build/final
ctest --test-dir build/final --output-on-failure
./build/final/metal_smoke
./build/final/heterobench --backend all --size 64
```

`RelWithDebInfo` is optimized and includes symbols for profiling. On Apple Silicon with Python available, expect five passing CTest entries. A skipped GPU test is not GPU verification. `--help` lists options. Defaults remain CPU and size 2.

Backends are `cpu`, `accelerate`, `metal`, `metal-tiled`, and `mps`. `both` means CPU and naive Metal; `all` means every compiled backend. Each result is validated before and after measured execution.

## Run and analyze an experiment

```sh
python3 scripts/run_experiments.py --binary build/final/heterobench --backend all --sizes 64,128,256,512 --runs 3 --iterations 20 --output results/my-run
python3 scripts/analyze_results.py results/my-run.summary.json --output results/my-run-plots
```

For larger matrices, exclude the scalar backend from the timed selection and explicitly select a fast validation reference:

```sh
python3 scripts/run_experiments.py --binary build/final/heterobench --backend accelerate,metal,metal-tiled,mps --reference accelerate --sizes 1024,2048,4096 --runs 3 --iterations 20 --output results/large-run
```

The default reference remains `cpu`. `--reference accelerate` compares complete outputs against the optimized library and independently checks 16 reference coordinates with double accumulation. It does not turn Accelerate's own full-matrix output into an independent reference for itself. The known small tests remain important.

Use a new output prefix; existing files are refused. The runner saves samples, summaries, stdout, exact argv, executable hash, and source hashes. Plotting uses Matplotlib; the C++ build and Python export tests require no plotting library. Curated results and figures are in `experiments/`; scratch output and large traces in `results/` are ignored.

Host latency includes execution, waiting, and GPU output readback. GPU duration is the completed command-buffer interval, not isolated kernel time. Input uploads, pipeline creation, validation, and export are excluded. Missing GPU durations are unavailable, never fabricated zero timings. Read the [methodology](docs/WEEK3_WALKTHROUGH.md) before interpreting throughput or ratios.

## Profile separately

```sh
python3 scripts/profile_metal.py --binary build/final/heterobench --backend metal --size 512 --iterations 300 --seconds 20 --output results/profiling/naive.trace
python3 scripts/profile_metal.py --binary build/final/heterobench --backend metal-tiled --size 512 --iterations 300 --seconds 20 --output results/profiling/tiled.trace
```

Verify target activity and completion before accepting a capture. Never merge instrumented timings into uninstrumented comparison samples. See the [profiling walkthrough](docs/WEEK5_WALKTHROUGH.md) for exports, filtering, and caveats.

## CPU only

```sh
cmake -S . -B build/cpu-only -DCMAKE_BUILD_TYPE=Release -DHETEROBENCH_ENABLE_METAL=OFF -DHETEROBENCH_ENABLE_ACCELERATE=OFF
cmake --build build/cpu-only
ctest --test-dir build/cpu-only --output-on-failure
./build/cpu-only/heterobench --backend cpu --size 64
```

This path does not enable Objective-C++, compile shaders, or link Apple compute frameworks. It registers three tests if Python is available; explicit requests for unavailable backends fail.

## Walkthroughs

| Week | Guide |
| --- | --- |
| 1 | [C++ foundation and architecture](docs/WEEK1_WALKTHROUGH.md) |
| 2 | [Metal lifecycle and ownership](docs/WEEK2_WALKTHROUGH.md) |
| 3 | [Timing, statistics, export, and reproducibility](docs/WEEK3_WALKTHROUGH.md) |
| 4 | [Tiling, barriers, and optimized references](docs/WEEK4_WALKTHROUGH.md) |
| 5 | [Instruments profiling and evidence](docs/WEEK5_WALKTHROUGH.md) |
| 6 | [Results, reproduction, and portfolio review](docs/WEEK6_WALKTHROUGH.md) |

The earlier walkthroughs describe their milestone snapshots. Current commands and test counts are above. The [project context](PROJECTCONTEXT.md) records priorities, progress, and deferred stretch goals.

## Troubleshooting and limits

- Missing Metal compiler: select full Xcode, then run `xcodebuild -downloadComponent MetalToolchain`. Shader compilation needs access to its module cache.
- GPU unavailable: run in a local terminal with GPU access. Restricted environments can compile successfully yet skip GPU execution.
- New Xcode/compiler: use a fresh build directory to refresh CMake detection. Export also records the actual compiler's built-in version string.
- Shader library: CMake embeds a build-tree path. Keep the generated library in place; standalone installation/packaging is not implemented.
- Profiling: xctrace needs its normal platform permissions and cache access. A saved trace without target events is not a successful capture.
- Thermal state and background activity are not controlled. Roofline, FP16, dispatch batching, additional kernels, power attribution, and full inference models remain deferred.
