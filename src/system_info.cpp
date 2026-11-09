#include "heterobench/results.hpp"

#include <cstdint>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <thread>
#ifdef __APPLE__
#include <sys/sysctl.h>
#endif
#if defined(__unix__) || defined(__APPLE__)
#include <sys/utsname.h>
#endif

namespace heterobench {
namespace {
#ifdef __APPLE__
std::string system_string(const char* key) {
    std::size_t size = 0;
    if (sysctlbyname(key, nullptr, &size, nullptr, 0) != 0 || size == 0) return "unavailable";
    std::string value(size, '\0');
    if (sysctlbyname(key, value.data(), &size, nullptr, 0) != 0) return "unavailable";
    while (!value.empty() && value.back() == '\0') value.pop_back();
    return value;
}
#endif
} // namespace

Metadata collect_metadata() {
    Metadata metadata;
    const std::time_t now = std::time(nullptr);
    std::ostringstream timestamp;
    timestamp << std::put_time(std::gmtime(&now), "%Y-%m-%dT%H:%M:%SZ");
    metadata["timestamp_utc"] = timestamp.str();
#ifdef __VERSION__
    metadata["compiler"] = __VERSION__;
#else
    metadata["compiler"] = HETEROBENCH_COMPILER;
#endif
    metadata["configured_compiler"] = HETEROBENCH_COMPILER;
    metadata["build_type"] = HETEROBENCH_BUILD_TYPE;
    metadata["cxx_flags"] = HETEROBENCH_BUILD_FLAGS;
    metadata["git_revision"] = HETEROBENCH_GIT_REVISION;
    metadata["source_state_at_configure"] = HETEROBENCH_SOURCE_STATE;
    metadata["logical_threads"] = std::to_string(std::thread::hardware_concurrency());
    metadata["inputs"] = "deterministic modular FP32 values; same inputs for every backend";
    metadata["timing"] = "host=execute+synchronize+readback; gpu=completed command-buffer duration";
    metadata["setup"] = "allocation, input upload, library/pipeline creation and validation excluded";
    metadata["thermal_conditions"] = "not controlled or sampled; no power attribution";
    metadata["backend_order"] = "rotated by repetition; dimensions in requested order";
#if defined(__unix__) || defined(__APPLE__)
    utsname system{};
    if (uname(&system) == 0) {
        metadata["os"] = std::string(system.sysname) + " " + system.release;
        metadata["architecture"] = system.machine;
    }
#endif
#ifdef __APPLE__
    metadata["os_version"] = system_string("kern.osproductversion");
    metadata["hardware_model"] = system_string("hw.model");
    metadata["cpu"] = system_string("machdep.cpu.brand_string");
    std::uint64_t memory = 0;
    std::size_t bytes = sizeof(memory);
    if (sysctlbyname("hw.memsize", &memory, &bytes, nullptr, 0) == 0) {
        metadata["memory_bytes"] = std::to_string(memory);
    }
#endif
    return metadata;
}

} // namespace heterobench
