#include "heterobench/results.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace heterobench {
namespace {

std::string json_string(const std::string& value) {
    std::ostringstream out;
    out << '"';
    for (unsigned char character : value) {
        switch (character) {
            case '"': out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (character < 0x20) {
                    out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                        << static_cast<unsigned int>(character) << std::dec;
                } else out << character;
        }
    }
    out << '"';
    return out.str();
}

std::string csv_string(const std::string& value) {
    std::string out = "\"";
    for (char character : value) {
        if (character == '"') out += '"';
        out += character;
    }
    return out + '"';
}

void write_stats(std::ostream& out, const LatencyStats& stats) {
    out << "{\"min_ms\":" << stats.min_ms << ",\"median_ms\":" << stats.median_ms
        << ",\"mean_ms\":" << stats.mean_ms << ",\"p95_ms\":" << stats.p95_ms
        << ",\"variance_ms2\":" << stats.variance_ms2 << ",\"stddev_ms\":" << stats.stddev_ms << '}';
}



} // namespace


void write_results(const std::string& prefix, const Metadata& metadata,
                   const std::vector<ExperimentResult>& results) {
    if (prefix.empty() || results.empty()) throw std::invalid_argument("Output and results must be nonempty");
    const std::filesystem::path csv_path = prefix + ".samples.csv";
    const std::filesystem::path json_path = prefix + ".summary.json";
    if (std::filesystem::exists(csv_path) || std::filesystem::exists(json_path)) {
        throw std::runtime_error("Result files already exist; choose a new --output prefix");
    }
    // Validate before creating output so malformed samples cannot leave misleading results.
    for (const auto& result : results) {
        if (result.samples.size() != result.config.measurement_iterations) {
            throw std::invalid_argument("Result sample count does not match configuration");
        }
        std::vector<double> host, gpu;
        for (const auto& sample : result.samples) {
            host.push_back(sample.host_ms);
            if (sample.gpu_ms) gpu.push_back(*sample.gpu_ms);
        }
        summarize_latencies(host);
        if (!gpu.empty()) summarize_latencies(gpu);
    }
    if (!csv_path.parent_path().empty()) std::filesystem::create_directories(csv_path.parent_path());
    std::ofstream csv(csv_path);
    std::ofstream json(json_path);
    if (!csv || !json) throw std::runtime_error("Cannot open result output files");
    csv << std::setprecision(17);
    json << std::setprecision(17);
    csv << "backend,device,dimension,repetition,iteration,host_ms,gpu_ms\n";
    json << "{\n\"schema_version\":1,\n\"metadata\":{";
    bool first = true;
    for (const auto& [key, value] : metadata) {
        if (!first) json << ',';
        first = false;
        json << json_string(key) << ':' << json_string(value);
    }
    json << "},\n\"experiments\":[\n";
    for (std::size_t index = 0; index < results.size(); ++index) {
        const auto& result = results[index];
        std::vector<double> host, gpu;
        for (std::size_t sample_index = 0; sample_index < result.samples.size(); ++sample_index) {
            const auto& sample = result.samples[sample_index];
            host.push_back(sample.host_ms);
            if (sample.gpu_ms) gpu.push_back(*sample.gpu_ms);
            csv << csv_string(result.backend) << ',' << csv_string(result.device) << ','
                << result.dimension << ',' << result.repetition << ',' << sample_index << ',' << sample.host_ms << ',';
            if (sample.gpu_ms) csv << *sample.gpu_ms;
            csv << '\n';
        }
        if (index != 0) json << ",\n";
        const auto host_stats = summarize_latencies(host);
        const double operations = 2.0 * result.dimension * result.dimension * result.dimension;
        json << "{\"backend\":" << json_string(result.backend) << ",\"device\":" << json_string(result.device)
             << ",\"dimension\":" << result.dimension << ",\"repetition\":" << result.repetition
             << ",\"warmup_iterations\":" << result.config.warmup_iterations
             << ",\"measurement_iterations\":" << result.config.measurement_iterations
             << ",\"gpu_sample_count\":" << gpu.size() << ",\"host\":";
        write_stats(json, host_stats);
        json << ",\"host_gflops_from_median\":";
        if (host_stats.median_ms > 0.0) json << operations / (host_stats.median_ms * 1.0e6);
        else json << "null";
        json << ",\"gpu\":";
        if (!gpu.empty()) write_stats(json, summarize_latencies(gpu));
        else json << "null";
        json << ",\"gpu_gflops_from_median\":";
        if (!gpu.empty() && summarize_latencies(gpu).median_ms > 0.0) {
            json << operations / (summarize_latencies(gpu).median_ms * 1.0e6);
        } else json << "null";
        json << '}';
    }
    json << "\n]}\n";
    csv.close();
    json.close();
    if (!csv || !json) throw std::runtime_error("Failed while writing result files");
}

} // namespace heterobench
