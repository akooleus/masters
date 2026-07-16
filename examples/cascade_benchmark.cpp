#include "cascade_fir.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct Case {
    const char* label;
    FilterSpec spec;
    size_t cascade_samples;
};

struct Timing {
    double median_ns_per_sample = 0.0;
    double checksum = 0.0;
    long double peak_internal = 0.0L;
};

std::vector<real_t> cosine_transition(const FilterSpec& spec)
{
    const unsigned count = spec.k_stop() - spec.k_pass() - 1;
    std::vector<real_t> transition(count, 0.0);
    for (unsigned i = 0; i < count; ++i) {
        const double phase = PI * static_cast<double>(i + 1)
                           / static_cast<double>(count + 1);
        transition[i] = 0.5 * (1.0 + std::cos(phase));
    }
    return transition;
}

std::vector<double> make_input(size_t length)
{
    std::vector<double> input(length, 0.0);
    uint64_t state = 0x9e3779b97f4a7c15ULL;
    for (size_t i = 0; i < length; ++i) {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        const double noise = static_cast<double>(state >> 11)
                           * (1.0 / 9007199254740992.0) - 0.5;
        input[i] = 0.45 * std::sin(0.013 * static_cast<double>(i))
                 + 0.20 * std::cos(0.071 * static_cast<double>(i))
                 + 0.10 * noise;
    }
    return input;
}

double median(std::vector<double> values)
{
    std::sort(values.begin(), values.end());
    const size_t middle = values.size() / 2;
    if (values.size() % 2 != 0) {
        return values[middle];
    }
    return 0.5 * (values[middle - 1] + values[middle]);
}

Timing benchmark_direct(const DirectFIR& fir,
                        const std::vector<double>& input,
                        size_t warmup,
                        size_t samples,
                        unsigned repeats)
{
    std::vector<double> timings;
    timings.reserve(repeats);
    Timing result;
    for (unsigned repeat = 0; repeat < repeats; ++repeat) {
        DoubleFilterState state;
        state.init(fir.h);
        for (size_t i = 0; i < warmup; ++i) {
            result.checksum += state.push(input[i]);
        }

        double checksum = 0.0;
        const auto start = Clock::now();
        for (size_t i = 0; i < samples; ++i) {
            checksum += state.push(input[warmup + i]);
        }
        const double nanoseconds =
            std::chrono::duration<double, std::nano>(Clock::now() - start).count();
        timings.push_back(nanoseconds / static_cast<double>(samples));
        result.checksum += checksum;
    }
    result.median_ns_per_sample = median(std::move(timings));
    return result;
}

Timing benchmark_cascade(const CascadeDecomposition& dec,
                         const std::vector<double>& input,
                         size_t warmup,
                         size_t samples,
                         unsigned repeats,
                         bool track_peak,
                         CascadeRuntimeKernel kernel)
{
    std::vector<double> timings;
    timings.reserve(repeats);
    Timing result;
    for (unsigned repeat = 0; repeat < repeats; ++repeat) {
        CascadeFilterState state;
        state.init(dec, CascadeRuntimeOptions{
            track_peak, kernel});
        for (size_t i = 0; i < warmup; ++i) {
            result.checksum += state.push_double(input[i]);
        }

        double checksum = 0.0;
        const auto start = Clock::now();
        for (size_t i = 0; i < samples; ++i) {
            checksum += state.push_double(input[warmup + i]);
        }
        const double nanoseconds =
            std::chrono::duration<double, std::nano>(Clock::now() - start).count();
        timings.push_back(nanoseconds / static_cast<double>(samples));
        result.checksum += checksum;
        result.peak_internal = std::max(result.peak_internal, state.peak_internal);
    }
    result.median_ns_per_sample = median(std::move(timings));
    return result;
}

double maximum_signal_error(const DirectFIR& fir,
                            const CascadeDecomposition& dec,
                            const std::vector<double>& input,
                            size_t samples,
                            CascadeRuntimeKernel kernel)
{
    DoubleFilterState direct;
    direct.init(fir.h);
    CascadeFilterState cascade;
    cascade.init(dec, CascadeRuntimeOptions{false, kernel});

    double maximum = 0.0;
    for (size_t i = 0; i < samples; ++i) {
        maximum = std::max(
            maximum,
            std::abs(direct.push(input[i]) - cascade.push_double(input[i])));
    }
    return maximum;
}

double maximum_impulse_error(const DirectFIR& fir,
                             const CascadeDecomposition& dec,
                             CascadeRuntimeKernel kernel)
{
    CascadeFilterState cascade;
    cascade.init(dec, CascadeRuntimeOptions{false, kernel});
    double maximum = 0.0;
    for (size_t i = 0; i < fir.length() * 2u; ++i) {
        const double output = cascade.push_double(i == 0u ? 1.0 : 0.0);
        const double expected = (i < fir.h.size()) ? fir.h[i] : 0.0;
        maximum = std::max(maximum, std::abs(output - expected));
    }
    return maximum;
}

unsigned maximum_block_order(const CascadeDecomposition& dec)
{
    unsigned result = 0;
    for (const CascadeBlock& block : dec.blocks) {
        if (!block.coefficients.empty()) {
            result = std::max(
                result,
                static_cast<unsigned>(block.coefficients.size() - 1));
        }
    }
    return result;
}

const char* status_name(CascadeBuildStatus status)
{
    switch (status) {
    case CascadeBuildStatus::Unspecified: return "unspecified";
    case CascadeBuildStatus::AnalyticalReduction: return "analytical";
    case CascadeBuildStatus::ShortBlockCascade: return "short";
    case CascadeBuildStatus::HighPrecisionShortBlockCascade: return "short-mp";
    case CascadeBuildStatus::FullOrderFactorBlock: return "full-factor";
    case CascadeBuildStatus::DirectFormFallback: return "direct";
    case CascadeBuildStatus::Failed: return "failed";
    }
    return "unknown";
}

const char* precision_name(CascadeRuntimePrecision precision)
{
    switch (precision) {
    case CascadeRuntimePrecision::Native: return "native";
    case CascadeRuntimePrecision::DoubleDouble: return "double-double";
    case CascadeRuntimePrecision::Extended34: return "extended34";
    case CascadeRuntimePrecision::Extended50: return "extended50";
    }
    return "unknown";
}

const char* kernel_name(CascadeRuntimeKernel kernel)
{
    switch (kernel) {
    case CascadeRuntimeKernel::Auto: return "auto";
    case CascadeRuntimeKernel::Scalar: return "scalar";
    case CascadeRuntimeKernel::Avx2Fma: return "avx2-fma";
    }
    return "unknown";
}

size_t parse_size(const char* value, const char* name)
{
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    if (end == value || *end != '\0' || parsed == 0) {
        std::fprintf(stderr, "invalid %s: %s\n", name, value);
        std::exit(2);
    }
    return static_cast<size_t>(parsed);
}

CascadeRuntimeKernel parse_kernel(const char* value)
{
    const std::string name(value);
    if (name == "auto") {
        return CascadeRuntimeKernel::Auto;
    }
    if (name == "scalar") {
        return CascadeRuntimeKernel::Scalar;
    }
    if (name == "avx2" || name == "avx2-fma") {
        return CascadeRuntimeKernel::Avx2Fma;
    }
    std::fprintf(stderr,
                 "invalid kernel: %s (expected auto, scalar or avx2)\n",
                 value);
    std::exit(2);
}

} // namespace

int main(int argc, char** argv)
{
    const std::vector<Case> cases = {
        {"N257T", {257, 48000.0, 8000.0, 12000.0}, 262144u},
        {"N513T", {513, 48000.0, 8000.0, 12000.0}, 16384u},
    };

    const std::string requested = (argc > 1) ? argv[1] : "N257T";
    const auto selected = std::find_if(
        cases.begin(), cases.end(), [&](const Case& item) {
            return requested == item.label;
        });
    if (selected == cases.end()) {
        std::fprintf(stderr, "unknown case: %s (expected N257T or N513T)\n",
                     requested.c_str());
        return 2;
    }
    if (argc > 6) {
        std::fprintf(stderr,
                     "usage: %s [N257T|N513T] [samples] [repeats] [artifact] [kernel]\n",
                     argv[0]);
        return 2;
    }

    const size_t cascade_samples = (argc > 2)
        ? parse_size(argv[2], "sample count") : selected->cascade_samples;
    const unsigned repeats = static_cast<unsigned>(
        (argc > 3) ? parse_size(argv[3], "repeat count") : 5u);
    const std::string artifact_path = (argc > 4) ? argv[4] : "";
    const CascadeRuntimeKernel requested_kernel = (argc > 5)
        ? parse_kernel(argv[5]) : CascadeRuntimeKernel::Auto;
    const size_t direct_samples = std::max<size_t>(262144u, cascade_samples);
    const size_t warmup = 4096u;
    const size_t input_size = warmup + std::max(direct_samples, cascade_samples);
    const std::vector<double> input = make_input(input_size);

    const auto design_start = Clock::now();
    const DirectFIR fir = design_freq_sampling(
        selected->spec, cosine_transition(selected->spec));
    const double design_ms = std::chrono::duration<double, std::milli>(
        Clock::now() - design_start).count();

    CascadeDecomposition dec;
    double decompose_ms = 0.0;
    double artifact_save_ms = 0.0;
    double artifact_load_ms = 0.0;
    std::size_t artifact_bytes = 0u;
    const char* artifact_mode = "disabled";
    try {
        bool artifact_exists = false;
        if (!artifact_path.empty()) {
            std::ifstream probe(artifact_path, std::ios::binary);
            artifact_exists = probe.good();
        }

        if (artifact_exists) {
            const auto load_start = Clock::now();
            CascadeArtifact artifact = load_cascade_artifact(artifact_path);
            artifact_load_ms = std::chrono::duration<double, std::milli>(
                Clock::now() - load_start).count();
            if (!cascade_artifact_matches(artifact, fir)) {
                throw std::runtime_error(
                    "artifact does not match the requested source FIR");
            }
            dec = std::move(artifact.decomposition);
            artifact_mode = "loaded";
        } else {
            const auto decompose_start = Clock::now();
            dec = decompose_exact_fs_full(fir, true);
            decompose_ms = std::chrono::duration<double, std::milli>(
                Clock::now() - decompose_start).count();
            if (!artifact_path.empty()) {
                const CascadeArtifact artifact =
                    make_cascade_artifact(fir, dec);
                const auto save_start = Clock::now();
                save_cascade_artifact(artifact_path, artifact);
                artifact_save_ms = std::chrono::duration<double, std::milli>(
                    Clock::now() - save_start).count();

                const auto load_start = Clock::now();
                CascadeArtifact loaded = load_cascade_artifact(artifact_path);
                artifact_load_ms = std::chrono::duration<double, std::milli>(
                    Clock::now() - load_start).count();
                if (!cascade_artifact_matches(loaded, fir)) {
                    throw std::runtime_error(
                        "new artifact does not match the source FIR");
                }
                dec = std::move(loaded.decomposition);
                artifact_mode = "built";
            }
        }

        if (!artifact_path.empty()) {
            std::ifstream size_probe(
                artifact_path, std::ios::binary | std::ios::ate);
            const std::streamoff size = size_probe.tellg();
            if (size >= 0) {
                artifact_bytes = static_cast<std::size_t>(size);
            }
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "artifact error: %s\n", error.what());
        return 2;
    }

    const auto direct_init_start = Clock::now();
    DoubleFilterState direct_init_probe;
    direct_init_probe.init(fir.h);
    const double direct_init_us = std::chrono::duration<double, std::micro>(
        Clock::now() - direct_init_start).count();

    const auto cascade_init_start = Clock::now();
    CascadeFilterState cascade_init_probe;
    try {
        cascade_init_probe.init(
            dec, CascadeRuntimeOptions{false, requested_kernel});
    } catch (const std::exception& error) {
        std::fprintf(stderr, "runtime kernel error: %s\n", error.what());
        return 2;
    }
    const double cascade_init_us = std::chrono::duration<double, std::micro>(
        Clock::now() - cascade_init_start).count();

    const Timing direct = benchmark_direct(
        fir, input, warmup, direct_samples, repeats);
    const Timing cascade = benchmark_cascade(
        dec, input, warmup, cascade_samples, repeats,
        false, requested_kernel);
    const Timing cascade_tracked = benchmark_cascade(
        dec, input, warmup, cascade_samples, repeats,
        true, requested_kernel);
    const size_t error_samples = std::min<size_t>(4096u, cascade_samples);
    const double signal_error = maximum_signal_error(
        fir, dec, input, error_samples, requested_kernel);
    const double impulse_error = maximum_impulse_error(
        fir, dec, requested_kernel);

    const double direct_msamples = 1000.0 / direct.median_ns_per_sample;
    const double cascade_msamples = 1000.0 / cascade.median_ns_per_sample;
    const double realtime_factor = cascade_msamples * 1e6 / selected->spec.fs;

    std::printf("case=%s N=%u status=%s blocks=%zu max_order=%u precision=%s digits=%u kernel=%s verified_impulse_error=%.6e\n",
                selected->label, selected->spec.N,
                status_name(dec.diagnostics.status), dec.blocks.size(),
                maximum_block_order(dec),
                precision_name(dec.runtime_precision),
                dec.diagnostics.runtime_decimal_digits,
                kernel_name(cascade_init_probe.selected_kernel),
                dec.diagnostics.runtime_impulse_error);
    std::printf("design_ms=%.3f decompose_ms=%.3f direct_init_us=%.3f cascade_init_us=%.3f\n",
                design_ms, decompose_ms, direct_init_us, cascade_init_us);
    std::printf("artifact_mode=%s artifact_bytes=%zu save_ms=%.3f load_ms=%.3f\n",
                artifact_mode, artifact_bytes,
                artifact_save_ms, artifact_load_ms);
    std::printf("direct_samples=%zu direct_ns_per_sample=%.3f direct_Msample_s=%.6f\n",
                direct_samples, direct.median_ns_per_sample, direct_msamples);
    std::printf("cascade_samples=%zu cascade_ns_per_sample=%.3f cascade_Msample_s=%.6f realtime_x=%.3f\n",
                cascade_samples, cascade.median_ns_per_sample,
                cascade_msamples, realtime_factor);
    std::printf("cascade_tracked_ns_per_sample=%.3f peak_tracking_overhead_x=%.3f runtime_peak=%.6Le\n",
                cascade_tracked.median_ns_per_sample,
                cascade_tracked.median_ns_per_sample
                    / cascade.median_ns_per_sample,
                cascade_tracked.peak_internal);
    std::printf("error_samples=%zu max_impulse_error=%.6e max_signal_error=%.6e checksum=%.12e\n",
                error_samples, impulse_error, signal_error,
                direct.checksum + cascade.checksum);
    return (std::isfinite(impulse_error) && std::isfinite(signal_error))
        ? 0 : 1;
}
