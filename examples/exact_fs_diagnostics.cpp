#include "cascade_fir.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {

unsigned count_known_zeros(const DirectFIR& fir)
{
    unsigned count = 0;
    for (unsigned k = 0; k <= fir.length() / 2; ++k) {
        if (fir.frequency_samples[k] != 0.0) {
            continue;
        }
        if (k == 0 || (fir.length() % 2 == 0 && k == fir.length() / 2)) {
            ++count;
        } else {
            count += 2;
        }
    }
    return count;
}

double impulse_max_error(const DirectFIR& fir,
                         const CascadeDecomposition& dec,
                         double& output_sum,
                         bool& finite)
{
    std::vector<double> impulse(fir.length() * 2u, 0.0);
    impulse[0] = 1.0;
    const std::vector<double> output = filter_cascade_double(dec, impulse);

    double max_error = 0.0;
    output_sum = 0.0;
    finite = true;
    for (size_t i = 0; i < output.size(); ++i) {
        finite = finite && std::isfinite(output[i]);
        output_sum += output[i];
        const double reference = (i < fir.h.size()) ? fir.h[i] : 0.0;
        max_error = std::max(max_error, std::abs(output[i] - reference));
    }
    return max_error;
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
    case CascadeBuildStatus::Unspecified:          return "unspecified";
    case CascadeBuildStatus::AnalyticalReduction: return "analytical";
    case CascadeBuildStatus::ShortBlockCascade:   return "short";
    case CascadeBuildStatus::HighPrecisionShortBlockCascade:
        return "short-mp";
    case CascadeBuildStatus::FullOrderFactorBlock:return "full-factor";
    case CascadeBuildStatus::DirectFormFallback:  return "direct";
    case CascadeBuildStatus::Failed:              return "failed";
    }
    return "unknown";
}

} // namespace

int main(int argc, char** argv)
{
    struct Case {
        const char* label;
        FilterSpec spec;
        bool smooth_transition;
    };

    const std::vector<Case> cases = {
        {"N33",   {33, 32000.0, 4000.0,  6000.0}, false},
        {"N49",   {49, 32000.0, 4000.0,  6000.0}, false},
        {"N65",   {65, 48000.0, 8000.0, 12000.0}, false},
        {"N97",   {97, 48000.0, 8000.0, 12000.0}, false},
        {"N129",  {129,48000.0, 8000.0, 12000.0}, false},
        {"N65T",  {65, 48000.0, 8000.0, 12000.0}, true},
        {"N129T", {129,48000.0, 8000.0, 12000.0}, true},
        {"N257T", {257,48000.0, 8000.0, 12000.0}, true},
        {"N513T", {513,48000.0, 8000.0, 12000.0}, true},
    };

    std::printf("Complement-FS diagnostics\n\n");
    std::printf("%-6s %5s %8s %8s %8s %8s %6s %7s %6s %12s %10s %12s %7s %-11s\n",
                "case", "N", "known_z", "stage_z", "stage_r", "full_r",
                "blocks", "max_ord", "digits", "imp_maxerr", "filter_ms",
                "sum_y", "finite", "status");

    for (const auto& tc : cases) {
        if (argc > 1 && std::string(argv[1]) != tc.label) {
            continue;
        }
        std::vector<real_t> transition;
        if (tc.smooth_transition) {
            const unsigned count = tc.spec.k_stop() - tc.spec.k_pass() - 1;
            transition.resize(count);
            for (unsigned i = 0; i < count; ++i) {
                const double phase = PI * static_cast<double>(i + 1)
                                   / static_cast<double>(count + 1);
                transition[i] = 0.5 * (1.0 + std::cos(phase));
            }
        }
        const DirectFIR fir = design_freq_sampling(tc.spec, transition);
        const CascadeDecomposition stage = decompose_exact_fs(fir, true);
        const CascadeDecomposition full = decompose_exact_fs_full(fir, true);
        const unsigned stage_degree = stage.remainder.empty()
            ? 0u : static_cast<unsigned>(stage.remainder.size() - 1);
        const unsigned full_degree = full.remainder.empty()
            ? 0u : static_cast<unsigned>(full.remainder.size() - 1);
        const unsigned blocks = static_cast<unsigned>(
            full.first_order.size() + full.biquads.size()
            + full.quartics.size() + full.blocks.size());
        double output_sum = 0.0;
        bool finite = false;
        const auto filter_start = std::chrono::steady_clock::now();
        const double impulse_error = impulse_max_error(
            fir, full, output_sum, finite);
        const double filter_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - filter_start).count();

        std::printf("%-6s %5u %8u %8u %8u %8u %6u %7u %6u %12.4e %10.2f %12.4e %7s %-11s\n",
                    tc.label, tc.spec.N, count_known_zeros(fir),
                    stage.zeros_extracted(), stage_degree, full_degree,
                    blocks, maximum_block_order(full),
                    full.diagnostics.runtime_decimal_digits, impulse_error,
                    filter_ms, output_sum, finite ? "yes" : "NO",
                    status_name(full.diagnostics.status));
    }
    return 0;
}
