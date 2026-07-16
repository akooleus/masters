#include "cascade_fir.h"

#include "test_utils.h"

#include <string>
#include <vector>

int main()
{
    struct Case {
        const char* label;
        FilterSpec spec;
        bool smooth_transition;
        unsigned maximum_expected_block_order;
        double maximum_impulse_error;
        double maximum_signal_error;
        bool high_precision_runtime;
    };

    // Native precision is preferred through N257T. N513T verifies the
    // explicitly reported double-double short-block backend.
    const std::vector<Case> cases = {
        {"N32T",  {32,  48000.0, 8000.0, 12000.0}, true,  4,  1e-11, 1e-10, false},
        {"N33",   {33,  32000.0, 4000.0,  6000.0}, false, 4,  1e-11, 1e-10, false},
        {"N65T",  {65,  48000.0, 8000.0, 12000.0}, true,  4,  1e-11, 1e-10, false},
        {"N129",  {129, 48000.0, 8000.0, 12000.0}, false, 4,  1e-10, 1e-9, false},
        {"N129T", {129, 48000.0, 8000.0, 12000.0}, true,  4,  1e-9,  1e-8, false},
        {"N257T", {257, 48000.0, 8000.0, 12000.0}, true,  16, 1e-8,  1e-7, false},
        {"N513T", {513, 48000.0, 8000.0, 12000.0}, true,  8,  1e-9,  1e-8, true},
        {"N129W", {129, 48000.0, 20000.0, 22000.0}, true, 4, 1e-9, 1e-8, false},
    };

    bool ok = true;
    for (const Case& tc : cases) {
        const std::vector<real_t> transition = tc.smooth_transition
            ? cosine_transition(tc.spec) : std::vector<real_t>{};
        const DirectFIR fir = design_freq_sampling(tc.spec, transition);
        const CascadeDecomposition full = decompose_exact_fs_full(fir, true);
        const std::string prefix = std::string(tc.label) + ": ";

        ok &= expect(remainder_degree(full) == 0,
                     prefix + "full factorization left a residual polynomial");
        const CascadeBuildStatus expected_status = tc.high_precision_runtime
            ? CascadeBuildStatus::HighPrecisionShortBlockCascade
            : CascadeBuildStatus::ShortBlockCascade;
        ok &= expect(full.diagnostics.status == expected_status,
                     prefix + "expected a verified short-block cascade");
        ok &= expect(full.diagnostics.runtime_decimal_digits
                         == (tc.high_precision_runtime ? 31u : 0u),
                     prefix + "runtime precision selection mismatch");
        ok &= expect(full.runtime_precision
                         == (tc.high_precision_runtime
                             ? CascadeRuntimePrecision::DoubleDouble
                             : CascadeRuntimePrecision::Native),
                     prefix + "runtime backend selection mismatch");
        ok &= expect(full.diagnostics.complement_verified,
                     prefix + "complementary identity was not verified");
        ok &= expect(full.diagnostics.roots_verified,
                     prefix + "root set was not verified");
        ok &= expect(full.diagnostics.runtime_verified,
                     prefix + "runtime cascade was not verified");
        ok &= expect(full.zeros_extracted() == fir.order(),
                     prefix + "full factorization did not cover every root");
        ok &= expect(!full.blocks.empty(),
                     prefix + "runtime cascade has no blocks");
        ok &= expect(full.execution_order.size() == full.blocks.size(),
                     prefix + "execution order does not cover every block");
        ok &= expect(maximum_block_order(full)
                         <= tc.maximum_expected_block_order,
                     prefix + "adaptive block order exceeded its limit");
        ok &= expect(full.diagnostics.selected_max_block_order
                         == maximum_block_order(full),
                     prefix + "reported block order mismatch");
        ok &= expect(decomposition_is_finite(full),
                     prefix + "decomposition contains NaN or infinity");

        bool impulse_finite = false;
        const double impulse_error = cascade_impulse_error_double(
            fir, full, impulse_finite);
        ok &= expect(impulse_finite,
                     prefix + "impulse response contains NaN or infinity");
        ok &= expect(impulse_error <= tc.maximum_impulse_error,
                     prefix + "impulse max error="
                         + std::to_string(impulse_error));
        bool scalar_impulse_finite = false;
        const double scalar_impulse_error = cascade_impulse_error_double(
            fir, full, scalar_impulse_finite,
            CascadeRuntimeOptions{false, CascadeRuntimeKernel::Scalar});
        ok &= expect(scalar_impulse_finite
                         && scalar_impulse_error <= tc.maximum_impulse_error,
                     prefix + "scalar impulse backend failed");
        ok &= expect(std::abs(scalar_impulse_error
                              - full.diagnostics.runtime_impulse_error)
                         <= 1e-18,
                     prefix + "reported scalar impulse error mismatch");

        bool signal_finite = false;
        const double signal_error = cascade_signal_error_double(
            fir, full, signal_finite,
            tc.high_precision_runtime ? 64u : 0u);
        ok &= expect(signal_finite,
                     prefix + "bounded-signal output contains NaN or infinity");
        ok &= expect(signal_error <= tc.maximum_signal_error,
                     prefix + "bounded-signal max error="
                         + std::to_string(signal_error));

        if (tc.high_precision_runtime) {
            CascadeFilterState auto_state;
            auto_state.init(full);
            const CascadeRuntimeKernel expected_kernel =
                cascade_runtime_kernel_available(
                    CascadeRuntimeKernel::Avx2Fma)
                ? CascadeRuntimeKernel::Avx2Fma
                : CascadeRuntimeKernel::Scalar;
            ok &= expect(auto_state.selected_kernel == expected_kernel,
                         prefix + "runtime ISA dispatch mismatch");

            std::vector<sample_t> short_impulse(32u, 0.0f);
            short_impulse[0] = 1.0f;
            const CompareMetrics float_impulse = compare_signals(
                filter_direct(fir, short_impulse),
                filter_cascade(full, short_impulse));
            ok &= expect(float_impulse.max_abs_err <= 1e-6,
                         prefix + "float API did not use the verified backend");

            CascadeDecomposition fallback_precision = full;
            fallback_precision.runtime_precision =
                CascadeRuntimePrecision::Extended50;
            const std::vector<double> probe(32u, 0.25);
            const std::vector<double> fallback_output =
                filter_cascade_double(fallback_precision, probe);
            ok &= expect(std::all_of(
                             fallback_output.begin(), fallback_output.end(),
                             [](double value) { return std::isfinite(value); }),
                         prefix + "50-digit fallback backend is not finite");
        }
    }

    // If h[n] no longer matches the stored frequency samples, the factorized
    // candidates cannot reproduce the requested direct filter. The public full
    // path must then return an explicit, verified direct-form fallback.
    DirectFIR inconsistent = design_freq_sampling(
        {33, 32000.0, 4000.0, 6000.0});
    inconsistent.h.front() += 1e-3;
    const CascadeDecomposition fallback =
        decompose_exact_fs_full(inconsistent, true);
    bool fallback_finite = false;
    const double fallback_error = cascade_impulse_error_double(
        inconsistent, fallback, fallback_finite);
    ok &= expect(fallback.diagnostics.status
                     == CascadeBuildStatus::DirectFormFallback,
                 "inconsistent input should use direct-form fallback");
    ok &= expect(fallback.diagnostics.runtime_verified && fallback_finite,
                 "direct-form fallback should be finite and verified");
    ok &= expect(fallback_error == 0.0,
                 "direct-form fallback should preserve h[n] exactly");

    return ok ? 0 : 1;
}
