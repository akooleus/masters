#ifndef CASCADE_FIR_TEST_UTILS_H
#define CASCADE_FIR_TEST_UTILS_H

#include "cascade_fir.h"

#include <cmath>
#include <algorithm>
#include <iostream>
#include <limits>
#include <string>
#include <cstdint>
#include <vector>

inline bool expect(bool condition, const std::string& message)
{
    if (!condition) {
        std::cerr << "[FAIL] " << message << '\n';
        return false;
    }
    return true;
}

inline FilterSpec smoke_spec()
{
    return FilterSpec{33, 32000.0, 4000.0, 6000.0};
}

inline DirectFIR smoke_fir()
{
    return design_freq_sampling(smoke_spec());
}

inline std::vector<real_t> cosine_transition(const FilterSpec& spec)
{
    const unsigned first = spec.k_pass() + 1;
    const unsigned last = spec.k_stop();
    if (last <= first) {
        return {};
    }

    const unsigned count = last - first;
    std::vector<real_t> transition(count, 0.0);
    for (unsigned i = 0; i < count; ++i) {
        const double phase = PI * static_cast<double>(i + 1)
                           / static_cast<double>(count + 1);
        transition[i] = 0.5 * (1.0 + std::cos(phase));
    }
    return transition;
}

inline unsigned count_known_fs_zeros(const DirectFIR& fir)
{
    if (fir.frequency_samples.size() != fir.length()) {
        return 0;
    }

    unsigned count = 0;
    for (unsigned k = 0; k <= fir.length() / 2; ++k) {
        if (fir.frequency_samples[k] != 0.0) {
            continue;
        }
        if (k == 0
            || (fir.length() % 2 == 0 && k == fir.length() / 2)) {
            ++count;
        } else {
            count += 2;
        }
    }
    return count;
}

inline unsigned remainder_degree(const CascadeDecomposition& dec)
{
    return dec.remainder.empty()
        ? 0u : static_cast<unsigned>(dec.remainder.size() - 1);
}

inline unsigned maximum_block_order(const CascadeDecomposition& dec)
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

inline bool decomposition_is_finite(const CascadeDecomposition& dec)
{
    if (!std::isfinite(dec.gain) || !std::isfinite(dec.runtime_gain)) {
        return false;
    }
    for (const CascadeBlock& block : dec.blocks) {
        for (double coefficient : block.coefficients) {
            if (!std::isfinite(coefficient)) {
                return false;
            }
        }
    }
    for (double coefficient : dec.remainder) {
        if (!std::isfinite(coefficient)) {
            return false;
        }
    }
    for (const CascadeSectionPlacement& placement : dec.execution_order) {
        if (!std::isfinite(placement.scale)) {
            return false;
        }
    }
    return true;
}

inline double cascade_impulse_error_double(
    const DirectFIR& fir,
    const CascadeDecomposition& dec,
    bool& finite)
{
    std::vector<double> impulse(fir.length() * 2u, 0.0);
    impulse[0] = 1.0;
    const std::vector<double> actual = filter_cascade_double(dec, impulse);

    finite = true;
    double max_error = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
        finite = finite && std::isfinite(actual[i]);
        const double expected = (i < fir.h.size()) ? fir.h[i] : 0.0;
        max_error = std::max(max_error, std::abs(actual[i] - expected));
    }
    return finite ? max_error : std::numeric_limits<double>::infinity();
}

inline double cascade_signal_error_double(
    const DirectFIR& fir,
    const CascadeDecomposition& dec,
    bool& finite,
    size_t num_samples = 0)
{
    if (num_samples == 0) {
        num_samples = fir.length() * 3u;
    }
    std::vector<double> input(num_samples, 0.0);
    for (size_t n = 0; n < input.size(); ++n) {
        input[n] = 0.35 * std::sin(0.071 * static_cast<double>(n))
                 + 0.20 * std::cos(0.319 * static_cast<double>(n))
                 + 0.05 * std::sin(1.713 * static_cast<double>(n));
    }

    const std::vector<double> expected = filter_direct_double(fir, input);
    const std::vector<double> actual = filter_cascade_double(dec, input);
    finite = true;
    double max_error = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
        finite = finite && std::isfinite(expected[i])
                        && std::isfinite(actual[i]);
        max_error = std::max(max_error, std::abs(actual[i] - expected[i]));
    }
    return finite ? max_error : std::numeric_limits<double>::infinity();
}

inline bool expect_coeff_error(const std::vector<real_t>& lhs,
                               const std::vector<real_t>& rhs,
                               double max_allowed,
                               const std::string& label)
{
    const CoeffMetrics metrics = compare_coeffs(lhs, rhs);
    return expect(metrics.max_abs_err <= max_allowed,
                  label + " max_abs_err=" + std::to_string(metrics.max_abs_err));
}

inline double compute_rebuild_err(const DirectFIR& fir,
                                  const CascadeDecomposition& dec)
{
    const std::vector<real_t> rebuilt = recompose(dec, fir.length());
    double max_err = 0.0;
    for (size_t i = 0; i < fir.h.size(); ++i) {
        const double v = (i < rebuilt.size()) ? rebuilt[i] : 0.0;
        const double err = std::abs(static_cast<double>(fir.h[i]) - v);
        if (err > max_err) {
            max_err = err;
        }
    }
    return max_err;
}

inline CompareMetrics cascade_noise_metrics(const DirectFIR& fir,
                                            const CascadeDecomposition& dec,
                                            size_t num_samples = 50000,
                                            uint32_t seed = 777)
{
    const std::vector<sample_t> noise = generate_white_noise(num_samples, seed);
    const std::vector<sample_t> y_direct = filter_direct(fir, noise);
    const std::vector<sample_t> y_cascade = filter_cascade(dec, noise);
    return compare_signals(y_direct, y_cascade);
}

inline CompareMetrics cascade_impulse_metrics(const DirectFIR& fir,
                                              const CascadeDecomposition& dec)
{
    const std::vector<sample_t> impulse = generate_impulse(fir.length() * 2u);
    const std::vector<sample_t> y_direct = filter_direct(fir, impulse);
    const std::vector<sample_t> y_cascade = filter_cascade(dec, impulse);
    return compare_signals(y_direct, y_cascade);
}

#endif
