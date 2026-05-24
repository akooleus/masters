#ifndef CASCADE_FIR_TEST_UTILS_H
#define CASCADE_FIR_TEST_UTILS_H

#include "cascade_fir.h"

#include <cmath>
#include <iostream>
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
