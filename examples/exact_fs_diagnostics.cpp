#include "cascade_fir.h"

#include <complex>
#include <cstdio>
#include <vector>

namespace {

double compute_rebuild_err(const CascadeDecomposition& dec,
                           const std::vector<real_t>& h_orig,
                           unsigned original_length)
{
    const std::vector<real_t> rebuilt = recompose(dec, original_length);
    double max_err = 0.0;
    for (size_t i = 0; i < h_orig.size(); ++i) {
        const double v = (i < rebuilt.size()) ? rebuilt[i] : 0.0;
        const double err = std::abs(h_orig[i] - v);
        if (err > max_err) {
            max_err = err;
        }
    }
    return max_err;
}

unsigned count_exact_unit_circle_zeros(const DirectFIR& fir)
{
    const unsigned N = fir.length();
    const unsigned k_half = N / 2;

    long double max_abs = 0.0L;
    std::vector<long double> mags(k_half + 1, 0.0L);

    for (unsigned k = 0; k <= k_half; ++k) {
        const long double theta =
            2.0L * static_cast<long double>(PI)
            * static_cast<long double>(k)
            / static_cast<long double>(N);
        const std::complex<long double> z_inv = std::polar(1.0L, -theta);

        std::complex<long double> z_pow = 1.0L;
        std::complex<long double> acc = 0.0L;
        for (real_t coeff : fir.h) {
            acc += static_cast<long double>(coeff) * z_pow;
            z_pow *= z_inv;
        }

        const long double mag = std::abs(acc);
        mags[k] = mag;
        if (mag > max_abs) {
            max_abs = mag;
        }
    }

    const long double zero_tol = std::max(1e-18L, max_abs * 1e-12L);

    unsigned zero_count = 0;
    for (unsigned k = 0; k <= k_half; ++k) {
        if (mags[k] > zero_tol) {
            continue;
        }

        if (k == 0 || (N % 2 == 0 && k == k_half)) {
            zero_count += 1;
        } else {
            zero_count += 2;
        }
    }

    return zero_count;
}

} // namespace

int main()
{
    struct Case {
        const char* label;
        FilterSpec spec;
    };

    const std::vector<Case> cases = {
        {"N33",  {33, 32000.0, 4000.0,  6000.0}},
        {"N49",  {49, 32000.0, 4000.0,  6000.0}},
        {"N65",  {65, 48000.0, 8000.0, 12000.0}},
        {"N97",  {97, 48000.0, 8000.0, 12000.0}},
        {"N129", {129, 48000.0, 8000.0, 12000.0}},
    };

    std::printf("Exact-FS diagnostics\n\n");
    std::printf("%-6s %5s %8s %8s %8s %14s %14s %14s\n",
                "case", "N", "known_z", "taken_z", "rem_deg",
                "rebuild_s1", "rebuild_struct", "rebuild_full");
    std::printf("------ ----- -------- -------- -------- -------------- -------------- --------------\n");

    for (const auto& tc : cases) {
        const DirectFIR fir = design_freq_sampling(tc.spec);
        const CascadeDecomposition dec = decompose_exact_fs(fir, true);
        const CascadeDecomposition structured = decompose_exact_fs_structured(fir, true);
        const CascadeDecomposition full = decompose_exact_fs_full(fir, true);

        const unsigned known_zeros = count_exact_unit_circle_zeros(fir);
        const unsigned taken_zeros = dec.zeros_extracted();
        const unsigned rem_deg =
            dec.remainder.empty() ? 0u : static_cast<unsigned>(dec.remainder.size() - 1);
        const double rebuild_err = compute_rebuild_err(dec, fir.h, tc.spec.N);
        const double rebuild_struct = compute_rebuild_err(structured, fir.h, tc.spec.N);
        const double rebuild_full = compute_rebuild_err(full, fir.h, tc.spec.N);

        std::printf("%-6s %5u %8u %8u %8u %14.6e %14.6e %14.6e\n",
                    tc.label, tc.spec.N, known_zeros, taken_zeros, rem_deg,
                    rebuild_err, rebuild_struct, rebuild_full);
    }

    return 0;
}
