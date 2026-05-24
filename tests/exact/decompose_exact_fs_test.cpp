#include "cascade_fir.h"

#include "test_utils.h"

#include <complex>
#include <vector>

namespace {

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
    bool ok = true;

    const std::vector<FilterSpec> specs = {
        {33, 32000.0, 4000.0,  6000.0},
        {49, 32000.0, 4000.0,  6000.0},
        {65, 48000.0, 8000.0, 12000.0},
        {97, 48000.0, 8000.0, 12000.0},
        {129, 48000.0, 8000.0, 12000.0},
    };

    for (const FilterSpec& spec : specs) {
        const DirectFIR fir = design_freq_sampling(spec);
        const CascadeDecomposition dec = decompose_exact_fs(fir, true);
        const std::string prefix = "N=" + std::to_string(spec.N) + ": ";

        ok &= expect(!dec.biquads.empty(),
                     prefix + "exact_fs should extract at least one exact biquad");
        ok &= expect(dec.remainder.size() < fir.h.size(),
                     prefix + "remainder should be shorter than original FIR");

        const double rebuild_err = compute_rebuild_err(fir, dec);
        ok &= expect(rebuild_err <= 1e-3,
                     prefix + "rebuild error too large: " + std::to_string(rebuild_err));

        const unsigned extracted_zeros = dec.zeros_extracted();
        const unsigned remainder_degree =
            dec.remainder.empty() ? 0u : static_cast<unsigned>(dec.remainder.size() - 1);
        ok &= expect(extracted_zeros + remainder_degree == fir.order(),
                     prefix + "degree accounting mismatch");

        const unsigned expected_exact_zeros = count_exact_unit_circle_zeros(fir);
        ok &= expect(extracted_zeros <= expected_exact_zeros,
                     prefix + "extracted more zeros than exact known set");

        if (spec.N <= 49) {
            ok &= expect(extracted_zeros == expected_exact_zeros,
                         prefix + "small-N exact zero count mismatch");
        }

        const CompareMetrics imp = cascade_impulse_metrics(fir, dec);
        const CompareMetrics noi =
            cascade_noise_metrics(fir, dec, 50000u, 700u + spec.N);
        ok &= expect(imp.snr_db >= 110.0,
                     prefix + "impulse SNR too low: " + std::to_string(imp.snr_db));
        ok &= expect(noi.snr_db >= 110.0,
                     prefix + "noise SNR too low: " + std::to_string(noi.snr_db));
    }

    return ok ? 0 : 1;
}
