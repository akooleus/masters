#include "cascade_fir.h"

#include "test_utils.h"

#include <vector>

int main()
{
    bool ok = true;

    const std::vector<FilterSpec> specs = {
        {33, 32000.0, 4000.0,  6000.0},
        {49, 32000.0, 4000.0,  6000.0},
        {65, 48000.0, 8000.0, 12000.0},
        {97, 48000.0, 8000.0, 12000.0},
    };

    for (const FilterSpec& spec : specs) {
        const DirectFIR fir = design_freq_sampling(spec);
        const CascadeDecomposition s1 = decompose_exact_fs(fir, true);
        const CascadeDecomposition full = decompose_exact_fs_full(fir, true);

        const double err_s1 = compute_rebuild_err(fir, s1);
        const double err_full = compute_rebuild_err(fir, full);
        const unsigned deg_s1 =
            s1.remainder.empty() ? 0u : static_cast<unsigned>(s1.remainder.size() - 1);
        const unsigned deg_full =
            full.remainder.empty() ? 0u : static_cast<unsigned>(full.remainder.size() - 1);

        const std::string prefix = "N=" + std::to_string(spec.N) + ": ";
        ok &= expect(err_full <= std::max(1e-6, 10.0 * err_s1),
                     prefix + "full path worsened rebuild too much");
        ok &= expect(deg_full <= deg_s1,
                     prefix + "full path increased residual degree");

        if (spec.N <= 49) {
            ok &= expect(err_full <= 10.0 * std::max(1e-12, err_s1),
                         prefix + "small-N full path should stay near stage1 quality");
        }

        const CompareMetrics noi_s1 =
            cascade_noise_metrics(fir, s1, 50000u, 1700u + spec.N);
        const CompareMetrics noi_full =
            cascade_noise_metrics(fir, full, 50000u, 1700u + spec.N);
        ok &= expect(noi_full.snr_db >= 110.0,
                     prefix + "full path noise SNR too low: " + std::to_string(noi_full.snr_db));
        ok &= expect(noi_full.snr_db >= noi_s1.snr_db - 10.0,
                     prefix + "full path lost too much noise SNR vs stage1");
    }

    return ok ? 0 : 1;
}
