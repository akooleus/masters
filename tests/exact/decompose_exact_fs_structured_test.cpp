#include "cascade_fir.h"

#include "test_utils.h"

#include <vector>

int main()
{
    bool ok = true;

    const std::vector<FilterSpec> specs = {
        {65,  48000.0, 8000.0, 12000.0},
        {97,  48000.0, 8000.0, 12000.0},
        {129, 48000.0, 8000.0, 12000.0},
    };

    for (const FilterSpec& spec : specs) {
        const DirectFIR fir = design_freq_sampling(spec);
        const CascadeDecomposition s1 = decompose_exact_fs(fir, true);
        const CascadeDecomposition st = decompose_exact_fs_structured(fir, true);

        const double err_s1 = compute_rebuild_err(fir, s1);
        const double err_st = compute_rebuild_err(fir, st);
        const unsigned deg_s1 =
            s1.remainder.empty() ? 0u : static_cast<unsigned>(s1.remainder.size() - 1);
        const unsigned deg_st =
            st.remainder.empty() ? 0u : static_cast<unsigned>(st.remainder.size() - 1);

        const std::string prefix = "N=" + std::to_string(spec.N) + ": ";
        ok &= expect(err_st <= std::max(1e-6, 10.0 * err_s1),
                     prefix + "structured path worsened rebuild too much");
        ok &= expect(deg_st <= deg_s1,
                     prefix + "structured path increased residual degree");

        const CompareMetrics noi_s1 =
            cascade_noise_metrics(fir, s1, 50000u, 2700u + spec.N);
        const CompareMetrics noi_st =
            cascade_noise_metrics(fir, st, 50000u, 2700u + spec.N);
        ok &= expect(noi_st.snr_db >= 110.0,
                     prefix + "structured path noise SNR too low: " + std::to_string(noi_st.snr_db));
        ok &= expect(noi_st.snr_db >= noi_s1.snr_db - 10.0,
                     prefix + "structured path lost too much noise SNR vs stage1");
    }

    return ok ? 0 : 1;
}
