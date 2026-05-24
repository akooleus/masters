#include "test_utils.h"

int main()
{
    const FilterSpec spec = smoke_spec();
    const DirectFIR fir = design_freq_sampling(spec);

    bool ok = true;
    ok &= expect(fir.length() == spec.N, "frequency sampling design keeps requested length");
    ok &= expect(spec.k_pass() < spec.k_stop(), "frequency sample indices preserve pass/stop ordering");
    ok &= expect(std::fabs(fir.h.front() - fir.h.back()) < 1e-12,
                 "frequency sampling design remains symmetric");
    ok &= expect(compute_magnitude_response(fir.h, 256).size() == 256,
                 "frequency response helper returns requested FFT size");
    return ok ? 0 : 1;
}
