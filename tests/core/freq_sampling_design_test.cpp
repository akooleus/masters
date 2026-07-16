#include "test_utils.h"

#include <limits>
#include <stdexcept>

namespace {

bool rejects(const FilterSpec& spec,
             const std::vector<real_t>& transition = {})
{
    try {
        (void)design_freq_sampling(spec, transition);
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

} // namespace

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
    ok &= expect(rejects({2, 32000.0, 4000.0, 6000.0}),
                 "designer rejects N < 3");
    ok &= expect(rejects({33, 0.0, 4000.0, 6000.0}),
                 "designer rejects nonpositive sample rate");
    ok &= expect(rejects({33, 32000.0, 7000.0, 6000.0}),
                 "designer rejects reversed pass/stop edges");
    ok &= expect(rejects(spec, {0.5}),
                 "designer rejects a transition vector of the wrong size");
    std::vector<real_t> invalid_transition(
        spec.k_stop() - spec.k_pass() - 1, 0.5);
    invalid_transition.front() = std::numeric_limits<double>::quiet_NaN();
    ok &= expect(rejects(spec, invalid_transition),
                 "designer rejects non-finite transition values");
    return ok ? 0 : 1;
}
