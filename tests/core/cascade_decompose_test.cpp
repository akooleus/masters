#include "test_utils.h"

int main()
{
    const DirectFIR fir = smoke_fir();
    const CascadeDecomposition dec = decompose(fir, true);
    const std::vector<real_t> rebuilt = recompose(dec, fir.length());

    bool ok = true;
    ok &= expect(dec.zeros_extracted() > 0, "baseline decomposition extracts zeros");
    ok &= expect(!dec.biquads.empty(), "baseline decomposition emits biquad sections");
    ok &= expect_coeff_error(fir.h, rebuilt, 1e-8, "baseline decomposition rebuild");
    return ok ? 0 : 1;
}
