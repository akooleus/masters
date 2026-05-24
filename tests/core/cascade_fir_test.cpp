#include "test_utils.h"

int main()
{
    const DirectFIR fir = smoke_fir();
    bool ok = true;

    ok &= expect(fir.length() == smoke_spec().N, "public API returns expected FIR length");
    ok &= expect(!fir.h.empty(), "public API exposes FIR coefficients");
    ok &= expect(std::fabs(fir.h.front() - fir.h.back()) < 1e-12,
                 "public API produces symmetric FIR coefficients");

    return ok ? 0 : 1;
}
