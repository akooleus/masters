#include "cascade_fir.h"

#include "test_utils.h"

#include <string>
#include <vector>

int main()
{
    struct Case {
        const char* label;
        FilterSpec spec;
        bool smooth_transition;
    };

    const std::vector<Case> cases = {
        {"N32",   {32,  48000.0, 8000.0, 12000.0}, false},
        {"N33",   {33,  32000.0, 4000.0,  6000.0}, false},
        {"N49",   {49,  32000.0, 4000.0,  6000.0}, false},
        {"N65",   {65,  48000.0, 8000.0, 12000.0}, false},
        {"N97",   {97,  48000.0, 8000.0, 12000.0}, false},
        {"N129",  {129, 48000.0, 8000.0, 12000.0}, false},
        {"N128T", {128, 48000.0, 8000.0, 12000.0}, true},
        {"N257T", {257, 48000.0, 8000.0, 12000.0}, true},
        {"N513T", {513, 48000.0, 8000.0, 12000.0}, true},
    };

    bool ok = true;
    for (const Case& tc : cases) {
        const std::vector<real_t> transition = tc.smooth_transition
            ? cosine_transition(tc.spec) : std::vector<real_t>{};
        const DirectFIR fir = design_freq_sampling(tc.spec, transition);
        const CascadeDecomposition stage = decompose_exact_fs(fir, true);
        const std::string prefix = std::string(tc.label) + ": ";

        const unsigned known = count_known_fs_zeros(fir);
        const unsigned expected_residual_degree = fir.order() - known;
        ok &= expect(stage.zeros_extracted() == known,
                     prefix + "analytical zero count mismatch");
        ok &= expect(stage.diagnostics.status
                         == CascadeBuildStatus::AnalyticalReduction,
                     prefix + "stage status mismatch");
        ok &= expect(stage.diagnostics.complement_verified,
                     prefix + "complementary identity was not verified");
        ok &= expect(stage.diagnostics.complement_identity_error <= 1e-60,
                     prefix + "complementary identity error too large");
        ok &= expect(remainder_degree(stage) == expected_residual_degree,
                     prefix + "complement residual degree mismatch");
        ok &= expect(stage.zeros_extracted() + remainder_degree(stage)
                         == fir.order(),
                     prefix + "degree accounting mismatch");
        ok &= expect(decomposition_is_finite(stage),
                     prefix + "stage contains NaN or infinity");

        if (tc.smooth_transition) {
            ok &= expect(!transition.empty(),
                         prefix + "transition samples were not generated");
            for (double value : transition) {
                ok &= expect(value > 0.0 && value < 1.0,
                             prefix + "transition must be strictly nonzero");
            }
        }
    }

    return ok ? 0 : 1;
}
