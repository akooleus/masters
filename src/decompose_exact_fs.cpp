// Stable decomposition of linear-phase frequency-sampling FIR filters.
//
// The known unit-circle zeros are removed with the complementary-polynomial
// identity from Aliphas, Narayan and Peterson (1983):
//
//   H = H0 H1,        1 - z^-N = H0 H2
//   ------------------------------------------------
//                 H H2 = H1 (1 - z^-N)
//
// Hence H1 is the first coefficient block of the convolution H * H2.  No
// deflation by H0 is performed.  The reduced palindromic H1 is factored in
// the Chebyshev basis with a colleague-matrix seed followed by a simultaneous
// multiprecision Aberth-Ehrlich refinement.

#include "cascade_fir.h"

#include <boost/multiprecision/cpp_bin_float.hpp>
#include <boost/multiprecision/cpp_complex.hpp>

#include <algorithm>
#include <complex>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace {

using mp_real = boost::multiprecision::number<
    boost::multiprecision::cpp_bin_float<200>>;
using mp_complex = boost::multiprecision::cpp_complex<200>;

extern "C" {
void dgeev_(const char* jobvl, const char* jobvr,
            const int* n, double* a, const int* lda,
            double* wr, double* wi,
            double* vl, const int* ldvl,
            double* vr, const int* ldvr,
            double* work, const int* lwork, int* info);
}

const mp_real& mp_pi()
{
    static const mp_real value = acos(mp_real(-1));
    return value;
}

template <class T>
std::vector<T> poly_multiply_t(const std::vector<T>& a,
                               const std::vector<T>& b)
{
    if (a.empty() || b.empty()) {
        return {};
    }

    std::vector<T> c(a.size() + b.size() - 1, T(0));
    for (size_t i = 0; i < a.size(); ++i) {
        for (size_t j = 0; j < b.size(); ++j) {
            c[i + j] += a[i] * b[j];
        }
    }
    return c;
}

template <class T>
std::vector<T> balanced_product(std::vector<std::vector<T>> factors)
{
    if (factors.empty()) {
        return {T(1)};
    }

    while (factors.size() > 1) {
        std::vector<std::vector<T>> next;
        next.reserve((factors.size() + 1) / 2);
        for (size_t i = 0; i < factors.size(); i += 2) {
            if (i + 1 < factors.size()) {
                next.push_back(poly_multiply_t(factors[i], factors[i + 1]));
            } else {
                next.push_back(std::move(factors[i]));
            }
        }
        factors = std::move(next);
    }
    return std::move(factors.front());
}

void enforce_palindrome(std::vector<mp_real>& p)
{
    for (size_t i = 0; i < p.size() / 2; ++i) {
        const mp_real avg = (p[i] + p[p.size() - 1 - i]) / 2;
        p[i] = avg;
        p[p.size() - 1 - i] = avg;
    }
}

mp_real max_abs(const std::vector<mp_real>& p)
{
    mp_real result = 0;
    for (const mp_real& v : p) {
        result = std::max(result, abs(v));
    }
    return result;
}

std::vector<unsigned> interleave_indices(std::vector<unsigned> indices)
{
    std::sort(indices.begin(), indices.end());
    std::vector<unsigned> order;
    order.reserve(indices.size());

    size_t lo = 0;
    size_t hi = indices.size();
    while (lo < hi) {
        order.push_back(indices[lo++]);
        if (lo < hi) {
            order.push_back(indices[--hi]);
        }
    }
    return order;
}

std::vector<bool> exact_zero_mask(const DirectFIR& fir)
{
    const unsigned N = fir.length();
    const unsigned half = N / 2;
    std::vector<bool> zero(half + 1, false);

    if (fir.frequency_samples.size() == N) {
        for (unsigned k = 0; k <= half; ++k) {
            zero[k] = (fir.frequency_samples[k] == 0.0);
        }
        return zero;
    }

    // Compatibility path for a manually assembled DirectFIR.  It is not
    // called "exact": without the original H[k], only numerical detection is
    // possible.
    long double max_mag = 0.0L;
    std::vector<long double> mags(half + 1, 0.0L);
    for (unsigned k = 0; k <= half; ++k) {
        const long double theta =
            2.0L * acosl(-1.0L) * static_cast<long double>(k)
            / static_cast<long double>(N);
        const std::complex<long double> step = std::polar(1.0L, -theta);
        std::complex<long double> power = 1.0L;
        std::complex<long double> value = 0.0L;
        for (double h : fir.h) {
            value += static_cast<long double>(h) * power;
            power *= step;
        }
        mags[k] = std::abs(value);
        max_mag = std::max(max_mag, mags[k]);
    }

    const long double tol = std::max(1e-18L, max_mag * 1e-12L);
    for (unsigned k = 0; k <= half; ++k) {
        zero[k] = mags[k] <= tol;
    }
    return zero;
}

std::vector<mp_real> design_impulse_mp(const DirectFIR& fir)
{
    const unsigned N = fir.length();
    if (fir.frequency_samples.size() != N) {
        std::vector<mp_real> result;
        result.reserve(fir.h.size());
        for (double h : fir.h) {
            result.emplace_back(h);
        }
        return result;
    }

    const mp_real alpha = mp_real(N - 1) / 2;
    std::vector<mp_real> h(N, mp_real(fir.frequency_samples[0]));
    const unsigned pair_last = (N % 2 == 0) ? N / 2 - 1 : N / 2;

    for (unsigned k = 1; k <= pair_last; ++k) {
        const mp_real magnitude = mp_real(fir.frequency_samples[k]);
        if (magnitude == 0) {
            continue;
        }

        const mp_real step = 2 * mp_pi() * k / N;
        const mp_real phase0 = -step * alpha;
        mp_real c = cos(phase0);
        mp_real s = sin(phase0);
        const mp_real c_step = cos(step);
        const mp_real s_step = sin(step);

        for (unsigned n = 0; n < N; ++n) {
            h[n] += 2 * magnitude * c;
            const mp_real next_c = c * c_step - s * s_step;
            const mp_real next_s = s * c_step + c * s_step;
            c = next_c;
            s = next_s;
        }
    }

    if (N % 2 == 0 && fir.frequency_samples[N / 2] != 0.0) {
        const mp_real magnitude = mp_real(fir.frequency_samples[N / 2]);
        for (unsigned n = 0; n < N; ++n) {
            h[n] += magnitude * cos(mp_pi() * (mp_real(n) - alpha));
        }
    }

    for (mp_real& value : h) {
        value /= N;
    }
    enforce_palindrome(h);
    return h;
}

struct ComplementReduction {
    CascadeDecomposition decomposition;
    std::vector<mp_real> residual;
    std::vector<mp_real> impulse;
    unsigned known_zero_count = 0;
    mp_real identity_error = 0;
};

ComplementReduction reduce_by_complement(const DirectFIR& fir,
                                         bool interleave_order)
{
    ComplementReduction result;
    result.impulse = design_impulse_mp(fir);

    if (fir.h.empty()) {
        result.decomposition.remainder = {};
        return result;
    }

    const unsigned N = fir.length();
    const std::vector<bool> zero = exact_zero_mask(fir);
    std::vector<std::vector<mp_real>> complement_factors;
    std::vector<unsigned> known_pairs;

    if (zero[0]) {
        result.decomposition.first_order.push_back(FirstOrder{-1});
        ++result.known_zero_count;
    } else {
        complement_factors.push_back({mp_real(1), mp_real(-1)});
    }

    const unsigned pair_last = (N % 2 == 0) ? N / 2 - 1 : N / 2;
    for (unsigned k = 1; k <= pair_last; ++k) {
        const mp_real gamma = 2 * cos(2 * mp_pi() * k / N);
        if (zero[k]) {
            known_pairs.push_back(k);
            result.known_zero_count += 2;
        } else {
            complement_factors.push_back({mp_real(1), -gamma, mp_real(1)});
        }
    }

    if (N % 2 == 0) {
        if (zero[N / 2]) {
            result.decomposition.first_order.push_back(FirstOrder{+1});
            ++result.known_zero_count;
        } else {
            complement_factors.push_back({mp_real(1), mp_real(1)});
        }
    }

    const std::vector<unsigned> output_order = interleave_order
        ? interleave_indices(known_pairs)
        : known_pairs;
    for (unsigned k : output_order) {
        const mp_real gamma = 2 * cos(2 * mp_pi() * k / N);
        result.decomposition.biquads.push_back(
            Biquad{static_cast<double>(gamma), k});
    }

    const std::vector<mp_real> H2 = balanced_product(std::move(complement_factors));
    const std::vector<mp_real> product = poly_multiply_t(result.impulse, H2);
    const unsigned residual_degree = (N - 1) - result.known_zero_count;
    const size_t residual_size = static_cast<size_t>(residual_degree) + 1;

    if (product.size() < residual_size) {
        throw std::runtime_error("complement reduction produced a short convolution");
    }

    result.residual.assign(product.begin(), product.begin() + residual_size);
    enforce_palindrome(result.residual);

    // Verify H*H2 = H1 - z^-N H1 without expanding the ill-conditioned H0.
    mp_real max_err = 0;
    for (size_t i = 0; i < product.size(); ++i) {
        mp_real expected = 0;
        if (i < residual_size) {
            expected += result.residual[i];
        }
        if (i >= N && i - N < residual_size) {
            expected -= result.residual[i - N];
        }
        max_err = std::max(max_err, abs(product[i] - expected));
    }
    result.identity_error = max_err / std::max(mp_real("1e-90"), max_abs(product));

    result.decomposition.diagnostics.status =
        CascadeBuildStatus::AnalyticalReduction;
    result.decomposition.diagnostics.complement_identity_error =
        static_cast<double>(result.identity_error);
    result.decomposition.diagnostics.complement_verified =
        result.identity_error <= mp_real("1e-60");

    result.decomposition.remainder.reserve(result.residual.size());
    for (const mp_real& value : result.residual) {
        result.decomposition.remainder.push_back(static_cast<double>(value));
    }
    result.decomposition.gain = 1.0;
    result.decomposition.runtime_gain = 1.0;
    return result;
}

std::vector<std::complex<double>> chebyshev_colleague_seeds(
    const std::vector<mp_real>& coefficients)
{
    const int n = static_cast<int>(coefficients.size()) - 1;
    if (n <= 0) {
        return {};
    }
    if (n == 1) {
        return {{
            -static_cast<double>(coefficients[0] / coefficients[1]), 0.0
        }};
    }

    const mp_real coefficient_scale = std::max(mp_real("1e-90"), max_abs(coefficients));
    std::vector<double> c(coefficients.size(), 0.0);
    for (size_t i = 0; i < coefficients.size(); ++i) {
        c[i] = static_cast<double>(coefficients[i] / coefficient_scale);
    }

    if (!std::isfinite(c.back()) || c.back() == 0.0) {
        return {};
    }

    std::vector<double> matrix(static_cast<size_t>(n * n), 0.0);
    auto A = [&](int row, int col) -> double& {
        return matrix[static_cast<size_t>(row + col * n)];
    };

    const double sqrt_half = std::sqrt(0.5);
    A(0, 1) = sqrt_half;
    A(1, 0) = sqrt_half;
    for (int i = 1; i + 1 < n; ++i) {
        A(i, i + 1) = 0.5;
        A(i + 1, i) = 0.5;
    }

    const double scl_last = sqrt_half;
    for (int row = 0; row < n; ++row) {
        const double scl_row = (row == 0) ? 1.0 : sqrt_half;
        A(row, n - 1) -=
            0.5 * (c[static_cast<size_t>(row)] / c.back())
            * (scl_row / scl_last);
    }

    std::vector<double> wr(static_cast<size_t>(n), 0.0);
    std::vector<double> wi(static_cast<size_t>(n), 0.0);
    double work_query = 0.0;
    int lwork = -1;
    int info = 0;
    int one = 1;

    dgeev_("N", "N", &n, matrix.data(), &n,
           wr.data(), wi.data(), nullptr, &one, nullptr, &one,
           &work_query, &lwork, &info);
    if (info != 0 || !std::isfinite(work_query)) {
        return {};
    }

    lwork = std::max(4 * n, static_cast<int>(work_query));
    std::vector<double> work(static_cast<size_t>(lwork), 0.0);
    dgeev_("N", "N", &n, matrix.data(), &n,
           wr.data(), wi.data(), nullptr, &one, nullptr, &one,
           work.data(), &lwork, &info);
    if (info != 0) {
        return {};
    }

    std::vector<std::complex<double>> roots(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        roots[static_cast<size_t>(i)] = {wr[static_cast<size_t>(i)],
                                         wi[static_cast<size_t>(i)]};
        if (!std::isfinite(wr[static_cast<size_t>(i)])
            || !std::isfinite(wi[static_cast<size_t>(i)])) {
            return {};
        }
    }
    return roots;
}

struct EvalResult {
    mp_complex value;
    mp_complex derivative;
    mp_real absolute_sum;
};

EvalResult evaluate_chebyshev_with_derivative(
    const std::vector<mp_real>& c,
    const mp_complex& z)
{
    EvalResult result{mp_complex(c[0]), mp_complex(0), abs(c[0])};
    if (c.size() == 1) {
        return result;
    }

    mp_complex T_prev2 = 1;
    mp_complex T_prev1 = z;
    mp_complex d_prev2 = 0;
    mp_complex d_prev1 = 1;
    result.value += c[1] * T_prev1;
    result.derivative += c[1];
    result.absolute_sum += abs(c[1] * T_prev1);

    for (size_t k = 2; k < c.size(); ++k) {
        const mp_complex T = 2 * z * T_prev1 - T_prev2;
        const mp_complex dT = 2 * T_prev1 + 2 * z * d_prev1 - d_prev2;
        result.value += c[k] * T;
        result.derivative += c[k] * dT;
        result.absolute_sum += abs(c[k] * T);
        T_prev2 = T_prev1;
        T_prev1 = T;
        d_prev2 = d_prev1;
        d_prev1 = dT;
    }
    return result;
}

struct RootSolveResult {
    std::vector<mp_complex> roots;
    unsigned iterations = 0;
    mp_real max_backward_error = 1;
    bool converged = false;
};

RootSolveResult solve_chebyshev_roots(const std::vector<mp_real>& input_c)
{
    RootSolveResult result;
    if (input_c.size() <= 1) {
        result.converged = true;
        result.max_backward_error = 0;
        return result;
    }

    std::vector<mp_real> c = input_c;
    const mp_real scale = std::max(mp_real("1e-90"), max_abs(c));
    for (mp_real& value : c) {
        value /= scale;
    }

    const std::vector<std::complex<double>> seeds = chebyshev_colleague_seeds(c);
    const size_t degree = c.size() - 1;
    result.roots.resize(degree);

    if (seeds.size() == degree) {
        for (size_t i = 0; i < degree; ++i) {
            result.roots[i] = mp_complex(seeds[i].real(), seeds[i].imag());
        }
    } else {
        // Deterministic fallback.  The colleague path is normally used; this
        // circle only keeps failure explicit instead of returning no roots.
        const mp_real radius = 2;
        for (size_t i = 0; i < degree; ++i) {
            const mp_real angle =
                2 * mp_pi() * (mp_real(i) + mp_real("0.5")) / degree;
            result.roots[i] = mp_complex(radius * cos(angle), radius * sin(angle));
        }
    }

    static const mp_real step_tolerance("1e-120");
    static const mp_real residual_tolerance("1e-120");
    static const mp_real tiny("1e-180");
    mp_real previous_max_step = std::numeric_limits<mp_real>::max();

    for (unsigned iteration = 0; iteration < 800; ++iteration) {
        const std::vector<mp_complex> old = result.roots;
        mp_real max_relative_step = 0;

        for (size_t i = 0; i < degree; ++i) {
            const EvalResult eval = evaluate_chebyshev_with_derivative(c, old[i]);
            if (abs(eval.derivative) < tiny) {
                continue;
            }

            const mp_complex newton = eval.value / eval.derivative;
            mp_complex repulsion = 0;
            for (size_t j = 0; j < degree; ++j) {
                if (i == j) {
                    continue;
                }
                const mp_complex difference = old[i] - old[j];
                if (abs(difference) > tiny) {
                    repulsion += 1 / difference;
                }
            }

            mp_complex denominator = 1 - newton * repulsion;
            if (abs(denominator) < tiny) {
                denominator = 1;
            }
            mp_complex step = newton / denominator;

            const mp_real root_scale = 1 + abs(old[i]);
            const mp_real step_abs = abs(step);
            if (step_abs > root_scale / 2) {
                step *= (root_scale / 2) / step_abs;
            }

            result.roots[i] = old[i] - step;
            max_relative_step = std::max(
                max_relative_step, abs(step) / root_scale);
        }

        result.iterations = iteration + 1;

        mp_real max_backward = 0;
        for (const mp_complex& root : result.roots) {
            const EvalResult eval = evaluate_chebyshev_with_derivative(c, root);
            const mp_real denominator = std::max(tiny, eval.absolute_sum);
            max_backward = std::max(max_backward, abs(eval.value) / denominator);
        }
        result.max_backward_error = max_backward;

        if (max_relative_step < step_tolerance
            && max_backward < residual_tolerance) {
            result.converged = true;
            break;
        }

        // A very small backward error is sufficient even for a clustered root
        // whose forward step is limited by conditioning.
        if (max_backward < mp_real("1e-140")
            && max_relative_step >= previous_max_step
            && iteration > 20) {
            result.converged = true;
            break;
        }
        previous_max_step = max_relative_step;
    }
    return result;
}

std::complex<double> section_value(const CascadeDecomposition& dec,
                                   const CascadeSectionPlacement& placement,
                                   double omega)
{
    const std::complex<double> x = std::polar(1.0, -omega);
    switch (placement.kind) {
    case CascadeSectionKind::FirstOrder: {
        const auto& section = dec.first_order[placement.index];
        return 1.0 + static_cast<double>(section.sign) * x;
    }
    case CascadeSectionKind::Biquad: {
        const auto& section = dec.biquads[placement.index];
        return 1.0 - section.gamma * x + x * x;
    }
    case CascadeSectionKind::Quartic: {
        const auto& section = dec.quartics[placement.index];
        const std::complex<double> x2 = x * x;
        return 1.0 - section.alpha * x + section.beta * x2
             - section.alpha * x2 * x + x2 * x2;
    }
    case CascadeSectionKind::Block: {
        const auto& block = dec.blocks[placement.index].coefficients;
        std::complex<double> value = 0.0;
        for (size_t i = block.size(); i-- > 0;) {
            value = value * x + block[i];
        }
        return value;
    }
    }
    return 1.0;
}

double section_dc_value(const CascadeDecomposition& dec,
                        const CascadeSectionPlacement& placement)
{
    switch (placement.kind) {
    case CascadeSectionKind::FirstOrder:
        return 1.0 + dec.first_order[placement.index].sign;
    case CascadeSectionKind::Biquad:
        return 2.0 - dec.biquads[placement.index].gamma;
    case CascadeSectionKind::Quartic: {
        const auto& section = dec.quartics[placement.index];
        return 2.0 - 2.0 * section.alpha + section.beta;
    }
    case CascadeSectionKind::Block:
        return std::accumulate(
            dec.blocks[placement.index].coefficients.begin(),
            dec.blocks[placement.index].coefficients.end(), 0.0);
    }
    return 1.0;
}

void build_balanced_execution_order(CascadeDecomposition& dec,
                                    const DirectFIR& fir)
{
    std::vector<CascadeSectionPlacement> candidates;
    candidates.reserve(dec.first_order.size() + dec.biquads.size()
                       + dec.quartics.size());
    for (size_t i = 0; i < dec.first_order.size(); ++i) {
        candidates.push_back({CascadeSectionKind::FirstOrder, i, 1.0});
    }
    for (size_t i = 0; i < dec.biquads.size(); ++i) {
        candidates.push_back({CascadeSectionKind::Biquad, i, 1.0});
    }
    for (size_t i = 0; i < dec.quartics.size(); ++i) {
        candidates.push_back({CascadeSectionKind::Quartic, i, 1.0});
    }
    for (size_t i = 0; i < dec.blocks.size(); ++i) {
        candidates.push_back({CascadeSectionKind::Block, i, 1.0});
    }

    // The public designer currently produces low-pass filters with nonzero
    // DC gain.  Normalising every section to unity at DC prevents the huge
    // final compensation factor used by the old L1 scheme.
    for (auto& placement : candidates) {
        const double dc = section_dc_value(dec, placement);
        if (std::isfinite(dc) && std::abs(dc) > 1e-18) {
            placement.scale = 1.0 / dc;
        } else {
            placement.scale = 1.0;
        }
    }

    static constexpr size_t GRID_SIZE = 65;
    const double pass_edge = std::clamp(
        2.0 * PI * fir.spec.f_pass / fir.spec.fs, 0.0, PI);
    std::vector<std::vector<double>> logs(
        candidates.size(), std::vector<double>(GRID_SIZE, 0.0));
    for (size_t i = 0; i < candidates.size(); ++i) {
        for (size_t g = 0; g < GRID_SIZE; ++g) {
            const double omega = pass_edge * static_cast<double>(g)
                               / static_cast<double>(GRID_SIZE - 1);
            const double magnitude = std::abs(
                candidates[i].scale * section_value(dec, candidates[i], omega));
            logs[i][g] = std::log(std::max(1e-300, magnitude));
        }
    }

    std::vector<bool> used(candidates.size(), false);
    std::vector<double> cumulative(GRID_SIZE, 0.0);
    dec.execution_order.clear();
    dec.execution_order.reserve(candidates.size());

    for (size_t position = 0; position < candidates.size(); ++position) {
        size_t best = candidates.size();
        double best_score = std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < candidates.size(); ++i) {
            if (used[i]) {
                continue;
            }
            double score = 0.0;
            for (size_t g = 0; g < GRID_SIZE; ++g) {
                score = std::max(score, std::abs(cumulative[g] + logs[i][g]));
            }
            if (score < best_score) {
                best_score = score;
                best = i;
            }
        }

        used[best] = true;
        dec.execution_order.push_back(candidates[best]);
        for (size_t g = 0; g < GRID_SIZE; ++g) {
            cumulative[g] += logs[best][g];
        }
    }

    if (fir.frequency_samples.size() == fir.length()) {
        dec.runtime_gain = fir.frequency_samples[0];
    } else {
        dec.runtime_gain = std::accumulate(fir.h.begin(), fir.h.end(), 0.0);
    }
}

std::vector<mp_real> placement_factor_mp(
    const CascadeDecomposition& dec,
    const CascadeSectionPlacement& placement)
{
    switch (placement.kind) {
    case CascadeSectionKind::FirstOrder: {
        const auto& section = dec.first_order[placement.index];
        return {mp_real(1), mp_real(section.sign)};
    }
    case CascadeSectionKind::Biquad: {
        const auto& section = dec.biquads[placement.index];
        return {mp_real(1), -mp_real(section.gamma), mp_real(1)};
    }
    case CascadeSectionKind::Quartic: {
        const auto& section = dec.quartics[placement.index];
        return {mp_real(1), -mp_real(section.alpha), mp_real(section.beta),
                -mp_real(section.alpha), mp_real(1)};
    }
    case CascadeSectionKind::Block: {
        std::vector<mp_real> factor;
        for (double coefficient : dec.blocks[placement.index].coefficients) {
            factor.emplace_back(coefficient);
        }
        return factor;
    }
    }
    return {mp_real(1)};
}

CascadeDecomposition recombine_into_blocks(
    const CascadeDecomposition& source,
    const DirectFIR& fir,
    const std::vector<std::vector<mp_real>>& ordered_precise_factors,
    unsigned max_block_order)
{
    CascadeDecomposition result = source;
    result.first_order.clear();
    result.biquads.clear();
    result.quartics.clear();
    result.blocks.clear();
    result.execution_order.clear();

    static constexpr size_t GRID_SIZE = 65;
    const double pass_edge = std::clamp(
        2.0 * PI * fir.spec.f_pass / fir.spec.fs, 0.0, PI);
    std::vector<std::vector<double>> factor_logs(
        ordered_precise_factors.size(), std::vector<double>(GRID_SIZE, 0.0));
    std::vector<unsigned> factor_orders(ordered_precise_factors.size(), 0u);

    for (size_t i = 0; i < ordered_precise_factors.size(); ++i) {
        const auto& factor = ordered_precise_factors[i];
        factor_orders[i] = factor.empty()
            ? 0u : static_cast<unsigned>(factor.size() - 1);
        mp_real dc_mp = 0;
        for (const mp_real& coefficient : factor) {
            dc_mp += coefficient;
        }
        const double dc = static_cast<double>(dc_mp);
        const double normalizer = (std::isfinite(dc) && std::abs(dc) > 1e-300)
            ? 1.0 / dc : 1.0;
        for (size_t g = 0; g < GRID_SIZE; ++g) {
            const double omega = pass_edge * static_cast<double>(g)
                               / static_cast<double>(GRID_SIZE - 1);
            const std::complex<double> x = std::polar(1.0, -omega);
            std::complex<double> value = 0.0;
            for (size_t k = factor.size(); k-- > 0;) {
                value = value * x + static_cast<double>(factor[k]);
            }
            factor_logs[i][g] = std::log(std::max(
                1e-300, std::abs(normalizer * value)));
        }
    }

    // Form every block independently.  Resetting the cumulative response at
    // a block boundary pairs attenuating and amplifying factors inside the
    // same rounded polynomial instead of leaving a catastrophic cancellation
    // between two different double-precision blocks.
    std::vector<bool> factor_used(ordered_precise_factors.size(), false);
    size_t factors_left = ordered_precise_factors.size();
    while (factors_left > 0) {
        std::vector<mp_real> block_mp = {mp_real(1)};
        std::vector<double> block_log(GRID_SIZE, 0.0);
        unsigned block_order = 0;

        while (true) {
            size_t best = ordered_precise_factors.size();
            double best_score = std::numeric_limits<double>::infinity();
            for (size_t i = 0; i < ordered_precise_factors.size(); ++i) {
                if (factor_used[i]
                    || block_order + factor_orders[i] > max_block_order) {
                    continue;
                }
                double score = 0.0;
                for (size_t g = 0; g < GRID_SIZE; ++g) {
                    score = std::max(
                        score, std::abs(block_log[g] + factor_logs[i][g]));
                }
                if (score < best_score) {
                    best_score = score;
                    best = i;
                }
            }

            if (best == ordered_precise_factors.size()) {
                break;
            }
            factor_used[best] = true;
            --factors_left;
            block_mp = poly_multiply_t(
                block_mp, ordered_precise_factors[best]);
            block_order += factor_orders[best];
            for (size_t g = 0; g < GRID_SIZE; ++g) {
                block_log[g] += factor_logs[best][g];
            }
        }

        CascadeBlock block;
        block.coefficients.reserve(block_mp.size());
        for (const mp_real& coefficient : block_mp) {
            block.coefficients.push_back(static_cast<double>(coefficient));
        }
        result.blocks.push_back(std::move(block));
    }

    // Distribute the mathematical gain across all blocks in the logarithmic
    // domain. Every scaled block receives the same L1 norm, so no single
    // section has to carry a huge compensation factor.
    std::vector<CascadeSectionPlacement> placements;
    std::vector<double> log_l1(result.blocks.size(), 0.0);
    double log_l1_sum = 0.0;
    for (size_t i = 0; i < result.blocks.size(); ++i) {
        double l1 = 0.0;
        for (double coefficient : result.blocks[i].coefficients) {
            l1 += std::abs(coefficient);
        }
        l1 = std::max(l1, 1e-300);
        log_l1[i] = std::log(l1);
        log_l1_sum += log_l1[i];
    }

    const double abs_gain = std::max(1e-300, std::abs(result.gain));
    const double common_log_l1 = result.blocks.empty()
        ? 0.0
        : (std::log(abs_gain) + log_l1_sum)
          / static_cast<double>(result.blocks.size());

    // Order the balanced blocks on the whole Nyquist interval. In the
    // passband the cumulative response is kept close to unity; outside it
    // attenuation is harmless and only positive growth is penalised.
    static constexpr size_t BLOCK_GRID_SIZE = 129;
    std::vector<std::vector<double>> block_logs(
        result.blocks.size(), std::vector<double>(BLOCK_GRID_SIZE, 0.0));
    for (size_t i = 0; i < result.blocks.size(); ++i) {
        const double scale = std::exp(common_log_l1 - log_l1[i]);
        placements.push_back({CascadeSectionKind::Block, i, scale});
        for (size_t g = 0; g < BLOCK_GRID_SIZE; ++g) {
            const double omega = PI * static_cast<double>(g)
                               / static_cast<double>(BLOCK_GRID_SIZE - 1);
            block_logs[i][g] = std::log(std::max(
                1e-300, std::abs(scale * section_value(
                    result, placements.back(), omega))));
        }
    }

    std::vector<bool> block_used(result.blocks.size(), false);
    std::vector<double> cumulative(BLOCK_GRID_SIZE, 0.0);
    for (size_t position = 0; position < result.blocks.size(); ++position) {
        size_t best = result.blocks.size();
        double best_score = std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < result.blocks.size(); ++i) {
            if (block_used[i]) {
                continue;
            }
            double score = 0.0;
            for (size_t g = 0; g < BLOCK_GRID_SIZE; ++g) {
                const double omega = PI * static_cast<double>(g)
                                   / static_cast<double>(BLOCK_GRID_SIZE - 1);
                const double next = cumulative[g] + block_logs[i][g];
                const double local_score = (omega <= pass_edge)
                    ? std::abs(next) : std::max(0.0, next);
                score = std::max(score, local_score);
            }
            if (score < best_score) {
                best_score = score;
                best = i;
            }
        }
        block_used[best] = true;
        result.execution_order.push_back(placements[best]);
        for (size_t g = 0; g < BLOCK_GRID_SIZE; ++g) {
            cumulative[g] += block_logs[best][g];
        }
    }

    result.runtime_gain = (result.gain < 0.0) ? -1.0 : 1.0;
    return result;
}

double cascade_impulse_error(const CascadeDecomposition& dec,
                             const DirectFIR& fir,
                             long double* peak_internal = nullptr)
{
    std::vector<double> impulse(fir.length() * 2u, 0.0);
    impulse[0] = 1.0;
    CascadeFilterState state;
    state.init(dec);
    double max_error = 0.0;
    for (size_t i = 0; i < impulse.size(); ++i) {
        const double output = state.push_double(impulse[i]);
        if (!std::isfinite(output)) {
            if (peak_internal != nullptr) {
                *peak_internal = state.peak_internal;
            }
            return std::numeric_limits<double>::infinity();
        }
        const double reference = (i < fir.h.size()) ? fir.h[i] : 0.0;
        max_error = std::max(max_error, std::abs(output - reference));
    }
    if (peak_internal != nullptr) {
        *peak_internal = state.peak_internal;
    }
    return max_error;
}

unsigned maximum_block_order(const CascadeDecomposition& dec)
{
    unsigned result = 0;
    for (const CascadeBlock& block : dec.blocks) {
        if (!block.coefficients.empty()) {
            result = std::max(
                result,
                static_cast<unsigned>(block.coefficients.size() - 1));
        }
    }
    return result;
}

CascadeDecomposition direct_form_fallback(
    const DirectFIR& fir,
    CascadeDiagnostics diagnostics)
{
    CascadeDecomposition result;
    result.remainder = {1.0};
    result.gain = 1.0;
    result.runtime_gain = 1.0;
    result.diagnostics = diagnostics;

    if (fir.h.empty()) {
        result.diagnostics.status = CascadeBuildStatus::Failed;
        result.diagnostics.runtime_verified = false;
        return result;
    }

    result.blocks.push_back(CascadeBlock{fir.h});
    result.execution_order.push_back(
        {CascadeSectionKind::Block, 0, 1.0});
    result.diagnostics.status = CascadeBuildStatus::DirectFormFallback;
    result.diagnostics.selected_max_block_order = fir.order();
    long double peak_internal = 0.0L;
    result.diagnostics.runtime_impulse_error =
        cascade_impulse_error(result, fir, &peak_internal);
    result.diagnostics.runtime_peak_internal =
        static_cast<double>(peak_internal);
    result.diagnostics.runtime_verified =
        std::isfinite(result.diagnostics.runtime_impulse_error);
    return result;
}

double block_coefficient_error_mp(const CascadeDecomposition& dec,
                                  const DirectFIR& fir)
{
    std::vector<std::vector<mp_real>> factors;
    factors.reserve(dec.blocks.size());
    for (const CascadeBlock& block : dec.blocks) {
        std::vector<mp_real> factor;
        factor.reserve(block.coefficients.size());
        for (double coefficient : block.coefficients) {
            factor.emplace_back(coefficient);
        }
        factors.push_back(std::move(factor));
    }
    std::vector<mp_real> rebuilt = balanced_product(std::move(factors));
    mp_real max_error = 0;
    for (size_t i = 0; i < std::max(rebuilt.size(), fir.h.size()); ++i) {
        const mp_real value = (i < rebuilt.size()) ? rebuilt[i] * dec.gain : 0;
        const mp_real reference = (i < fir.h.size()) ? mp_real(fir.h[i]) : 0;
        max_error = std::max(max_error, abs(value - reference));
    }
    return static_cast<double>(max_error);
}

CascadeDecomposition choose_stable_blocking(
    const CascadeDecomposition& sections,
    const DirectFIR& fir,
    const std::vector<std::vector<mp_real>>& ordered_precise_factors)
{
    static const unsigned preferred_block_orders[] = {
        4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512
    };
    std::vector<unsigned> block_orders;
    for (unsigned order : preferred_block_orders) {
        if (order < fir.order()) {
            block_orders.push_back(order);
        }
    }
    if (fir.order() > 0) {
        block_orders.push_back(fir.order());
    }

    double reference_scale = 0.0;
    for (double coefficient : fir.h) {
        reference_scale = std::max(reference_scale, std::abs(coefficient));
    }
    const double tolerance = std::max(1e-11, reference_scale * 1e-8);

    CascadeDecomposition best;
    CascadeDecomposition high_precision_best;
    bool has_high_precision_candidate = false;
    double best_error = std::numeric_limits<double>::infinity();
    unsigned best_order = 0;
    for (unsigned order : block_orders) {
        CascadeDecomposition candidate = recombine_into_blocks(
            sections, fir, ordered_precise_factors, order);
        long double peak_internal = 0.0L;
        const double error = cascade_impulse_error(
            candidate, fir, &peak_internal);
        candidate.diagnostics.runtime_peak_internal =
            static_cast<double>(peak_internal);
        candidate.diagnostics.runtime_impulse_error = error;
        candidate.diagnostics.runtime_verified =
            std::isfinite(error) && error <= tolerance;
        candidate.diagnostics.selected_max_block_order =
            maximum_block_order(candidate);

        // Prefer a native-precision short cascade whenever one exists. If
        // only a full-order native block is viable, keep the first genuinely
        // short factorization whose rounded coefficients are accurate and
        // whose 50-decimal-digit runtime passes the same impulse criterion.
        if (!has_high_precision_candidate
            && error > tolerance
            && order <= 32
            && order < fir.order()) {
            const double coefficient_error =
                block_coefficient_error_mp(candidate, fir);
            if (std::isfinite(coefficient_error)
                && coefficient_error <= tolerance) {
                candidate.diagnostics.runtime_decimal_digits = 50;
                candidate.diagnostics.status =
                    CascadeBuildStatus::HighPrecisionShortBlockCascade;
                long double mp_peak = 0.0L;
                const double mp_error = cascade_impulse_error(
                    candidate, fir, &mp_peak);
                if (std::isfinite(mp_error) && mp_error <= tolerance) {
                    candidate.diagnostics.runtime_verified = true;
                    candidate.diagnostics.runtime_impulse_error = mp_error;
                    candidate.diagnostics.runtime_peak_internal =
                        static_cast<double>(mp_peak);
                    high_precision_best = candidate;
                    has_high_precision_candidate = true;
                } else {
                    candidate.diagnostics.runtime_decimal_digits = 0;
                    candidate.diagnostics.status =
                        CascadeBuildStatus::Unspecified;
                }
            }
        }
        if (error < best_error) {
            best = std::move(candidate);
            best_error = error;
            best_order = order;
        }
        if (error <= tolerance) {
            CascadeDecomposition accepted = std::move(best);
            if (accepted.diagnostics.selected_max_block_order >= fir.order()
                && has_high_precision_candidate) {
                std::cerr << "decompose_exact_fs_full: selected high-precision "
                          << "short-block cascade (order <= "
                          << high_precision_best.diagnostics.selected_max_block_order
                          << ", impulse max error " << std::scientific
                          << high_precision_best.diagnostics.runtime_impulse_error
                          << ")" << std::defaultfloat << "\n";
                return high_precision_best;
            }
            accepted.diagnostics.status =
                (accepted.diagnostics.selected_max_block_order < fir.order())
                ? CascadeBuildStatus::ShortBlockCascade
                : CascadeBuildStatus::FullOrderFactorBlock;
            std::cerr << "decompose_exact_fs_full: selected block order <= "
                      << order << " (impulse max error " << std::scientific
                      << error << ")" << std::defaultfloat << "\n";
            return accepted;
        }
    }

    std::cerr << "decompose_exact_fs_full: no block size met tolerance; "
              << "using best order " << best_order << " with impulse max error "
              << std::scientific << best_error << std::defaultfloat << "\n";
    CascadeDiagnostics diagnostics = sections.diagnostics;
    diagnostics.runtime_impulse_error = best_error;
    diagnostics.selected_max_block_order = best_order;
    if (has_high_precision_candidate) {
        return high_precision_best;
    }
    return direct_form_fallback(fir, diagnostics);
}

CascadeDecomposition factor_reduced_residual(const DirectFIR& fir,
                                              ComplementReduction reduction)
{
    CascadeDecomposition stage1 = reduction.decomposition;
    if (!stage1.diagnostics.complement_verified) {
        std::cerr << "decompose_exact_fs_full: complementary identity failed; "
                     "using direct-form fallback\n";
        return direct_form_fallback(fir, stage1.diagnostics);
    }
    const int residual_degree = static_cast<int>(reduction.residual.size()) - 1;
    if (residual_degree <= 0) {
        stage1.remainder = {1.0};
        stage1.gain = static_cast<double>(reduction.impulse.front());
        build_balanced_execution_order(stage1, fir);
        stage1.diagnostics.roots_verified = true;
        stage1.diagnostics.root_backward_error = 0.0;
        stage1.diagnostics.root_rebuild_error = 0.0;
        stage1.diagnostics.runtime_impulse_error =
            cascade_impulse_error(stage1, fir);
        stage1.diagnostics.runtime_verified =
            std::isfinite(stage1.diagnostics.runtime_impulse_error)
            && stage1.diagnostics.runtime_impulse_error <= 1e-11;
        if (!stage1.diagnostics.runtime_verified) {
            return direct_form_fallback(fir, stage1.diagnostics);
        }
        stage1.diagnostics.status = CascadeBuildStatus::ShortBlockCascade;
        return stage1;
    }
    if ((residual_degree % 2) != 0) {
        std::cerr << "decompose_exact_fs_full: odd residual degree "
                  << residual_degree << "; using direct-form fallback\n";
        return direct_form_fallback(fir, stage1.diagnostics);
    }

    const size_t M = static_cast<size_t>(residual_degree / 2);
    std::vector<mp_real> chebyshev(M + 1, 0);
    chebyshev[0] = reduction.residual[M];
    for (size_t k = 1; k <= M; ++k) {
        chebyshev[k] = 2 * reduction.residual[M - k];
    }

    RootSolveResult roots = solve_chebyshev_roots(chebyshev);
    stage1.diagnostics.root_iterations = roots.iterations;
    stage1.diagnostics.root_backward_error =
        static_cast<double>(roots.max_backward_error);
    if (!roots.converged || roots.roots.size() != M) {
        std::cerr << "decompose_exact_fs_full: multiprecision root solve failed"
                  << " (degree=" << M
                  << ", iterations=" << roots.iterations
                  << ", backward=" << std::scientific
                  << static_cast<double>(roots.max_backward_error)
                  << std::defaultfloat << "); using direct-form fallback\n";
        return direct_form_fallback(fir, stage1.diagnostics);
    }

    std::cerr << "decompose_exact_fs_full: residual u-degree=" << M
              << ", Aberth iterations=" << roots.iterations
              << ", backward=" << std::scientific
              << static_cast<double>(roots.max_backward_error)
              << std::defaultfloat << "\n";

    std::vector<bool> used(M, false);
    std::vector<std::vector<mp_real>> residual_factors;
    std::vector<std::vector<mp_real>> precise_biquads;
    std::vector<std::vector<mp_real>> precise_quartics;
    precise_biquads.reserve(stage1.biquads.size() + M);
    for (const auto& biquad : stage1.biquads) {
        const mp_real gamma = 2 * cos(
            2 * mp_pi() * biquad.k_index / fir.length());
        precise_biquads.push_back({mp_real(1), -gamma, mp_real(1)});
    }
    static const mp_real real_tolerance("1e-35");
    // Highly clustered roots may have a tiny backward error while their
    // forward conjugacy error is much larger.  Pairing is therefore followed
    // by a full multiprecision coefficient rebuild, which is the authoritative
    // acceptance test.
    static const mp_real pair_tolerance("1e-12");
    const unsigned numerical_k = std::numeric_limits<unsigned>::max();

    for (size_t i = 0; i < M; ++i) {
        if (used[i]) {
            continue;
        }

        const mp_complex u = 2 * roots.roots[i];
        if (abs(u.imag()) <= real_tolerance * (1 + abs(u.real()))) {
            stage1.biquads.push_back(
                Biquad{static_cast<double>(u.real()), numerical_k});
            std::vector<mp_real> factor = {
                mp_real(1), -u.real(), mp_real(1)
            };
            residual_factors.push_back(factor);
            precise_biquads.push_back(std::move(factor));
            used[i] = true;
            continue;
        }

        size_t best = M;
        mp_real best_distance = std::numeric_limits<mp_real>::max();
        for (size_t j = 0; j < M; ++j) {
            if (j == i || used[j]) {
                continue;
            }
            const mp_real distance = abs(roots.roots[j] - conj(roots.roots[i]));
            if (distance < best_distance) {
                best_distance = distance;
                best = j;
            }
        }

        if (best == M
            || best_distance > pair_tolerance * (1 + abs(roots.roots[i]))) {
            std::cerr << "decompose_exact_fs_full: unmatched complex u-root"
                      << " (nearest relative distance=" << std::scientific
                      << static_cast<double>(best_distance / (1 + abs(roots.roots[i])))
                      << std::defaultfloat << "); "
                         "using direct-form fallback\n";
            return direct_form_fallback(fir, stage1.diagnostics);
        }

        const mp_complex paired_u = 2 * conj(roots.roots[best]);
        const mp_complex averaged_u = (u + paired_u) / 2;
        const mp_real alpha = 2 * averaged_u.real();
        const mp_real beta = 2 + norm(averaged_u);
        stage1.quartics.push_back(
            Quartic{static_cast<double>(alpha), static_cast<double>(beta)});
        std::vector<mp_real> factor = {
            mp_real(1), -alpha, beta, -alpha, mp_real(1)
        };
        residual_factors.push_back(factor);
        precise_quartics.push_back(std::move(factor));
        used[i] = true;
        used[best] = true;
    }

    std::vector<mp_real> rebuilt_residual =
        balanced_product(std::move(residual_factors));
    for (mp_real& value : rebuilt_residual) {
        value *= reduction.residual.front();
    }
    mp_real root_rebuild_error = 0;
    if (rebuilt_residual.size() != reduction.residual.size()) {
        root_rebuild_error = 1;
    } else {
        for (size_t i = 0; i < rebuilt_residual.size(); ++i) {
            root_rebuild_error = std::max(
                root_rebuild_error,
                abs(rebuilt_residual[i] - reduction.residual[i]));
        }
        root_rebuild_error /= std::max(
            mp_real("1e-90"), max_abs(reduction.residual));
    }

    std::cerr << "decompose_exact_fs_full: root-set rebuild="
              << std::scientific << static_cast<double>(root_rebuild_error)
              << std::defaultfloat << "\n";
    stage1.diagnostics.root_rebuild_error =
        static_cast<double>(root_rebuild_error);
    stage1.diagnostics.roots_verified =
        root_rebuild_error <= mp_real("1e-45");
    if (root_rebuild_error > mp_real("1e-45")) {
        std::cerr << "decompose_exact_fs_full: root set does not reconstruct "
                     "the reduced polynomial; using direct-form fallback\n";
        return direct_form_fallback(fir, stage1.diagnostics);
    }

    stage1.remainder = {1.0};
    stage1.gain = static_cast<double>(reduction.impulse.front());
    build_balanced_execution_order(stage1, fir);

    std::vector<std::vector<mp_real>> ordered_precise_factors;
    ordered_precise_factors.reserve(stage1.execution_order.size());
    for (const auto& placement : stage1.execution_order) {
        switch (placement.kind) {
        case CascadeSectionKind::FirstOrder: {
            const int sign = stage1.first_order[placement.index].sign;
            ordered_precise_factors.push_back({mp_real(1), mp_real(sign)});
            break;
        }
        case CascadeSectionKind::Biquad:
            ordered_precise_factors.push_back(
                precise_biquads[placement.index]);
            break;
        case CascadeSectionKind::Quartic:
            ordered_precise_factors.push_back(
                precise_quartics[placement.index]);
            break;
        case CascadeSectionKind::Block:
            ordered_precise_factors.push_back(
                placement_factor_mp(stage1, placement));
            break;
        }
    }

    std::cerr << "decompose_exact_fs_full: sections fo="
              << stage1.first_order.size()
              << ", bq=" << stage1.biquads.size()
              << ", qt=" << stage1.quartics.size()
              << ", gain=" << std::scientific << stage1.gain
              << ", runtime_gain=" << stage1.runtime_gain
              << std::defaultfloat << "\n";
    return choose_stable_blocking(stage1, fir, ordered_precise_factors);
}

} // namespace

CascadeDecomposition decompose_exact_fs(const DirectFIR& fir,
                                        bool interleave_order)
{
    return reduce_by_complement(fir, interleave_order).decomposition;
}

CascadeDecomposition decompose_exact_fs_full(const DirectFIR& fir,
                                             bool interleave_order)
{
    return factor_reduced_residual(
        fir, reduce_by_complement(fir, interleave_order));
}

CascadeDecomposition decompose_exact_fs_structured(const DirectFIR& fir,
                                                   bool interleave_order)
{
    // The former greedy residual deflation was neither structured nor stable.
    // The full path now performs a simultaneous, structure-preserving solve.
    return decompose_exact_fs_full(fir, interleave_order);
}
