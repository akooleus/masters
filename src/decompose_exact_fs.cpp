// ═══════════════════════════════════════════════════════════════
//  decompose_exact_fs.cpp
//
//  Exact-first decomposition family for frequency-sampling FIR filters.
//
//  This file contains three public entry points:
//    1. decompose_exact_fs            — exact stage-1 extraction
//    2. decompose_exact_fs_structured — conservative residual reduction
//    3. decompose_exact_fs_full       — best-effort fuller factorization
//
//  The core idea is exact-first:
//    detect structurally known zero-valued H[k] samples,
//    convert them into exact UC factors,
//    extract them stably in __float128,
//    and only then touch the smaller residual polynomial.
// ═══════════════════════════════════════════════════════════════

#include "cascade_fir.h"

#include <quadmath.h>

#include <complex>
#include <iostream>
#include <limits>
#include <vector>

namespace {

using quad = __float128;

void enforce_palindromic_symmetry_q128(std::vector<quad>& h);

using cx = std::complex<long double>;

extern "C" {
void dgeev_(const char* jobvl, const char* jobvr,
            const int* n, double* a, const int* lda,
            double* wr, double* wi,
            double* vl, const int* ldvl,
            double* vr, const int* ldvr,
            double* work, const int* lwork, int* info);
}

struct ExactStopbandInfo {
    std::vector<unsigned> pair_k;
    bool has_dc_zero = false;
    bool has_nyq_zero = false;
};

ExactStopbandInfo find_exact_stopband_zeros(const DirectFIR& fir)
{
    ExactStopbandInfo info;

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
    for (unsigned k = 0; k <= k_half; ++k) {
        if (mags[k] > zero_tol) {
            continue;
        }

        if (k == 0) {
            info.has_dc_zero = true;
        } else if (N % 2 == 0 && k == k_half) {
            info.has_nyq_zero = true;
        } else {
            info.pair_k.push_back(k);
        }
    }

    return info;
}

std::vector<unsigned> interleave_indices(std::vector<unsigned> indices)
{
    std::sort(indices.begin(), indices.end());

    std::vector<unsigned> order;
    order.reserve(indices.size());

    if (indices.empty()) {
        return order;
    }

    size_t lo = 0;
    size_t hi = indices.size() - 1;
    bool take_lo = true;

    while (lo <= hi) {
        if (take_lo) {
            order.push_back(indices[lo]);
            ++lo;
        } else {
            order.push_back(indices[hi]);
            if (hi == 0) {
                break;
            }
            --hi;
        }
        take_lo = !take_lo;
    }

    return order;
}

std::vector<quad> q128_poly_multiply(const std::vector<quad>& a,
                                     const std::vector<quad>& b)
{
    if (a.empty() || b.empty()) {
        return {};
    }

    std::vector<quad> c(a.size() + b.size() - 1, 0.0Q);
    for (size_t i = 0; i < a.size(); ++i) {
        for (size_t j = 0; j < b.size(); ++j) {
            c[i + j] += a[i] * b[j];
        }
    }
    return c;
}

std::vector<quad> build_exact_factor_product_q128(const CascadeDecomposition& dec)
{
    std::vector<quad> product = {1.0Q};

    for (const auto& fo : dec.first_order) {
        const std::vector<quad> factor = {
            1.0Q,
            static_cast<quad>(fo.sign)
        };
        product = q128_poly_multiply(product, factor);
    }

    for (const auto& bq : dec.biquads) {
        const std::vector<quad> factor = {
            1.0Q,
            -static_cast<quad>(bq.gamma),
            1.0Q
        };
        product = q128_poly_multiply(product, factor);
    }

    return product;
}

std::pair<std::vector<quad>, std::vector<quad>>
q128_poly_divide(const std::vector<quad>& a,
                 const std::vector<quad>& b)
{
    const size_t na = a.size();
    const size_t nb = b.size();

    if (nb == 0 || na < nb || fabsq(b[0]) < 1e-34Q) {
        return {{}, a};
    }

    std::vector<quad> r = a;
    const size_t qlen = na - nb + 1;
    std::vector<quad> q(qlen, 0.0Q);

    for (size_t i = 0; i < qlen; ++i) {
        q[i] = r[i] / b[0];
        for (size_t j = 0; j < nb; ++j) {
            r[i + j] -= q[i] * b[j];
        }
    }

    std::vector<quad> rem(r.begin() + static_cast<long>(qlen), r.end());
    return {std::move(q), std::move(rem)};
}

void q128_first_order_half_divide(const std::vector<quad>& h,
                                  int sign,
                                  std::vector<quad>& q)
{
    const size_t len = h.size();
    if (len < 2) {
        q = h;
        return;
    }

    const size_t qlen = len - 1;
    q.resize(qlen);

    const size_t mid = (qlen - 1) / 2;
    const quad s = static_cast<quad>(sign);

    q[0] = h[0];
    for (size_t n = 1; n <= mid; ++n) {
        q[n] = h[n] - s * q[n - 1];
    }

    for (size_t n = 0; n < qlen / 2; ++n) {
        q[qlen - 1 - n] = q[n];
    }
}

void q128_biquad_half_divide(const std::vector<quad>& h,
                             quad gamma,
                             std::vector<quad>& q)
{
    const size_t len = h.size();
    if (len < 4) {
        q = h;
        return;
    }

    const size_t qlen = len - 2;
    q.resize(qlen);

    const size_t mid = (qlen - 1) / 2;

    q[0] = h[0];
    if (qlen >= 2) {
        q[1] = h[1] + gamma * q[0];
    }
    for (size_t n = 2; n <= mid; ++n) {
        q[n] = h[n] + gamma * q[n - 1] - q[n - 2];
    }

    if (qlen % 2 == 0 && qlen >= 4) {
        const size_t center = qlen / 2;
        if (center > mid) {
            q[center] = h[center] + gamma * q[center - 1] - q[center - 2];
        }
    }

    for (size_t n = 0; n < qlen / 2; ++n) {
        q[qlen - 1 - n] = q[n];
    }
}

std::vector<quad> q128_compute_U_sequence(size_t count, quad gamma)
{
    std::vector<quad> U(count, 0.0Q);
    if (count == 0) {
        return U;
    }

    U[0] = 1.0Q;
    if (count == 1) {
        return U;
    }

    U[1] = gamma;
    for (size_t n = 2; n < count; ++n) {
        U[n] = gamma * U[n - 1] - U[n - 2];
    }
    return U;
}

quad q128_biquad_divisibility_delta(const std::vector<quad>& h,
                                    quad gamma)
{
    const size_t len = h.size();
    if (len < 3) {
        return 0.0Q;
    }

    const size_t max_u = len - 2;
    const std::vector<quad> U = q128_compute_U_sequence(max_u, gamma);

    quad sum = -h[0];
    for (size_t m = 0; m <= len - 3; ++m) {
        sum += h[m] * U[len - 3 - m];
    }
    return sum;
}

void q128_correct_for_biquad_divisibility(std::vector<quad>& h,
                                          quad gamma)
{
    const size_t len = h.size();
    if (len < 3 || len % 2 == 0) {
        return;
    }

    const quad delta = q128_biquad_divisibility_delta(h, gamma);
    quad h_scale = 0.0Q;
    for (quad x : h) {
        const quad ax = fabsq(x);
        if (ax > h_scale) {
            h_scale = ax;
        }
    }

    h_scale = std::max(1e-30Q, h_scale);
    if (fabsq(delta) <= 1e-28Q * h_scale) {
        return;
    }

    const size_t center = len / 2;
    const size_t u_idx = len - 3 - center;
    const std::vector<quad> U = q128_compute_U_sequence(len - 2, gamma);
    const quad denom = U[u_idx];

    if (fabsq(denom) <= 1e-30Q) {
        return;
    }

    h[center] -= delta / denom;
    enforce_palindromic_symmetry_q128(h);
}

double q128_relative_rebuild_error(const std::vector<quad>& original,
                                   const std::vector<quad>& quotient,
                                   quad gamma)
{
    const std::vector<quad> factor = {1.0Q, -gamma, 1.0Q};
    const std::vector<quad> rebuilt = q128_poly_multiply(quotient, factor);

    const size_t len = std::max(original.size(), rebuilt.size());
    quad max_abs = 0.0Q;
    quad max_ref = 0.0Q;

    for (size_t i = 0; i < len; ++i) {
        const quad ref = (i < original.size()) ? original[i] : 0.0Q;
        const quad cur = (i < rebuilt.size()) ? rebuilt[i] : 0.0Q;
        const quad err = fabsq(ref - cur);
        if (err > max_abs) {
            max_abs = err;
        }
        const quad aref = fabsq(ref);
        if (aref > max_ref) {
            max_ref = aref;
        }
    }

    max_ref = std::max(max_ref, 1e-30Q);
    return static_cast<double>(max_abs / max_ref);
}

double q128_relative_rebuild_error_multi(
    const std::vector<quad>& original,
    const std::vector<quad>& quotient,
    const std::vector<quad>& gammas,
    unsigned n_factors)
{
    std::vector<quad> rebuilt = quotient;
    for (unsigned i = 0; i < n_factors; ++i) {
        rebuilt = q128_poly_multiply(rebuilt, {1.0Q, -gammas[i], 1.0Q});
    }

    const size_t len = std::max(original.size(), rebuilt.size());
    quad max_abs = 0.0Q;
    quad max_ref = 0.0Q;

    for (size_t i = 0; i < len; ++i) {
        const quad ref = (i < original.size()) ? original[i] : 0.0Q;
        const quad cur = (i < rebuilt.size()) ? rebuilt[i] : 0.0Q;
        const quad err = fabsq(ref - cur);
        if (err > max_abs) {
            max_abs = err;
        }
        const quad aref = fabsq(ref);
        if (aref > max_ref) {
            max_ref = aref;
        }
    }

    max_ref = std::max(max_ref, 1e-30Q);
    return static_cast<double>(max_abs / max_ref);
}

struct ExactPrefixTrial {
    unsigned accepted_biquad_count = 0;
    double rel_err = std::numeric_limits<double>::infinity();
    std::vector<quad> quotient;
};

ExactPrefixTrial run_exact_prefix_trial(const std::vector<quad>& base_after_fo,
                                        const std::vector<quad>& gamma_order_q,
                                        unsigned refresh_interval,
                                        double quality_threshold)
{
    ExactPrefixTrial out;
    out.quotient = base_after_fo;

    auto seq_divide_known_biquads = [&](unsigned count) -> std::vector<quad> {
        std::vector<quad> cur = base_after_fo;
        std::vector<quad> q;
        for (unsigned j = 0; j < count; ++j) {
            std::vector<quad> trial = cur;
            q128_correct_for_biquad_divisibility(trial, gamma_order_q[j]);
            q128_biquad_half_divide(trial, gamma_order_q[j], q);
            enforce_palindromic_symmetry_q128(q);
            cur = std::move(q);
        }
        return cur;
    };

    unsigned last_good = 0;
    std::vector<quad> best_quotient = base_after_fo;
    std::vector<quad> current = base_after_fo;
    double best_err = 0.0;

    for (unsigned i = 0; i < gamma_order_q.size(); ++i) {
        std::vector<quad> trial = current;
        q128_correct_for_biquad_divisibility(trial, gamma_order_q[i]);
        std::vector<quad> next_q;
        q128_biquad_half_divide(trial, gamma_order_q[i], next_q);
        enforce_palindromic_symmetry_q128(next_q);
        current = std::move(next_q);

        const unsigned steps_in_block = (i + 1) - last_good;
        const bool do_refresh =
            (steps_in_block >= refresh_interval)
            || (i + 1 == gamma_order_q.size());

        if (!do_refresh) {
            continue;
        }

        std::vector<quad> fresh = seq_divide_known_biquads(i + 1);
        const double rel_err =
            q128_relative_rebuild_error_multi(base_after_fo, fresh, gamma_order_q, i + 1);

        if (rel_err > quality_threshold) {
            bool found = false;
            for (unsigned try_end = i; try_end > last_good; --try_end) {
                std::vector<quad> trial_q = seq_divide_known_biquads(try_end);
                const double trial_err =
                    q128_relative_rebuild_error_multi(
                        base_after_fo, trial_q, gamma_order_q, try_end);
                if (trial_err <= quality_threshold) {
                    best_quotient = std::move(trial_q);
                    out.accepted_biquad_count = try_end;
                    out.rel_err = trial_err;
                    found = true;
                    break;
                }
            }
            if (!found) {
                out.accepted_biquad_count = last_good;
                out.rel_err = best_err;
            }
            out.quotient = std::move(best_quotient);
            return out;
        }

        current = std::move(fresh);
        best_quotient = current;
        last_good = i + 1;
        best_err = rel_err;
        out.accepted_biquad_count = last_good;
        out.rel_err = best_err;
    }

    out.quotient = std::move(best_quotient);
    return out;
}

void enforce_palindromic_symmetry_q128(std::vector<quad>& h)
{
    for (size_t i = 0; i < h.size() / 2; ++i) {
        const quad avg = 0.5Q * (h[i] + h[h.size() - 1 - i]);
        h[i] = avg;
        h[h.size() - 1 - i] = avg;
    }
}

quad max_abs_value_q128(const std::vector<quad>& v)
{
    quad m = 0.0Q;
    for (quad x : v) {
        const quad ax = fabsq(x);
        if (ax > m) {
            m = ax;
        }
    }
    return m;
}

std::vector<quad> real_to_q128(const std::vector<real_t>& v)
{
    std::vector<quad> out(v.size(), 0.0Q);
    for (size_t i = 0; i < v.size(); ++i) {
        out[i] = static_cast<quad>(v[i]);
    }
    return out;
}

std::vector<real_t> q128_to_real(const std::vector<quad>& v)
{
    std::vector<real_t> out(v.size(), 0.0);
    for (size_t i = 0; i < v.size(); ++i) {
        out[i] = static_cast<real_t>(v[i]);
    }
    return out;
}

std::vector<quad> q128_palindrome_to_u_poly(const std::vector<quad>& Q)
{
    int deg = static_cast<int>(Q.size()) - 1;
    if (deg < 0) {
        return {};
    }
    if (deg % 2 != 0) {
        return {};
    }

    const int M = deg / 2;
    if (M == 0) {
        return {Q[0]};
    }

    std::vector<quad> F(static_cast<size_t>(M + 1), 0.0Q);
    F[0] = Q[static_cast<size_t>(M)];

    std::vector<quad> s_prev2 = {2.0Q};
    std::vector<quad> s_prev1 = {0.0Q, 1.0Q};

    F[0] += Q[static_cast<size_t>(M) - 1] * s_prev1[0];
    if (M >= 1) {
        F[1] += Q[static_cast<size_t>(M) - 1] * s_prev1[1];
    }

    for (int k = 2; k <= M; ++k) {
        const size_t len_k = static_cast<size_t>(k + 1);
        std::vector<quad> s_cur(len_k, 0.0Q);

        for (size_t j = 1; j < len_k && j - 1 < s_prev1.size(); ++j) {
            s_cur[j] = s_prev1[j - 1];
        }
        for (size_t j = 0; j < s_prev2.size() && j < len_k; ++j) {
            s_cur[j] -= s_prev2[j];
        }

        const quad weight = Q[static_cast<size_t>(M - k)];
        for (size_t j = 0; j < len_k && j < F.size(); ++j) {
            F[j] += weight * s_cur[j];
        }

        s_prev2 = std::move(s_prev1);
        s_prev1 = std::move(s_cur);
    }

    return F;
}

std::vector<cx> companion_eigenvalues(const std::vector<double>& coeffs)
{
    const int M = static_cast<int>(coeffs.size()) - 1;
    if (M <= 0) {
        return {};
    }
    if (M == 1) {
        return {cx(-static_cast<long double>(coeffs[0]) / static_cast<long double>(coeffs[1]),
                   0.0L)};
    }

    auto coeff_spread = [](const std::vector<double>& c, double log2_scale) {
        double min_log2 = std::numeric_limits<double>::infinity();
        double max_log2 = -std::numeric_limits<double>::infinity();
        for (size_t k = 0; k < c.size(); ++k) {
            const double ak = std::abs(c[k]);
            if (ak < 1e-300) {
                continue;
            }
            const double lv = std::log2(ak) + static_cast<double>(k) * log2_scale;
            min_log2 = std::min(min_log2, lv);
            max_log2 = std::max(max_log2, lv);
        }
        if (!std::isfinite(min_log2) || !std::isfinite(max_log2)) {
            return std::numeric_limits<double>::infinity();
        }
        return max_log2 - min_log2;
    };

    double best_log2_scale = 0.0;
    double best_spread = coeff_spread(coeffs, 0.0);
    for (int half_step = -64; half_step <= 64; ++half_step) {
        const double log2_scale = 0.5 * static_cast<double>(half_step);
        const double spread = coeff_spread(coeffs, log2_scale);
        if (spread < best_spread) {
            best_spread = spread;
            best_log2_scale = log2_scale;
        }
    }

    const double u_scale = std::exp2(best_log2_scale);
    std::vector<double> balanced = coeffs;
    if (std::abs(best_log2_scale) > 1e-12) {
        double scale_pow = 1.0;
        for (double& ck : balanced) {
            ck *= scale_pow;
            scale_pow *= u_scale;
        }
    }

    double lc = balanced[static_cast<size_t>(M)];
    int n = M;
    std::vector<double> A(static_cast<size_t>(n * n), 0.0);

    for (int i = 0; i < n; ++i) {
        A[static_cast<size_t>(i + (n - 1) * n)] = -balanced[static_cast<size_t>(i)] / lc;
        if (i + 1 < n) {
            A[static_cast<size_t>((i + 1) + i * n)] = 1.0;
        }
    }

    std::vector<double> wr(static_cast<size_t>(n));
    std::vector<double> wi(static_cast<size_t>(n));

    double work_query = 0.0;
    int lwork = -1;
    int info = 0;
    int one = 1;

    dgeev_("N", "N", &n, A.data(), &n,
           wr.data(), wi.data(),
           nullptr, &one, nullptr, &one,
           &work_query, &lwork, &info);

    lwork = static_cast<int>(work_query);
    std::vector<double> work(static_cast<size_t>(lwork));

    dgeev_("N", "N", &n, A.data(), &n,
           wr.data(), wi.data(),
           nullptr, &one, nullptr, &one,
           work.data(), &lwork, &info);

    if (info != 0) {
        std::cerr << "  exact_fs_full: dgeev failed, info=" << info << "\n";
        return {};
    }

    std::vector<cx> roots(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        roots[static_cast<size_t>(i)] =
            cx(static_cast<long double>(wr[static_cast<size_t>(i)]),
               static_cast<long double>(wi[static_cast<size_t>(i)]))
            * static_cast<long double>(u_scale);
    }
    return roots;
}

std::pair<std::complex<long double>, std::complex<long double>>
eval_poly_and_derivative_q128(const std::vector<quad>& coeffs,
                              std::complex<long double> z)
{
    std::complex<long double> p = 0.0L;
    std::complex<long double> dp = 0.0L;

    for (size_t i = coeffs.size(); i-- > 0;) {
        dp = dp * z + p;
        p = p * z + static_cast<long double>(coeffs[i]);
    }

    return {p, dp};
}

void polish_u_roots_q128(const std::vector<quad>& coeffs_q,
                         std::vector<cx>& roots)
{
    if (coeffs_q.size() < 2 || roots.empty()) {
        return;
    }

    for (cx& root : roots) {
        std::complex<long double> z(
            static_cast<long double>(root.real()),
            static_cast<long double>(root.imag()));

        for (int iter = 0; iter < 12; ++iter) {
            const auto [p, dp] = eval_poly_and_derivative_q128(coeffs_q, z);
            const long double dp_norm = std::abs(dp);
            if (!(dp_norm > 1e-24L)) {
                break;
            }

            std::complex<long double> step = p / dp;
            const long double step_abs = std::abs(step);
            const long double z_scale = 1.0L + std::abs(z);
            if (step_abs > 0.5L * z_scale) {
                step *= (0.5L * z_scale / step_abs);
            }

            z -= step;
            if (step_abs < 1e-22L * z_scale) {
                break;
            }
        }

        long double re = z.real();
        long double im = z.imag();
        if (std::abs(im) < 1e-18L * (1.0L + std::abs(re))) {
            im = 0.0L;
        }
        root = cx(re, im);
    }
}

unsigned reduce_residual_u_roots(CascadeDecomposition& dec,
                                 double div_tol,
                                 double global_tol,
                                 unsigned max_accept)
{
    if (dec.remainder.size() <= 1) {
        return 0;
    }

    const int deg = static_cast<int>(dec.remainder.size()) - 1;
    if (deg <= 0 || (deg % 2) != 0) {
        return 0;
    }

    const std::vector<quad> Q_q = real_to_q128(dec.remainder);
    const std::vector<quad> F_q = q128_palindrome_to_u_poly(Q_q);
    if (F_q.size() <= 1) {
        return 0;
    }

    static constexpr double REAL_TOL = 1e-8;
    static constexpr double PAIR_TOL = 1e-6;

    struct Candidate {
        enum Type { BQ, QT } type;
        double gamma = 0.0;
        double alpha = 0.0;
        double beta = 0.0;
        double quality = std::numeric_limits<double>::infinity();
        std::vector<quad> factor_u;
        std::vector<quad> factor_z;
    };

    struct AcceptedFactor {
        Candidate::Type type;
        double gamma = 0.0;
        double alpha = 0.0;
        double beta = 0.0;
        std::vector<quad> factor_z;
    };

    const std::vector<quad> Q_base = Q_q;
    std::vector<AcceptedFactor> accepted_factors;
    std::vector<quad> Q_work = Q_base;
    std::vector<quad> F_work = F_q;
    std::vector<Biquad> numerical_bq;
    std::vector<Quartic> numerical_qt;

    unsigned accepted = 0;

    auto refresh_from_base = [&](std::vector<quad>& Q_next,
                                 std::vector<quad>& F_next) -> bool
    {
        std::vector<quad> cur = Q_base;
        for (const auto& f : accepted_factors) {
            auto [q_trial, rem_trial] = q128_poly_divide(cur, f.factor_z);
            const double rel_Q = static_cast<double>(
                max_abs_value_q128(rem_trial)
                / std::max(1e-30Q, max_abs_value_q128(cur)));
            if (rel_Q > div_tol * 10.0) {
                return false;
            }
            enforce_palindromic_symmetry_q128(q_trial);
            cur = std::move(q_trial);
        }

        if (cur.size() <= 1) {
            Q_next = std::move(cur);
            F_next = {1.0Q};
            return true;
        }

        const int cur_deg = static_cast<int>(cur.size()) - 1;
        if ((cur_deg % 2) != 0) {
            return false;
        }

        Q_next = cur;
        F_next = q128_palindrome_to_u_poly(Q_next);
        return !F_next.empty();
    };

    auto cumulative_rebuild_error = [&](const std::vector<quad>& quotient) {
        std::vector<quad> rebuilt = quotient;
        for (const auto& f : accepted_factors) {
            rebuilt = q128_poly_multiply(rebuilt, f.factor_z);
        }

        const size_t len = std::max(Q_base.size(), rebuilt.size());
        quad max_abs = 0.0Q;
        quad max_ref = 1e-30Q;
        for (size_t i = 0; i < len; ++i) {
            const quad ref = (i < Q_base.size()) ? Q_base[i] : 0.0Q;
            const quad cur = (i < rebuilt.size()) ? rebuilt[i] : 0.0Q;
            max_abs = std::max(max_abs, fabsq(ref - cur));
            max_ref = std::max(max_ref, fabsq(ref));
        }
        return static_cast<double>(max_abs / max_ref);
    };

    while (F_work.size() > 1) {
        if (max_accept > 0 && accepted >= max_accept) {
            break;
        }

        std::vector<double> Fd(F_work.size(), 0.0);
        for (size_t i = 0; i < F_work.size(); ++i) {
            Fd[i] = static_cast<double>(F_work[i]);
        }

        std::vector<cx> u_roots = companion_eigenvalues(Fd);
        if (u_roots.empty()) {
            break;
        }
        polish_u_roots_q128(F_work, u_roots);

        std::vector<Candidate> candidates;
        std::vector<bool> used(u_roots.size(), false);

        for (size_t i = 0; i < u_roots.size(); ++i) {
            if (std::abs(u_roots[i].imag()) < REAL_TOL) {
                const long double gamma_ld = u_roots[i].real();
                Candidate c;
                c.type = Candidate::BQ;
                c.gamma = static_cast<double>(gamma_ld);
                c.quality = static_cast<double>(std::abs(u_roots[i].imag()));
                c.factor_u = {-static_cast<quad>(gamma_ld), 1.0Q};
                c.factor_z = {1.0Q, -static_cast<quad>(gamma_ld), 1.0Q};
                candidates.push_back(c);
                used[i] = true;
            }
        }

        for (size_t i = 0; i < u_roots.size(); ++i) {
            if (used[i] || std::abs(u_roots[i].imag()) < REAL_TOL) {
                continue;
            }

            size_t best_j = u_roots.size();
            double best_dist = std::numeric_limits<double>::infinity();
            for (size_t j = i + 1; j < u_roots.size(); ++j) {
                if (used[j] || std::abs(u_roots[j].imag()) < REAL_TOL) {
                    continue;
                }
                if (u_roots[i].imag() * u_roots[j].imag() > 0.0) {
                    continue;
                }
                const double dist = std::abs(u_roots[j] - std::conj(u_roots[i]));
                if (dist < best_dist) {
                    best_dist = dist;
                    best_j = j;
                }
            }

            const double pair_tol = PAIR_TOL * (1.0 + std::abs(u_roots[i]));
            if (best_j < u_roots.size() && best_dist < pair_tol) {
                const long double re_u =
                    0.5L * (u_roots[i].real() + u_roots[best_j].real());
                const long double im_u =
                    0.5L * std::abs(u_roots[i].imag() - u_roots[best_j].imag());
                const long double alpha_ld = 2.0L * re_u;
                const long double beta_ld = 2.0L + re_u * re_u + im_u * im_u;
                Candidate c;
                c.type = Candidate::QT;
                c.alpha = static_cast<double>(alpha_ld);
                c.beta = static_cast<double>(beta_ld);
                c.quality = static_cast<double>(best_dist);
                c.factor_u = {
                    static_cast<quad>(beta_ld - 2.0L),
                    -static_cast<quad>(alpha_ld),
                    1.0Q
                };
                c.factor_z = {
                    1.0Q,
                    -static_cast<quad>(alpha_ld),
                    static_cast<quad>(beta_ld),
                    -static_cast<quad>(alpha_ld),
                    1.0Q
                };
                candidates.push_back(c);
                used[i] = true;
                used[best_j] = true;
            }
        }

        std::sort(candidates.begin(), candidates.end(),
                  [](const Candidate& a, const Candidate& b) {
                      return a.quality < b.quality;
                  });

        if (candidates.empty()) {
            break;
        }

        size_t best_idx = candidates.size();
        double best_score = std::numeric_limits<double>::infinity();
        std::vector<quad> best_Q_next;
        std::vector<quad> best_F_next;
        Candidate best_cand;

        for (size_t ci = 0; ci < candidates.size(); ++ci) {
            const Candidate& cand = candidates[ci];
            if (cand.factor_u.size() > F_work.size() || cand.factor_z.size() > Q_work.size()) {
                continue;
            }

            auto [F_trial, F_rem] = q128_poly_divide(F_work, cand.factor_u);
            auto [Q_trial, Q_rem] = q128_poly_divide(Q_work, cand.factor_z);

            const double rel_F = static_cast<double>(
                max_abs_value_q128(F_rem) / std::max(1e-30Q, max_abs_value_q128(F_work)));
            const double rel_Q = static_cast<double>(
                max_abs_value_q128(Q_rem) / std::max(1e-30Q, max_abs_value_q128(Q_work)));

            double sym_err = 0.0;
            for (size_t i = 0; i < Q_trial.size() / 2; ++i) {
                sym_err = std::max(sym_err, static_cast<double>(
                    fabsq(Q_trial[i] - Q_trial[Q_trial.size() - 1 - i])));
            }

            enforce_palindromic_symmetry_q128(Q_trial);
            const double global_err = (cand.type == Candidate::BQ)
                ? q128_relative_rebuild_error(Q_work, Q_trial, static_cast<quad>(cand.gamma))
                : q128_relative_rebuild_error_multi(
                    Q_work, Q_trial,
                    {static_cast<quad>(cand.alpha), static_cast<quad>(cand.beta)},
                    0u);

            double score = std::max({rel_F, rel_Q, sym_err, global_err});
            if (cand.type == Candidate::QT) {
                const std::vector<quad> rebuilt = q128_poly_multiply(
                    Q_trial,
                    cand.factor_z);
                quad max_abs = 0.0Q;
                quad max_ref = 1e-30Q;
                for (size_t i = 0; i < std::max(Q_work.size(), rebuilt.size()); ++i) {
                    const quad ref = (i < Q_work.size()) ? Q_work[i] : 0.0Q;
                    const quad cur = (i < rebuilt.size()) ? rebuilt[i] : 0.0Q;
                    max_abs = std::max(max_abs, fabsq(ref - cur));
                    max_ref = std::max(max_ref, fabsq(ref));
                }
                score = std::max(score, static_cast<double>(max_abs / max_ref));
            }

            if (rel_F <= div_tol && rel_Q <= div_tol && score < best_score) {
                best_score = score;
                best_idx = ci;
                best_Q_next = std::move(Q_trial);
                best_F_next = std::move(F_trial);
                best_cand = cand;
            }
        }

        if (best_idx == candidates.size() || best_score > global_tol) {
            break;
        }

        if (best_cand.type == Candidate::BQ) {
            numerical_bq.push_back(Biquad{best_cand.gamma, 0u});
        } else {
            numerical_qt.push_back(Quartic{best_cand.alpha, best_cand.beta});
        }

        accepted_factors.push_back(AcceptedFactor{
            best_cand.type,
            best_cand.gamma,
            best_cand.alpha,
            best_cand.beta,
            best_cand.factor_z
        });

        if (!refresh_from_base(best_Q_next, best_F_next)) {
            if (best_cand.type == Candidate::BQ) {
                numerical_bq.pop_back();
            } else {
                numerical_qt.pop_back();
            }
            accepted_factors.pop_back();
            break;
        }

        const double cumulative_err = cumulative_rebuild_error(best_Q_next);
        if (cumulative_err > global_tol) {
            if (best_cand.type == Candidate::BQ) {
                numerical_bq.pop_back();
            } else {
                numerical_qt.pop_back();
            }
            accepted_factors.pop_back();
            break;
        }

        Q_work = std::move(best_Q_next);
        F_work = std::move(best_F_next);
        ++accepted;
    }

    for (const auto& bq : numerical_bq) {
        dec.biquads.push_back(bq);
    }
    for (const auto& qt : numerical_qt) {
        dec.quartics.push_back(qt);
    }

    dec.remainder = q128_to_real(Q_work);
    return accepted;
}

double compute_rebuild_max_err(const CascadeDecomposition& dec,
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

double exact_return_err_limit(double err_stage1)
{
    return std::max(1e-6, 10.0 * err_stage1);
}

} // namespace

CascadeDecomposition decompose_exact_fs(const DirectFIR& fir,
                                        bool interleave_order)
{
    CascadeDecomposition dec;
    dec.gain = 1.0;

    if (fir.h.empty()) {
        dec.remainder = {};
        return dec;
    }

    const ExactStopbandInfo info = find_exact_stopband_zeros(fir);
    std::vector<unsigned> pair_order = info.pair_k;
    if (interleave_order) {
        pair_order = interleave_indices(std::move(pair_order));
    } else {
        std::sort(pair_order.begin(), pair_order.end());
    }

    if (info.has_dc_zero) {
        dec.first_order.push_back(FirstOrder{-1});
    }
    if (info.has_nyq_zero) {
        dec.first_order.push_back(FirstOrder{+1});
    }

    for (unsigned k : pair_order) {
        const real_t theta =
            2.0 * PI * static_cast<real_t>(k) / static_cast<real_t>(fir.length());
        dec.biquads.push_back(Biquad{2.0 * std::cos(theta), k});
    }

    const std::vector<quad> h_q = real_to_q128(fir.h);
    std::vector<quad> residual_q = h_q;
    std::vector<quad> next_q;

    for (const auto& fo : dec.first_order) {
        q128_first_order_half_divide(residual_q, fo.sign, next_q);
        enforce_palindromic_symmetry_q128(next_q);
        residual_q = next_q;
    }

    std::vector<quad> gamma_order_q;
    gamma_order_q.reserve(pair_order.size());
    for (unsigned k : pair_order) {
        gamma_order_q.push_back(
            2.0Q * cosq(2.0Q * static_cast<quad>(PI)
            * static_cast<quad>(k)
            / static_cast<quad>(fir.length())));
    }

    const std::vector<quad> base_after_fo = residual_q;
    static constexpr double QUALITY_THRESHOLD = 1e-6;
    const std::vector<unsigned> refresh_candidates = {3u, 5u, 8u, 12u};
    ExactPrefixTrial best_trial;

    bool have_trial = false;
    for (unsigned refresh_interval : refresh_candidates) {
        ExactPrefixTrial trial =
            run_exact_prefix_trial(base_after_fo, gamma_order_q,
                                   refresh_interval, QUALITY_THRESHOLD);
        if (!have_trial
            || trial.accepted_biquad_count > best_trial.accepted_biquad_count
            || (trial.accepted_biquad_count == best_trial.accepted_biquad_count
                && trial.rel_err < best_trial.rel_err)) {
            best_trial = std::move(trial);
            have_trial = true;
        }
    }

    const unsigned accepted_biquad_count = best_trial.accepted_biquad_count;
    residual_q = std::move(best_trial.quotient);

    std::vector<Biquad> accepted_biquads;
    accepted_biquads.reserve(accepted_biquad_count);
    for (unsigned i = 0; i < accepted_biquad_count; ++i) {
        accepted_biquads.push_back(Biquad{
            static_cast<real_t>(gamma_order_q[i]),
            pair_order[i]
        });
    }

    if (accepted_biquad_count < pair_order.size()) {
        std::cerr << "  decompose_exact_fs: accepted "
                  << accepted_biquad_count << " / " << pair_order.size()
                  << " exact biquads before residual handoff"
                  << " (best rel err " << std::scientific << best_trial.rel_err
                  << ")" << std::defaultfloat << "\n";
    }

    dec.biquads = std::move(accepted_biquads);

    const std::vector<quad> exact_product = build_exact_factor_product_q128(dec);
    if (exact_product.size() > 1) {
        auto [bulk_q, bulk_rem] = q128_poly_divide(h_q, exact_product);
        const quad max_h_q = std::max(1e-30Q, max_abs_value_q128(h_q));
        const quad max_rem_q = max_abs_value_q128(bulk_rem);
        const double max_rem = static_cast<double>(max_rem_q);
        const double rel_rem = static_cast<double>(max_rem_q / max_h_q);

        if (rel_rem > 1e-8) {
            std::cerr << "  decompose_exact_fs: bulk diagnostic remainder="
                      << std::scientific << rel_rem
                      << " (max abs " << max_rem << ")"
                      << std::defaultfloat << "\n";
        }

        if (bulk_q.size() == residual_q.size()) {
            enforce_palindromic_symmetry_q128(bulk_q);
        }
    }

    dec.remainder = q128_to_real(residual_q);
    return dec;
}

CascadeDecomposition decompose_exact_fs_full(const DirectFIR& fir,
                                             bool interleave_order)
{
    CascadeDecomposition stage1 = decompose_exact_fs(fir, interleave_order);
    const double err_stage1 = compute_rebuild_max_err(stage1, fir.h, fir.length());
    const unsigned rem_deg_stage1 =
        stage1.remainder.empty() ? 0u : static_cast<unsigned>(stage1.remainder.size() - 1);

    CascadeDecomposition structured = stage1;
    const unsigned accepted_struct = reduce_residual_u_roots(structured, 1e-6, 1e-4, 8);
    const double err_struct = compute_rebuild_max_err(structured, fir.h, fir.length());
    const unsigned rem_deg_struct =
        structured.remainder.empty() ? 0u : static_cast<unsigned>(structured.remainder.size() - 1);

    const bool structured_reduced_degree = rem_deg_struct < rem_deg_stage1;
    const bool structured_internal_ok =
        err_struct <= exact_return_err_limit(err_stage1);
    const bool structured_return_ok =
        err_struct <= exact_return_err_limit(err_stage1);

    CascadeDecomposition best = stage1;
    double best_err = err_stage1;
    unsigned best_rem_deg = rem_deg_stage1;

    auto adopt_if_better =
        [&](const CascadeDecomposition& cand,
            double cand_err,
            unsigned cand_rem_deg,
            const char* tag,
            unsigned accepted) {
            const bool improves_degree = cand_rem_deg < rem_deg_stage1;
            const bool acceptable_error =
                cand_err <= exact_return_err_limit(err_stage1);
            if (accepted == 0 || !improves_degree || !acceptable_error) {
                return false;
            }
            const bool better =
                (cand_rem_deg < best_rem_deg)
                || (cand_rem_deg == best_rem_deg && cand_err < best_err);
            if (!better) {
                return false;
            }

            std::cerr << "  exact_fs_full: adopt " << tag
                      << " (accepted=" << accepted
                      << ", rem " << rem_deg_stage1 << " -> " << cand_rem_deg
                      << ", rebuild " << std::scientific << cand_err
                      << std::defaultfloat << ")\n";

            best = cand;
            best_err = cand_err;
            best_rem_deg = cand_rem_deg;
            return true;
        };

    if (accepted_struct > 0 && structured_reduced_degree && structured_return_ok) {
        adopt_if_better(structured, err_struct, rem_deg_struct, "structured", accepted_struct);
    }

    {
        CascadeDecomposition full_from_stage1 = stage1;
        const unsigned accepted_full_stage1 =
            reduce_residual_u_roots(full_from_stage1, 1e-8, 1e-6, 0);
        const double err_full_stage1 =
            compute_rebuild_max_err(full_from_stage1, fir.h, fir.length());
        const unsigned rem_deg_full_stage1 =
            full_from_stage1.remainder.empty()
                ? 0u
                : static_cast<unsigned>(full_from_stage1.remainder.size() - 1);

        adopt_if_better(full_from_stage1,
                        err_full_stage1,
                        rem_deg_full_stage1,
                        "full-from-stage1",
                        accepted_full_stage1);
    }

    if (accepted_struct > 0 && structured_reduced_degree && structured_internal_ok) {
        CascadeDecomposition full_from_structured = structured;
        const unsigned accepted_full_structured =
            reduce_residual_u_roots(full_from_structured, 1e-8, 1e-6, 0);
        const double err_full_structured =
            compute_rebuild_max_err(full_from_structured, fir.h, fir.length());
        const unsigned rem_deg_full_structured =
            full_from_structured.remainder.empty()
                ? 0u
                : static_cast<unsigned>(full_from_structured.remainder.size() - 1);

        adopt_if_better(full_from_structured,
                        err_full_structured,
                        rem_deg_full_structured,
                        "full-from-structured",
                        accepted_full_structured);
    }

    if (best_rem_deg == rem_deg_stage1 && best_err == err_stage1) {
        std::cerr << "  exact_fs_full: fallback to stage1 "
                  << "(structured accepted=" << accepted_struct
                  << ", rem " << rem_deg_stage1 << " -> " << rem_deg_struct
                  << ", rebuild " << std::scientific << err_struct
                  << " vs stage1 " << err_stage1 << ")"
                  << std::defaultfloat << "\n";
    }

    return best;
}

CascadeDecomposition decompose_exact_fs_structured(const DirectFIR& fir,
                                                   bool interleave_order)
{
    CascadeDecomposition stage1 = decompose_exact_fs(fir, interleave_order);
    const double err_stage1 = compute_rebuild_max_err(stage1, fir.h, fir.length());
    const unsigned rem_deg_stage1 =
        stage1.remainder.empty() ? 0u : static_cast<unsigned>(stage1.remainder.size() - 1);

    CascadeDecomposition trial = stage1;
    const unsigned accepted = reduce_residual_u_roots(trial, 1e-6, 1e-4, 8);
    const double err_trial = compute_rebuild_max_err(trial, fir.h, fir.length());
    const unsigned rem_deg_trial =
        trial.remainder.empty() ? 0u : static_cast<unsigned>(trial.remainder.size() - 1);

    const bool reduced_degree = rem_deg_trial < rem_deg_stage1;
    const bool acceptable_error =
        err_trial <= exact_return_err_limit(err_stage1);

    if (accepted > 0 && reduced_degree && acceptable_error) {
        std::cerr << "  exact_fs_structured: accepted " << accepted
                  << " numerical residual sections"
                  << ", residual degree " << rem_deg_stage1
                  << " -> " << rem_deg_trial
                  << ", rebuild " << std::scientific << err_trial
                  << std::defaultfloat << "\n";
        return trial;
    }

    std::cerr << "  exact_fs_structured: fallback to stage1 "
              << "(accepted=" << accepted
              << ", rem " << rem_deg_stage1 << " -> " << rem_deg_trial
              << ", rebuild " << std::scientific << err_trial
              << " vs stage1 " << err_stage1 << ")"
              << std::defaultfloat << "\n";
    return stage1;
}
