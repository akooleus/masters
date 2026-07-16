// ═══════════════════════════════════════════════════════════════
//  freq_sampling_design.cpp
//  Проектирование КИХ-фильтра методом частотной выборки.
//
//  Входные данные — спецификация FilterSpec (длина N, частота
//  дискретизации, границы полос).  Выходные данные — вектор
//  коэффициентов h[n], n = 0, …, N−1.
//
//  Алгоритм:
//    1. Сформировать N частотных выборок H_d[k]:
//       |H_d[k]| = 1   в полосе пропускания,
//       |H_d[k]| = T_m  в полосе перехода (если задано),
//       |H_d[k]| = 0   в полосе заграждения.
//       arg H_d[k] = −π·k·(N−1)/N   (линейная ФЧХ).
//    2. Вычислить h[n] через обратное ДПФ (прямое суммирование
//       или через формулу с косинусами для вещественного h[n]).
//    3. Принудить симметрию h[n] = h[N−1−n] для компенсации
//       ошибок округления.
//
//  Литература:
//    Айфичер Э., Джервис Б. Цифровая обработка сигналов. Гл. 7.7,
//    формулы (7.16)–(7.21).
// ═══════════════════════════════════════════════════════════════

#include "cascade_fir.h"
#include <stdexcept>

// -----------------------------------------------------------------
//  Вспомогательная функция: вычислить h[n] через формулу (7.21)
//  Айфичера (только вещественные операции).
//
//  Для фильтра типа I (симметрия, N нечётное) или типа II
//  (симметрия, N чётное) с линейной ФЧХ:
//
//    h[n] = (1/N) * [ H_d[0] +
//            2 * Σ_{k=1}^{M} |H_d[k]| · cos(2πk(n − α)/N) ]
//
//  где α = (N−1)/2,  M = (N−1)/2 при N нечётном,
//                     M = N/2 − 1 при N чётном.
//
//  Для N чётного к сумме добавляется член
//    H_d[N/2] · cos(π(n − α))   (если N/2-я выборка ненулевая).
// -----------------------------------------------------------------
static std::vector<real_t>
compute_h_cosine(unsigned N,
                 const std::vector<real_t>& Hd_mag,   // |H_d[k]|, k = 0..N-1
                 real_t Hd0,                           // H_d[0] (вещественное)
                 real_t HdNhalf)                       // H_d[N/2] (вещ., только для чётного N)
{
    const real_t alpha = static_cast<real_t>(N - 1) / 2.0;
    const unsigned M = (N % 2 == 1) ? (N - 1) / 2 : N / 2 - 1;

    std::vector<real_t> h(N);

    for (unsigned n = 0; n < N; ++n) {
        real_t val = Hd0;

        for (unsigned k = 1; k <= M; ++k) {
            real_t angle = 2.0 * PI * k * (static_cast<real_t>(n) - alpha) / N;
            val += 2.0 * Hd_mag[k] * std::cos(angle);
        }

        // Дополнительный член для чётного N
        if (N % 2 == 0) {
            real_t angle = PI * (static_cast<real_t>(n) - alpha);
            val += HdNhalf * std::cos(angle);
        }

        h[n] = val / static_cast<real_t>(N);
    }

    return h;
}

// -----------------------------------------------------------------
//  Принудительная симметризация: h[n] := (h[n] + h[N−1−n]) / 2
//  Устраняет машинную асимметрию порядка ε_mach.
// -----------------------------------------------------------------
static void enforce_symmetry(std::vector<real_t>& h)
{
    const size_t N = h.size();
    for (size_t n = 0; n < N / 2; ++n) {
        real_t avg = (h[n] + h[N - 1 - n]) * 0.5;
        h[n]         = avg;
        h[N - 1 - n] = avg;
    }
}

// ═══════════════════════════════════════════════════════════════
//  design_freq_sampling  —  основная функция проектирования
// ═══════════════════════════════════════════════════════════════
DirectFIR design_freq_sampling(const FilterSpec& spec,
                               const std::vector<real_t>& transition_values)
{
    const unsigned N = spec.N;
    if (N < 3) {
        throw std::invalid_argument("design_freq_sampling: N must be >= 3");
    }
    if (!std::isfinite(spec.fs) || spec.fs <= 0.0
        || !std::isfinite(spec.f_pass) || spec.f_pass < 0.0
        || !std::isfinite(spec.f_stop) || spec.f_stop <= spec.f_pass
        || spec.f_stop > spec.fs / 2.0) {
        throw std::invalid_argument(
            "design_freq_sampling: require 0 <= f_pass < f_stop <= fs/2");
    }

    // ──────────────────────────────────────────────────────────
    //  Шаг 1.  Определить амплитуды частотных выборок |H_d[k]|
    //
    //  k = 0, 1, …, N−1  соответствует частоте f_k = k·fs/N.
    //  Для k и N−k выборки эрмитово-сопряжены, поэтому
    //  достаточно задать k = 0 .. ⌊N/2⌋.
    // ──────────────────────────────────────────────────────────

    const unsigned k_pass = spec.k_pass();   // последний индекс полосы пропускания
    const unsigned k_stop = spec.k_stop();   // первый индекс полосы заграждения
    if (k_pass >= k_stop || k_stop > N / 2) {
        throw std::invalid_argument(
            "design_freq_sampling: pass/stop edges do not leave a valid sampled stopband");
    }

    // Амплитуды (половина спектра, k = 0 .. N-1)
    std::vector<real_t> Hd_mag(N, 0.0);

    // Полоса пропускания: |H_d[k]| = 1
    for (unsigned k = 0; k <= k_pass; ++k) {
        Hd_mag[k] = 1.0;
    }

    // Полоса перехода: значения из transition_values (если заданы)
    unsigned n_trans = k_stop - k_pass - 1;   // число выборок в переходной полосе
    if (!transition_values.empty()) {
        if (transition_values.size() != n_trans) {
            throw std::invalid_argument(
                "design_freq_sampling: transition_values size mismatch, need "
                + std::to_string(n_trans) + " values");
        }
        for (unsigned i = 0; i < n_trans; ++i) {
            if (!std::isfinite(transition_values[i])) {
                throw std::invalid_argument(
                    "design_freq_sampling: transition values must be finite");
            }
            Hd_mag[k_pass + 1 + i] = transition_values[i];
        }
    }
    // Если transition_values пуст — переходные выборки остаются 0
    // (резкий переход).

    // Полоса заграждения: |H_d[k]| = 0  (уже установлено).

    // Эрмитова симметрия: |H_d[N−k]| = |H_d[k]|
    for (unsigned k = 1; k < N / 2 + (N % 2 == 0 ? 0 : 1); ++k) {
        if (N - k < N) {    // защита от переполнения индекса
            Hd_mag[N - k] = Hd_mag[k];
        }
    }

    // ──────────────────────────────────────────────────────────
    //  Шаг 2.  H_d[0] и H_d[N/2] — вещественные выборки.
    //
    //  H_d[0] — коэффициент постоянной составляющей.
    //  Для ФНЧ типа I/II:  H_d[0] = 1.
    //  H_d[N/2] (при чётном N) определяется по амплитуде.
    // ──────────────────────────────────────────────────────────

    real_t Hd0     = Hd_mag[0];
    real_t HdNhalf = (N % 2 == 0) ? Hd_mag[N / 2] : 0.0;

    // ──────────────────────────────────────────────────────────
    //  Шаг 3.  Вычислить h[n] через формулу с косинусами.
    // ──────────────────────────────────────────────────────────

    std::vector<real_t> h = compute_h_cosine(N, Hd_mag, Hd0, HdNhalf);

    // ──────────────────────────────────────────────────────────
    //  Шаг 4.  Принудить симметрию.
    // ──────────────────────────────────────────────────────────

    enforce_symmetry(h);

    // ──────────────────────────────────────────────────────────
    //  Собрать результат.
    // ──────────────────────────────────────────────────────────

    DirectFIR result;
    result.h                 = std::move(h);
    result.spec              = spec;
    result.frequency_samples = std::move(Hd_mag);
    return result;
}

// ═══════════════════════════════════════════════════════════════
//  Вспомогательные функции (объявлены в cascade_fir.h)
// ═══════════════════════════════════════════════════════════════

// Полиномиальное умножение (свёртка)
std::vector<real_t> poly_multiply(const std::vector<real_t>& a,
                                  const std::vector<real_t>& b)
{
    if (a.empty() || b.empty()) return {};

    const size_t na = a.size();
    const size_t nb = b.size();
    std::vector<real_t> c(na + nb - 1, 0.0);

    for (size_t i = 0; i < na; ++i) {
        for (size_t j = 0; j < nb; ++j) {
            c[i + j] += a[i] * b[j];
        }
    }
    return c;
}

// АЧХ: |H(e^{jΘ})| на сетке из nfft точек
std::vector<real_t> compute_magnitude_response(const std::vector<real_t>& h,
                                               unsigned nfft)
{
    std::vector<real_t> mag(nfft);
    const unsigned N = static_cast<unsigned>(h.size());

    for (unsigned i = 0; i < nfft; ++i) {
        real_t theta = PI * static_cast<real_t>(i) / static_cast<real_t>(nfft);
        real_t re = 0.0, im = 0.0;
        for (unsigned n = 0; n < N; ++n) {
            re += h[n] * std::cos(theta * n);
            im -= h[n] * std::sin(theta * n);
        }
        mag[i] = std::sqrt(re * re + im * im);
    }
    return mag;
}

// Метрики сравнения коэффициентов
CoeffMetrics compare_coeffs(const std::vector<real_t>& h1,
                            const std::vector<real_t>& h2)
{
    CoeffMetrics m{};
    m.length = std::max(h1.size(), h2.size());
    m.max_abs_err = 0.0;

    real_t norm_sq = 0.0;
    real_t err_sq  = 0.0;

    for (size_t n = 0; n < m.length; ++n) {
        real_t v1 = (n < h1.size()) ? h1[n] : 0.0;
        real_t v2 = (n < h2.size()) ? h2[n] : 0.0;
        real_t diff = std::abs(v1 - v2);
        if (diff > m.max_abs_err) m.max_abs_err = diff;
        err_sq  += diff * diff;
        norm_sq += v1 * v1;
    }

    m.rel_l2_err = (norm_sq > 0.0) ? std::sqrt(err_sq / norm_sq) : 0.0;
    return m;
}

void CoeffMetrics::print(const std::string& label) const
{
    std::cout << "  [" << label << "]  length=" << length
              << "  max|Δh|=" << std::scientific << std::setprecision(3)
              << max_abs_err
              << "  rel_L2=" << rel_l2_err
              << std::defaultfloat << "\n";
}

// Метрики сравнения сигналов
CompareMetrics compare_signals(const std::vector<sample_t>& y1,
                               const std::vector<sample_t>& y2)
{
    CompareMetrics m{};
    m.num_samples = std::min(y1.size(), y2.size());
    m.max_abs_err = 0.0;
    m.max_rel_err = 0.0;

    double sum_sq1 = 0.0, sum_err_sq = 0.0;

    for (size_t n = 0; n < m.num_samples; ++n) {
        double v1 = static_cast<double>(y1[n]);
        double v2 = static_cast<double>(y2[n]);
        double diff = std::abs(v1 - v2);
        double maxv = std::max(std::abs(v1), std::abs(v2));

        if (diff > m.max_abs_err) m.max_abs_err = diff;
        if (maxv > 1e-30) {
            double rel = diff / maxv;
            if (rel > m.max_rel_err) m.max_rel_err = rel;
        }

        sum_sq1    += v1 * v1;
        sum_err_sq += diff * diff;
    }

    m.rms_err = std::sqrt(sum_err_sq / static_cast<double>(m.num_samples));
    m.snr_db  = (sum_err_sq > 0.0)
              ? 10.0 * std::log10(sum_sq1 / sum_err_sq)
              : 300.0;   // условная бесконечность
    return m;
}

void CompareMetrics::print(const std::string& label) const
{
    std::cout << "  [" << label << "]  N=" << num_samples
              << "  max|Δ|=" << std::scientific << std::setprecision(3)
              << max_abs_err
              << "  RMS=" << rms_err
              << "  SNR=" << std::fixed << std::setprecision(1)
              << snr_db << " dB"
              << std::defaultfloat << "\n";
}

// Печать информации о декомпозиции
void print_decomposition(const CascadeDecomposition& dec,
                         const FilterSpec& spec)
{
    std::cout << "\n=== Каскадная декомпозиция ===\n"
              << "  Звеньев 1-го порядка: " << dec.first_order.size() << "\n"
              << "  Звеньев 2-го порядка: " << dec.biquads.size() << "\n"
              << "  Нулей выделено:       " << dec.zeros_extracted() << "\n"
              << "  Порядок остатка:      " << (dec.remainder.size() - 1) << "\n"
              << "  Коэффициент усиления: " << std::setprecision(6) << dec.gain << "\n";

    if (!dec.biquads.empty()) {
        std::cout << "  Звенья 2-го порядка (γ, k, f/fs):\n";
        for (const auto& bq : dec.biquads) {
            real_t f_norm = static_cast<real_t>(bq.k_index) / spec.N;
            std::cout << "    γ=" << std::setw(10) << std::setprecision(6) << bq.gamma
                      << "  k=" << std::setw(3) << bq.k_index
                      << "  f/fs=" << std::setprecision(4) << f_norm
                      << "\n";
        }
    }
    std::cout << std::defaultfloat;
}

// Замер времени
// ═══════════════════════════════════════════════════════════════
//  Полиномиальное деление: a(z) / b(z) → (quotient, remainder)
//
//  Стандартный алгоритм «в столбик».
//  a — делимое (длина na), b — делитель (длина nb), na >= nb.
//  quotient имеет длину na - nb + 1,
//  remainder имеет длину nb - 1 (или меньше).
//
//  Предусловие: b[0] != 0.
//  Для наших задач b[0] = 1 (все сомножители начинаются с 1),
//  поэтому деление на b[0] тривиально.
// ═══════════════════════════════════════════════════════════════

std::pair<std::vector<real_t>, std::vector<real_t>>
poly_divide(const std::vector<real_t>& a,
            const std::vector<real_t>& b)
{
    const size_t na = a.size();
    const size_t nb = b.size();

    if (nb == 0 || na < nb) {
        // Делитель пуст или делимое короче делителя
        return {{}, a};
    }

    // Рабочая копия делимого (будет модифицироваться)
    std::vector<real_t> r = a;

    const size_t qlen = na - nb + 1;
    std::vector<real_t> q(qlen);

    for (size_t i = 0; i < qlen; ++i) {
        // Коэффициент частного: r[i] / b[0]
        q[i] = r[i] / b[0];

        // Вычитаем q[i] * b из r, начиная с позиции i
        for (size_t j = 0; j < nb; ++j) {
            r[i + j] -= q[i] * b[j];
        }
        // После этого r[i] ≈ 0 (в пределах машинной точности)
    }

    // Остаток — последние nb-1 элементов r
    std::vector<real_t> remainder(r.begin() + static_cast<long>(qlen),
                                  r.end());

    return {std::move(q), std::move(remainder)};
}

// ═══════════════════════════════════════════════════════════════
//  recompute_remainder
//
//  Пересчёт полинома-остатка из оригинальных коэффициентов h[n].
//
//  Проблема: при последовательном делении на P сомножителей
//  каждое частное содержит ошибку от предыдущего деления.
//  После P делений остаток может быть сильно искажён.
//
//  Решение: после определения списка γ_k (на этапе decompose)
//  пересчитать остаток ОДНИМ делением:
//
//    1. Построить произведение всех каскадных звеньев C(z)
//       из ТОЧНЫХ значений γ_k = 2·cos(2πk/N).
//       Поскольку γ_k вычислены из спецификации (а не из
//       промежуточного полинома), C(z) точен.
//
//    2. Разделить ОРИГИНАЛЬНЫЙ h[n] на C(z) стандартным
//       полиномиальным делением «в столбик».
//
//    3. Полученное частное — новый (точный) остаток.
//       Математический остаток деления должен быть ≈ 0
//       (все нули C(z) являются нулями h(z)).
//
//  Ошибка одного деления полинома степени N-1 на полином
//  степени 2P составляет O(N · ε_mach), что на порядки
//  лучше, чем O(P · N · ε_mach) при P последовательных
//  делениях.
// ═══════════════════════════════════════════════════════════════

void recompute_remainder(CascadeDecomposition& dec,
                         const std::vector<real_t>& h_original)
{
    // ── Шаг 1: построить произведение каскадных звеньев ──────
    std::vector<real_t> cascade_product = {1.0};

    // Звенья 1-го порядка: (1 + sign·z⁻¹)
    for (const auto& fo : dec.first_order) {
        std::vector<real_t> factor = {1.0, static_cast<real_t>(fo.sign)};
        cascade_product = poly_multiply(cascade_product, factor);
    }

    // Звенья 2-го порядка: (1 − γ·z⁻¹ + z⁻²)
    for (const auto& bq : dec.biquads) {
        std::vector<real_t> factor = {1.0, -bq.gamma, 1.0};
        cascade_product = poly_multiply(cascade_product, factor);
    }

    // ── Шаг 2: разделить оригинальный h[n] на cascade_product ─
    auto division = poly_divide(h_original, cascade_product);
    std::vector<real_t> quotient = std::move(division.first);

    // ── Шаг 3: принудить симметрию нового остатка ────────────
    //  Частное = h / C тоже симметрично (палиндром / палиндром).
    for (size_t n = 0; n < quotient.size() / 2; ++n) {
        real_t avg = (quotient[n] + quotient[quotient.size() - 1 - n]) * 0.5;
        quotient[n] = avg;
        quotient[quotient.size() - 1 - n] = avg;
    }

    // ── Шаг 4: заменить остаток в декомпозиции ──────────────
    dec.remainder = std::move(quotient);
}
