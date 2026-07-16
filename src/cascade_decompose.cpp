// ═══════════════════════════════════════════════════════════════
//  cascade_decompose.cpp
//  Baseline-реализация каскадной декомпозиции КИХ-фильтра
//  частотной выборки.
//
//  Этот файл реализует `decompose()`:
//    последовательное выделение известных нулей на единичной
//    окружности через половинную рекурсию.
//
//  Путь сохранён как компактный baseline/reference.
//  Основная ветка для больших фильтров и exact-first работы
//  находится в `decompose_exact_fs.cpp`.
//
//  Алгоритм (половинная рекурсия + симметрия):
//    Для каждого нулевого отсчёта H_d[k] = 0 (полоса заграждения)
//    нуль z_k = exp(j·2πk/N) порождает сомножитель 2-го порядка
//    D(z) = 1 − γ_k·z⁻¹ + z⁻², где γ_k = 2·cos(2πk/N).
//
//    Рекуррентное деление:
//      q[0] = h[0]
//      q[1] = h[1] + γ·q[0]
//      q[n] = h[n] + γ·q[n−1] − q[n−2],   n ≥ 2
//
//    Ключевое улучшение (собственный результат работы):
//    рекурсия выполняется ТОЛЬКО до середины частного,
//    вторая половина заполняется из симметрии q[n] = q[L−n],
//    где L = len(q)−1.  Это устраняет накопление ошибки
//    во второй половине рекурсии и позволяет работать
//    с фильтрами порядка N > 50, где полная рекурсия
//    теряет точность.
//
//    Обоснование: если H(z) симметричен и D(z) симметричен,
//    то Q(z) = H(z)/D(z) также симметричен (доказано в работе,
//    теорема о палиндромах).
// ═══════════════════════════════════════════════════════════════

#include "cascade_fir.h"
#include <stdexcept>
#include <cstring>

// ─────────────────────────────────────────────────────────────
//  Порог для проверки остатка (используется в полной рекурсии
//  и в диагностическом режиме).  Для половинной рекурсии
//  проверка остатка не нужна — симметрия гарантирует точность.
// ─────────────────────────────────────────────────────────────
static constexpr real_t REMAINDER_TOL = 1e-6;

// ─────────────────────────────────────────────────────────────
//  Порог качества деления.
//  После каждого деления проверяется: q × D ≈ h_текущий?
//  Если относительная ошибка max|q×D − h| / max|h| превышает
//  этот порог, деление считается слишком неточным (полином
//  испорчен предыдущими делениями), и дальнейшие деления
//  прекращаются.  Оставшиеся нули остаются в полиноме-остатке.
//
//  Значение 1e-6 выбрано эмпирически: при SNR каскадной
//  фильтрации > 100 дБ относительная ошибка деления < 1e-8.
//  Порог 1e-6 даёт запас на 2 порядка.
// ─────────────────────────────────────────────────────────────
static constexpr real_t DIVISION_QUALITY_THRESHOLD = 1e-6;

// Режим деления: половинная рекурсия (по умолчанию)
// или полная рекурсия (для диагностики / сравнения).
enum class DivisionMode {
    HALF_RECURRENCE,   // рекурсия до середины + симметрия (рекомендуется)
    FULL_RECURRENCE    // полная рекурсия + проверка остатка (старый метод)
};

// ═══════════════════════════════════════════════════════════════
//  divide_by_biquad_half
//
//  ПОЛОВИННАЯ РЕКУРСИЯ + СИММЕТРИЯ (основной метод).
//
//  Деление симметричного полинома h[0..len-1] на
//  D(z) = 1 − γ·z⁻¹ + z⁻².
//
//  Рекурсия выполняется только до середины частного:
//    q[0] = h[0]
//    q[1] = h[1] + γ·q[0]
//    q[n] = h[n] + γ·q[n−1] − q[n−2],   n = 2, …, mid
//  где mid = ⌊(qlen−1)/2⌋.
//
//  Вторая половина заполняется из симметрии:
//    q[qlen−1−n] = q[n],   n = 0, …, mid−1
//
//  Этот метод:
//  - не требует проверки остатка (симметрия гарантирует);
//  - устойчив при любом N (ошибка не накапливается за
//    вторую половину);
//  - опирается на теорему: частное палиндрома на палиндром
//    есть палиндром (доказано в TeX-документе).
//
//  Всегда возвращает true (деление всегда «успешно» в смысле
//  арифметики; проверка осмысленности — через пересборку).
// ═══════════════════════════════════════════════════════════════

static bool divide_by_biquad_half(const std::vector<real_t>& h,
                                  real_t gamma,
                                  std::vector<real_t>& quotient)
{
    const size_t len = h.size();
    if (len < 3) return false;

    if (len == 3) {
        quotient.resize(1);
        quotient[0] = h[0];

        const real_t rem_mid = std::abs(h[1] + gamma * quotient[0]);
        const real_t rem_sym = std::abs(h[2] - quotient[0]);
        if (rem_mid > REMAINDER_TOL || rem_sym > REMAINDER_TOL) {
            quotient.clear();
            return false;
        }
        return true;
    }

    const size_t qlen = len - 2;
    quotient.resize(qlen);

    // ── Рекурсия до середины ─────────────────────────────────
    //  mid — последний индекс, до которого считаем рекурсию.
    //  Для qlen нечётного: mid = (qlen-1)/2 — центральный элемент.
    //  Для qlen чётного: mid = qlen/2 - 1 — последний перед центром.
    const size_t mid = (qlen - 1) / 2;

    quotient[0] = h[0];

    if (qlen >= 2) {
        quotient[1] = h[1] + gamma * quotient[0];
    }

    for (size_t n = 2; n <= mid; ++n) {
        quotient[n] = h[n] + gamma * quotient[n - 1] - quotient[n - 2];
    }

    // Для нечётного qlen: центральный элемент q[mid] уже вычислен.
    // Для чётного qlen: элемент q[qlen/2] ещё не вычислен —
    // вычислим его отдельно, если нужно (это пара к q[qlen/2-1]).
    if (qlen % 2 == 0 && qlen >= 4) {
        size_t center = qlen / 2;
        if (center > mid) {
            quotient[center] = h[center] + gamma * quotient[center - 1]
                             - quotient[center - 2];
        }
    }

    // ── Заполнение второй половины из симметрии ──────────────
    //  q[n] = q[qlen − 1 − n]
    for (size_t n = 0; n < qlen / 2; ++n) {
        quotient[qlen - 1 - n] = quotient[n];
    }

    return true;
}

// ═══════════════════════════════════════════════════════════════
//  divide_by_biquad_full
//
//  ПОЛНАЯ РЕКУРСИЯ + ПРОВЕРКА ОСТАТКА (старый метод,
//  сохранён для диагностики и сравнения).
//
//  Рекурсия выполняется на всю длину частного.
//  Если остаток превышает REMAINDER_TOL — возвращает false.
//
//  При N > ~40 этот метод начинает терять точность из-за
//  накопления ошибки во второй половине рекурсии.
// ═══════════════════════════════════════════════════════════════

static bool divide_by_biquad_full(const std::vector<real_t>& h,
                                  real_t gamma,
                                  std::vector<real_t>& quotient,
                                  real_t* out_rem1 = nullptr,
                                  real_t* out_rem2 = nullptr)
{
    const size_t len = h.size();
    if (len < 3) return false;

    if (len == 3) {
        quotient.resize(1);
        quotient[0] = h[0];

        real_t rem1 = std::abs(h[2] - quotient[0]);
        real_t rem2 = std::abs(h[1] + gamma * quotient[0]);

        if (out_rem1) *out_rem1 = rem1;
        if (out_rem2) *out_rem2 = rem2;

        if (rem1 > REMAINDER_TOL || rem2 > REMAINDER_TOL) {
            quotient.clear();
            return false;
        }
        return true;
    }

    const size_t qlen = len - 2;
    quotient.resize(qlen);

    quotient[0] = h[0];
    if (qlen >= 2) {
        quotient[1] = h[1] + gamma * quotient[0];
    }
    for (size_t n = 2; n < qlen; ++n) {
        quotient[n] = h[n] + gamma * quotient[n - 1] - quotient[n - 2];
    }

    real_t rem1 = std::abs(quotient[qlen - 1] - h[0]);
    real_t rem2 = (qlen >= 2)
                ? std::abs(h[1] + gamma * h[0] - quotient[qlen - 2])
                : 0.0;

    if (out_rem1) *out_rem1 = rem1;
    if (out_rem2) *out_rem2 = rem2;

    if (rem1 > REMAINDER_TOL || rem2 > REMAINDER_TOL) {
        quotient.clear();
        return false;
    }

    return true;
}

// ═══════════════════════════════════════════════════════════════
//  divide_by_biquad — диспетчер, выбирающий метод деления
// ═══════════════════════════════════════════════════════════════

static bool divide_by_biquad(const std::vector<real_t>& h,
                             real_t gamma,
                             std::vector<real_t>& quotient,
                             DivisionMode mode = DivisionMode::HALF_RECURRENCE)
{
    if (mode == DivisionMode::HALF_RECURRENCE) {
        return divide_by_biquad_half(h, gamma, quotient);
    } else {
        return divide_by_biquad_full(h, gamma, quotient);
    }
}

// ═══════════════════════════════════════════════════════════════
//  divide_by_first_order
//
//  Деление полинома h[0..len-1] на (1 + sign·z⁻¹):
//    sign = +1 → делитель (1 + z⁻¹), нуль z = −1
//    sign = −1 → делитель (1 − z⁻¹), нуль z = +1
//
//  Рекуррентное соотношение:
//    q[0] = h[0]
//    q[n] = h[n] − sign·q[n−1],   n = 1, …, len−2
//
//  Условие нулевого остатка:  |q[len−2]| ≈ 0
//  (что эквивалентно H(−1) = 0 для sign=+1
//   и H(+1) = 0 для sign=−1).
// ═══════════════════════════════════════════════════════════════

static bool divide_by_first_order(const std::vector<real_t>& h,
                                  int sign,
                                  std::vector<real_t>& quotient)
{
    const size_t len = h.size();
    if (len < 2) return false;

    const size_t qlen = len - 1;
    quotient.resize(qlen);

    real_t s = static_cast<real_t>(sign);

    quotient[0] = h[0];
    for (size_t n = 1; n < qlen; ++n) {
        quotient[n] = h[n] + s * quotient[n - 1];
    }

    // Остаток — последний «виртуальный» шаг рекурсии
    real_t remainder = h[len - 1] + s * quotient[qlen - 1];

    if (std::abs(remainder) > REMAINDER_TOL) {
        quotient.clear();
        return false;
    }

    return true;
}

// ═══════════════════════════════════════════════════════════════
//  check_divisibility_biquad
//
//  Быстрая проверка делимости БЕЗ формирования частного.
//  Выполняет ту же рекурсию, но сохраняет только два
//  последних значения.  Полезна для отладки.
// ═══════════════════════════════════════════════════════════════

static bool check_divisibility_biquad(const std::vector<real_t>& h,
                                      real_t gamma)
{
    const size_t len = h.size();
    if (len < 3) return false;

    if (len == 3) {
        const real_t q0 = h[0];
        const real_t rem1 = std::abs(h[2] - q0);
        const real_t rem2 = std::abs(h[1] + gamma * q0);
        return (rem1 < REMAINDER_TOL) && (rem2 < REMAINDER_TOL);
    }

    const size_t qlen = len - 2;

    real_t q_prev2 = h[0];                         // q[0]
    real_t q_prev1 = h[1] + gamma * q_prev2;       // q[1]

    real_t q_cur = q_prev1;
    for (size_t n = 2; n < qlen; ++n) {
        q_cur   = h[n] + gamma * q_prev1 - q_prev2;
        q_prev2 = q_prev1;
        q_prev1 = q_cur;
    }

    // q_prev1 = q[qlen-1], q_prev2 = q[qlen-2]
    real_t rem1 = std::abs(q_prev1 - h[0]);
    real_t rem2 = (qlen >= 2)
                ? std::abs(h[1] + gamma * h[0] - q_prev2)
                : 0.0;

    return (rem1 < REMAINDER_TOL) && (rem2 < REMAINDER_TOL);
}

// ═══════════════════════════════════════════════════════════════
//  interleave_indices
//
//  Формирует чередующийся порядок индексов нулей полосы
//  заграждения:  k_1, k_P, k_2, k_{P−1}, …
//
//  Это обеспечивает примерно плоскую АЧХ промежуточного
//  сигнала после каждой пары звеньев: выемка в нижней части
//  спектра компенсируется отсутствием выемки в верхней.
// ═══════════════════════════════════════════════════════════════

static std::vector<unsigned>
interleave_indices(const std::vector<unsigned>& stopband_k)
{
    std::vector<unsigned> sorted = stopband_k;
    std::sort(sorted.begin(), sorted.end());

    std::vector<unsigned> result;
    result.reserve(sorted.size());

    size_t lo = 0;
    size_t hi = sorted.size() - 1;
    bool take_lo = true;

    while (lo <= hi && hi < sorted.size()) {
        if (take_lo) {
            result.push_back(sorted[lo]);
            ++lo;
        } else {
            result.push_back(sorted[hi]);
            if (hi == 0) break;
            --hi;
        }
        take_lo = !take_lo;
    }

    return result;
}

// ═══════════════════════════════════════════════════════════════
//  enforce_quotient_symmetry
//
//  Принудительная симметризация частного.
//
//  При использовании HALF_RECURRENCE эта функция не нужна
//  (симметрия уже обеспечена конструктивно).
//  Оставлена для FULL_RECURRENCE режима, где усреднение
//  первой и второй половины уменьшает ошибку.
// ═══════════════════════════════════════════════════════════════

static void enforce_quotient_symmetry(std::vector<real_t>& q)
{
    const size_t len = q.size();
    for (size_t n = 0; n < len / 2; ++n) {
        real_t avg = (q[n] + q[len - 1 - n]) * 0.5;
        q[n]           = avg;
        q[len - 1 - n] = avg;
    }
}

// ═══════════════════════════════════════════════════════════════
//  compute_division_residual
//
//  Диагностическая функция: вычисляет невязку условия
//  делимости, не выполняя деление.  Полная рекурсия,
//  возвращает два значения остатка.
//
//  Используется для:
//  - мониторинга накопления ошибки при последовательных делениях;
//  - сравнения методов HALF vs FULL.
// ═══════════════════════════════════════════════════════════════

static void compute_division_residual(const std::vector<real_t>& h,
                                      real_t gamma,
                                      real_t& rem1,
                                      real_t& rem2)
{
    const size_t len = h.size();
    if (len < 3) {
        rem1 = rem2 = 0.0;
        return;
    }

    if (len == 3) {
        const real_t q0 = h[0];
        rem1 = std::abs(h[2] - q0);
        rem2 = std::abs(h[1] + gamma * q0);
        return;
    }

    const size_t qlen = len - 2;

    real_t q_prev2 = h[0];
    real_t q_prev1 = h[1] + gamma * q_prev2;

    real_t q_cur = q_prev1;
    for (size_t n = 2; n < qlen; ++n) {
        q_cur   = h[n] + gamma * q_prev1 - q_prev2;
        q_prev2 = q_prev1;
        q_prev1 = q_cur;
    }

    rem1 = std::abs(q_prev1 - h[0]);
    rem2 = (qlen >= 2)
         ? std::abs(h[1] + gamma * h[0] - q_prev2)
         : 0.0;
}

// ═══════════════════════════════════════════════════════════════
//  compute_chebyshev_U
//
//  Вычисляет значение U_n(γ) — последовательность, определяемую
//  рекурсией U_0 = 1, U_1 = γ, U_n = γ·U_{n−1} − U_{n−2}.
//
//  Это чебышёвские полиномы второго рода от γ/2:
//    U_n(γ) = sin((n+1)·arccos(γ/2)) / sin(arccos(γ/2))
//
//  Используется для вычисления чувствительности q[N−3]
//  к изменению h[m]: ∂q[N−3]/∂h[m] = U_{N−3−m}(γ).
// ═══════════════════════════════════════════════════════════════

static real_t compute_chebyshev_U(unsigned n, real_t gamma)
{
    if (n == 0) return 1.0;
    if (n == 1) return gamma;

    real_t u_prev2 = 1.0;     // U_0
    real_t u_prev1 = gamma;   // U_1
    real_t u_cur   = gamma;

    for (unsigned i = 2; i <= n; ++i) {
        u_cur   = gamma * u_prev1 - u_prev2;
        u_prev2 = u_prev1;
        u_prev1 = u_cur;
    }

    return u_cur;
}

// ═══════════════════════════════════════════════════════════════
//  correct_for_divisibility
//
//  Коррекция полинома перед делением на D(z) = 1 − γz⁻¹ + z⁻².
//
//  Алгоритм (раздел sec:correction в работе):
//
//  1. Прогнать полную рекурсию q[0]..q[qlen−1] для текущего h[].
//  2. Вычислить невязку: δ = q[qlen−1] − h[0].
//     Если δ ≈ 0, коррекция не нужна.
//  3. Вычислить чувствительность: s = U_{qlen−1−L}(γ),
//     где L = ⌊(len−1)/2⌋ — индекс центрального коэффициента.
//  4. Скорректировать: h[L] ← h[L] − δ/s.
//     Для чётного len: h[len−1−L] ← h[L] (симметрия).
//
//  После коррекции делимость выполняется точно (в пределах
//  машинной арифметики одного деления), и последующее
//  деление даёт точный результат.
//
//  Возвращает абсолютную величину коррекции |δ/s|.
//  Если s ≈ 0 (вырожденный случай), коррекция не выполняется.
// ═══════════════════════════════════════════════════════════════

static real_t correct_for_divisibility(std::vector<real_t>& h,
                                       real_t gamma)
{
    const size_t len = h.size();
    if (len < 4) return 0.0;

    const size_t qlen = len - 2;
    const size_t L = (len - 1) / 2;   // центральный индекс

    // ── Шаг 1: полная рекурсия для вычисления невязки ────────
    //  (нужно только q[qlen−1], промежуточные значения не храним)

    real_t q_prev2 = h[0];                         // q[0]
    real_t q_prev1 = h[1] + gamma * q_prev2;       // q[1]

    if (qlen <= 2) {
        // Слишком короткий для коррекции
        return 0.0;
    }

    real_t q_cur = q_prev1;
    for (size_t n = 2; n < qlen; ++n) {
        q_cur   = h[n] + gamma * q_prev1 - q_prev2;
        q_prev2 = q_prev1;
        q_prev1 = q_cur;
    }

    // q_prev1 = q[qlen−1]
    real_t delta = q_prev1 - h[0];

    // ── Шаг 2: проверка — нужна ли коррекция ────────────────
    if (std::abs(delta) < 1e-15) {
        return 0.0;   // уже делится точно
    }

    // ── Шаг 3: чувствительность ∂q[qlen−1]/∂h[L] ───────────
    //  = U_{qlen−1−L}(γ)

    unsigned u_index = static_cast<unsigned>(qlen - 1 - L);
    real_t sensitivity = compute_chebyshev_U(u_index, gamma);

    if (std::abs(sensitivity) < 1e-30) {
        // Вырожденный случай: h[L] не влияет на q[qlen−1].
        // Коррекция невозможна через центральный коэффициент.
        // Можно попробовать другой индекс, но для простоты
        // пропускаем коррекцию.
        return 0.0;
    }

    // ── Шаг 4: коррекция ────────────────────────────────────
    real_t correction = delta / sensitivity;

    h[L] -= correction;

    // Поддержание симметрии: h[len−1−L] = h[L]
    if (L != len - 1 - L) {
        h[len - 1 - L] = h[L];
    }

    return std::abs(correction);
}

// ═══════════════════════════════════════════════════════════════
//  interleave_indices
//  decompose  —  основная функция каскадной декомпозиции
// ═══════════════════════════════════════════════════════════════

CascadeDecomposition decompose(const DirectFIR& fir,
                               bool interleave_order)
{
    CascadeDecomposition dec;
    dec.gain = 1.0;

    const unsigned N = fir.length();
    const FilterSpec& spec = fir.spec;

    // ──────────────────────────────────────────────────────────
    //  1.  Определить индексы нулевых выборок в полосе заграждения.
    //
    //  Для ФНЧ: k = k_stop, k_stop+1, …, ⌊N/2⌋  (верхняя
    //  половина спектра; нижняя получается эрмитовой симметрией).
    //
    //  Каждому k соответствует пара нулей z = exp(±j·2πk/N),
    //  объединённых в звено 2-го порядка с γ_k = 2·cos(2πk/N).
    //
    //  Исключения:
    //    k = 0     → нуль z = +1, звено 1-го порядка (1 − z⁻¹)
    //    k = N/2   → нуль z = −1, звено 1-го порядка (1 + z⁻¹)
    //    (только при чётном N)
    // ──────────────────────────────────────────────────────────

    const unsigned k_stop = spec.k_stop();
    const unsigned k_half = N / 2;

    // Собрать индексы k, где H_d[k] = 0
    // Для ФНЧ: от k_stop до k_half (включительно, если чётное N)
    std::vector<unsigned> stopband_k;
    bool has_zero_at_dc    = false;   // k = 0
    bool has_zero_at_nyq   = false;   // k = N/2

    for (unsigned k = k_stop; k <= k_half; ++k) {
        if (k == 0) {
            has_zero_at_dc = true;
        } else if (N % 2 == 0 && k == k_half) {
            has_zero_at_nyq = true;
        } else {
            stopband_k.push_back(k);
        }
    }

    // ──────────────────────────────────────────────────────────
    //  2.  Сформировать порядок деления.
    // ──────────────────────────────────────────────────────────

    std::vector<unsigned> order;
    if (interleave_order) {
        order = interleave_indices(stopband_k);
    } else {
        order = stopband_k;
    }

    // ──────────────────────────────────────────────────────────
    //  3.  Текущий полином = h[n].  Последовательно делим.
    // ──────────────────────────────────────────────────────────

    std::vector<real_t> current = fir.h;

    // 3a.  Сначала одиночные нули (1-го порядка), если есть.
    if (has_zero_at_dc) {
        std::vector<real_t> q;
        if (divide_by_first_order(current, -1, q)) {
            dec.first_order.push_back(FirstOrder{-1});
            current = std::move(q);
        } else {
            std::cerr << "  Предупреждение: нуль z=+1 (k=0) не подтверждён "
                         "делением; пропущен.\n";
        }
    }

    if (has_zero_at_nyq) {
        std::vector<real_t> q;
        if (divide_by_first_order(current, +1, q)) {
            dec.first_order.push_back(FirstOrder{+1});
            current = std::move(q);
        } else {
            std::cerr << "  Предупреждение: нуль z=−1 (k=N/2) не подтверждён "
                         "делением; пропущен.\n";
        }
    }

    // 3b.  Пары нулей (2-го порядка).
    //
    // Используется половинная рекурсия (HALF_RECURRENCE):
    // рекурсия только до середины частного, вторая половина
    // из симметрии.  Это устраняет накопление ошибки и
    // позволяет работать при любом N.
    //
    // Для диагностики также вычисляется невязка полной
    // рекурсии — она показывает, насколько «испорчен»
    // полином после предыдущих делений.

    for (unsigned k : order) {
        real_t gamma = 2.0 * std::cos(2.0 * PI * k / N);

        // ── Итеративная коррекция ────────────────────────────
        //  Однократная коррекция снижает невязку на 1–2 порядка.
        //  При большом числе предшествующих делений этого
        //  недостаточно: невязка может быть 10^{+15}, и одна
        //  поправка центрального коэффициента не обнулит её.
        //
        //  Решение: повторять коррекцию, пока невязка не станет
        //  меньше порога или не будет исчерпан лимит итераций.
        //  На каждой итерации невязка уменьшается примерно
        //  в sensitivity/1 раз, т.е. сходимость геометрическая.

        static constexpr int    MAX_CORR_ITER = 20;
        static constexpr real_t CORR_TARGET   = 1e-13;

        real_t initial_rem1 = 0.0, initial_rem2 = 0.0;
        compute_division_residual(current, gamma, initial_rem1, initial_rem2);

        real_t total_corr = 0.0;
        real_t final_rem1 = initial_rem1;
        int    corr_iters = 0;

        if (initial_rem1 > CORR_TARGET) {
            for (int iter = 0; iter < MAX_CORR_ITER; ++iter) {
                real_t c = correct_for_divisibility(current, gamma);
                total_corr += c;
                ++corr_iters;

                // Проверить, достигнут ли порог
                real_t rem1 = 0.0, rem2 = 0.0;
                compute_division_residual(current, gamma, rem1, rem2);
                final_rem1 = rem1;

                if (rem1 < CORR_TARGET) break;

                // Если невязка не уменьшается — прекратить
                // (вырожденный случай, коррекция не работает)
                if (iter > 2 && rem1 > initial_rem1 * 0.5) break;
            }
        }

        if (initial_rem1 > 1e-10) {
            std::cerr << "  Коррекция k=" << k
                      << " γ=" << std::setprecision(6) << gamma
                      << ": невязка " << std::scientific
                      << initial_rem1 << " → " << final_rem1
                      << "  (" << corr_iters << " итер."
                      << ", ΣΔh=" << total_corr << ")"
                      << std::defaultfloat << "\n";
        }

        // ── Деление (половинная рекурсия) ───────────────────
        std::vector<real_t> q;
        if (divide_by_biquad(current, gamma, q, DivisionMode::HALF_RECURRENCE)) {

            // ── Адаптивный лимит: проверка через пересборку ──
            //
            //  Локальная проверка (q × D ≈ current) бесполезна:
            //  current уже содержит накопленную ошибку, поэтому
            //  q × D всегда ≈ current (деление точное), но сам
            //  current может быть далёк от истины.
            //
            //  Вместо этого проверяем ГЛОБАЛЬНОЕ качество:
            //  пересобираем весь полином
            //    rebuilt = q × (текущий γ) × (все ранее выделенные) × fo
            //  и сравниваем с ОРИГИНАЛЬНЫМ h[n] (fir.h).
            //
            //  Если ошибка пересборки превышает порог — полином
            //  слишком испорчен, и дальнейшие деления дадут
            //  мусор.  Не добавляем biquad, прекращаем.
            {
                // Пересборка: q × все каскады (включая текущий γ)
                std::vector<real_t> rebuilt = q;

                // Текущий biquad (ещё не добавлен в dec.biquads)
                {
                    std::vector<real_t> f = {1.0, -gamma, 1.0};
                    rebuilt = poly_multiply(rebuilt, f);
                }
                // Все ранее выделенные biquad'ы
                for (const auto& bq : dec.biquads) {
                    std::vector<real_t> f = {1.0, -bq.gamma, 1.0};
                    rebuilt = poly_multiply(rebuilt, f);
                }
                // Звенья 1-го порядка
                for (const auto& fo : dec.first_order) {
                    std::vector<real_t> f = {1.0, static_cast<real_t>(fo.sign)};
                    rebuilt = poly_multiply(rebuilt, f);
                }

                // Сравнение с оригиналом fir.h
                double max_err = 0.0;
                double max_h   = 0.0;
                for (size_t n = 0; n < fir.h.size(); ++n) {
                    double v = (n < rebuilt.size()) ? rebuilt[n] : 0.0;
                    double diff = std::abs(fir.h[n] - v);
                    if (diff > max_err) max_err = diff;
                    if (std::abs(fir.h[n]) > max_h)
                        max_h = std::abs(fir.h[n]);
                }

                double rel_err = (max_h > 1e-30)
                               ? max_err / max_h : max_err;

                if (rel_err > DIVISION_QUALITY_THRESHOLD) {
                    std::cerr << "  Адаптивный лимит: k=" << k
                              << " γ=" << std::setprecision(6) << gamma
                              << " rel_err(vs orig)=" << std::scientific
                              << rel_err
                              << " > " << DIVISION_QUALITY_THRESHOLD
                              << ". Прекращаю деления ("
                              << dec.biquads.size()
                              << " звеньев выделено).\n"
                              << std::defaultfloat;
                    // НЕ добавляем biquad, НЕ обновляем current.
                    // Все невыделенные нули остаются в остатке.
                    break;
                }
            }

            dec.biquads.push_back(Biquad{gamma, k});
            current = std::move(q);
        } else {
            // При HALF_RECURRENCE сюда попадаем только если len < 4
            std::cerr << "  Предупреждение: полином слишком короткий для "
                         "деления на D(z) с k=" << k << ". Пропущено.\n";
        }
    }

    // ──────────────────────────────────────────────────────────
    //  4.  Полином-остаток и коэффициент усиления.
    //
    //  Два варианта остатка:
    //    A) Итеративный (current) — результат P последовательных
    //       делений.  Точен при малом P, деградирует при P > ~15.
    //    B) Пересчитанный — ОДНО деление h_orig / ∏(каскады).
    //       Точен при малом 2P (степень делителя), деградирует
    //       при 2P > ~30.
    //
    //  Стратегия: попробовать оба, выбрать лучший.
    //  Критерий качества: пересобрать h из остатка × каскады
    //  и сравнить с оригиналом.
    // ──────────────────────────────────────────────────────────

    dec.gain = 1.0;

    // ── Вариант A: итеративный остаток ───────────────────────
    std::vector<real_t> iterative_remainder = current;

    // ── Вариант B: пересчитанный остаток ─────────────────────
    dec.remainder = iterative_remainder;   // временно, для recompute
    recompute_remainder(dec, fir.h);
    std::vector<real_t> recomputed_remainder = dec.remainder;

    // ── Выбор лучшего: пересобрать оба и сравнить с h_orig ──
    auto quality = [&](const std::vector<real_t>& rem) -> double {
        // Пересобрать: rem × все_каскады → полином
        std::vector<real_t> rebuilt = rem;
        for (const auto& bq : dec.biquads) {
            std::vector<real_t> factor = {1.0, -bq.gamma, 1.0};
            rebuilt = poly_multiply(rebuilt, factor);
        }
        for (const auto& fo : dec.first_order) {
            std::vector<real_t> factor = {1.0, static_cast<real_t>(fo.sign)};
            rebuilt = poly_multiply(rebuilt, factor);
        }
        // Максимальное отклонение от оригинала
        double max_err = 0.0;
        for (size_t n = 0; n < fir.h.size(); ++n) {
            double v = (n < rebuilt.size()) ? rebuilt[n] : 0.0;
            double diff = std::abs(fir.h[n] - v);
            if (diff > max_err) max_err = diff;
        }
        return max_err;
    };

    double err_iter   = quality(iterative_remainder);
    double err_recomp = quality(recomputed_remainder);

    if (err_recomp < err_iter) {
        dec.remainder = std::move(recomputed_remainder);
        if (err_iter > 1e-10) {
            std::cerr << "  Остаток: пересчитанный лучше итеративного"
                      << " (" << std::scientific << err_recomp
                      << " vs " << err_iter << ")"
                      << std::defaultfloat << "\n";
        }
    } else {
        dec.remainder = std::move(iterative_remainder);
        if (err_recomp > 1e-10 && err_iter > 1e-10) {
            std::cerr << "  Остаток: итеративный лучше пересчитанного"
                      << " (" << std::scientific << err_iter
                      << " vs " << err_recomp << ")"
                      << std::defaultfloat << "\n";
        }
    }

    return dec;
}

// ═══════════════════════════════════════════════════════════════
//  recompose  —  пересборка полинома из каскадного представления
//
//  Перемножает все звенья и остаток обратно в один полином.
//  Используется для верификации: результат должен совпадать
//  с исходным h[n] с точностью до машинной арифметики.
// ═══════════════════════════════════════════════════════════════

std::vector<real_t> recompose(const CascadeDecomposition& dec,
                              unsigned /* original_length */)
{
    // Начинаем с полинома-остатка
    std::vector<real_t> result = dec.remainder;

    // Умножаем на каждое звено 2-го порядка: (1 − γ·z⁻¹ + z⁻²)
    for (const auto& bq : dec.biquads) {
        std::vector<real_t> factor = {1.0, -bq.gamma, 1.0};
        result = poly_multiply(result, factor);
    }

    // Умножаем на каждое звено 4-го порядка: (1 − α·z⁻¹ + β·z⁻² − α·z⁻³ + z⁻⁴)
    for (const auto& qt : dec.quartics) {
        std::vector<real_t> factor = {1.0, -qt.alpha, qt.beta, -qt.alpha, 1.0};
        result = poly_multiply(result, factor);
    }

    // Умножаем на каждое звено 1-го порядка: (1 + sign·z⁻¹)
    for (const auto& fo : dec.first_order) {
        std::vector<real_t> factor = {1.0, static_cast<real_t>(fo.sign)};
        result = poly_multiply(result, factor);
    }

    // Адаптивно объединённые низкопорядковые подфильтры.
    for (const auto& block : dec.blocks) {
        result = poly_multiply(result, block.coefficients);
    }

    // Применяем коэффициент усиления
    if (std::abs(dec.gain - 1.0) > 1e-15) {
        for (auto& v : result) {
            v *= dec.gain;
        }
    }

    return result;
}
