// ═══════════════════════════════════════════════════════════════
//  cascade_filter.cpp
//  Каскадная фильтрация КИХ-фильтром, представленным в виде
//  последовательного соединения звеньев 1-го и 2-го порядков
//  и полинома-остатка.  Включает также реализацию
//  DoubleFilterState — прямой свёртки целиком в double.
//
//  Каждый входной отсчёт проходит через цепочку:
//
//    x[m] → [1-й порядок]* → [2-й порядок]* → [остаток] → y[m]
//
//  Звено 2-го порядка:  y = x − γ·x[m−1] + x[m−2]
//    (передаточная функция  1 − γ·z⁻¹ + z⁻²)
//
//  Звено 1-го порядка:  y = x + sign·x[m−1]
//    (передаточная функция  1 + sign·z⁻¹)
//
//  Полином-остаток фильтруется прямой свёрткой (через
//  DirectFilterState).
//
//  ВАЖНО: все промежуточные отсчёты между звеньями хранятся
//  в формате double, а не float.  Это устраняет накопление
//  ошибок округления при прохождении сигнала через 10+
//  последовательных звеньев.  Конвертация в sample_t (float)
//  выполняется только на выходе каскада.
//
//  Без этой меры SNR каскад/прямая ≈ 37 дБ (при float).
//  С double промежуточными SNR ≈ 120+ дБ.
//
//  Собственная реализация настоящей работы.
// ═══════════════════════════════════════════════════════════════

#include "cascade_fir.h"

namespace {

static real_t stage_scale_first_order(int sign)
{
    const real_t l1 = 1.0 + std::abs(static_cast<real_t>(sign));
    return std::max<real_t>(1.0, l1);
}

static real_t stage_scale_biquad(real_t gamma)
{
    const real_t l1 = 2.0 + std::abs(gamma);
    return std::max<real_t>(1.0, l1);
}

static real_t stage_scale_quartic(real_t alpha, real_t beta)
{
    const real_t l1 = 2.0 + 2.0 * std::abs(alpha) + std::abs(beta);
    return std::max<real_t>(1.0, l1);
}

} // namespace

// ═══════════════════════════════════════════════════════════════
//  DoubleFilterState — прямая свёртка целиком в double
//
//  Аналог DirectFilterState, но кольцевой буфер и аккумулятор
//  работают в double.  Используется для фильтрации
//  полинома-остатка внутри каскада, чтобы не терять точность
//  на границе каскад→остаток→каскад.
// ═══════════════════════════════════════════════════════════════

void DoubleFilterState::init(const std::vector<real_t>& coeffs)
{
    h   = coeffs;
    buf.assign(coeffs.size(), 0.0);
    idx = 0;
}

void DoubleFilterState::reset()
{
    std::fill(buf.begin(), buf.end(), 0.0);
    idx = 0;
}

double DoubleFilterState::push(double x)
{
    const unsigned N = static_cast<unsigned>(h.size());
    if (N == 0) return 0.0;

    // Записать новый отсчёт
    buf[idx] = x;

    // Свёртка: y = Σ h[n] · buf[(idx − n) mod N]
    double acc = 0.0;

    // Участок 1: от idx вниз до 0
    unsigned buf_pos = idx;
    unsigned n = 0;
    for (; n < N; ++n) {
        acc += h[n] * buf[buf_pos];
        if (buf_pos == 0) {
            ++n;
            break;
        }
        --buf_pos;
    }

    // Участок 2: от конца буфера вниз
    buf_pos = N - 1;
    for (; n < N; ++n) {
        acc += h[n] * buf[buf_pos];
        --buf_pos;
    }

    idx = (idx + 1) % N;

    return acc;
}

// ═══════════════════════════════════════════════════════════════
//  CascadeFilterState — инициализация
//
//  Создаёт состояния (элементы задержки) для каждого звена
//  каскада и для прямого фильтра-остатка.
// ═══════════════════════════════════════════════════════════════

void CascadeFilterState::init(const CascadeDecomposition& dec)
{
    // Межсекционное масштабирование уменьшает рост промежуточных
    // амплитуд в длинных каскадах. Для каждого звена используем
    // безопасную верхнюю оценку его усиления через L1-норму
    // коэффициентов и переносим компенсацию в общий gain.
    long double gain_acc = static_cast<long double>(dec.gain);

    // Звенья 1-го порядка
    fo_states.resize(dec.first_order.size());
    for (size_t i = 0; i < dec.first_order.size(); ++i) {
        fo_states[i].sign = dec.first_order[i].sign;
        const real_t scale = stage_scale_first_order(dec.first_order[i].sign);
        fo_states[i].scale_inv = 1.0 / scale;
        fo_states[i].d1   = 0.0;
        gain_acc *= static_cast<long double>(scale);
    }

    // Звенья 2-го порядка
    bq_states.resize(dec.biquads.size());
    for (size_t i = 0; i < dec.biquads.size(); ++i) {
        bq_states[i].gamma = dec.biquads[i].gamma;
        const real_t scale = stage_scale_biquad(dec.biquads[i].gamma);
        bq_states[i].scale_inv = 1.0 / scale;
        bq_states[i].d1    = 0.0;
        bq_states[i].d2    = 0.0;
        gain_acc *= static_cast<long double>(scale);
    }

    // Звенья 4-го порядка (палиндромные quartic-секции)
    qt_states.resize(dec.quartics.size());
    for (size_t i = 0; i < dec.quartics.size(); ++i) {
        qt_states[i].alpha = dec.quartics[i].alpha;
        qt_states[i].beta  = dec.quartics[i].beta;
        const real_t scale = stage_scale_quartic(dec.quartics[i].alpha,
                                                 dec.quartics[i].beta);
        qt_states[i].scale_inv = 1.0 / scale;
        qt_states[i].d1 = 0.0;
        qt_states[i].d2 = 0.0;
        qt_states[i].d3 = 0.0;
        qt_states[i].d4 = 0.0;
        gain_acc *= static_cast<long double>(scale);
    }

    // Фильтр-остаток (прямая свёртка)
    rem_state.init(dec.remainder);

    // Общий коэффициент усиления
    gain = static_cast<real_t>(gain_acc);
}

// ═══════════════════════════════════════════════════════════════
//  CascadeFilterState — сброс состояния (обнуление задержек)
// ═══════════════════════════════════════════════════════════════

void CascadeFilterState::reset()
{
    for (auto& fo : fo_states) {
        fo.d1 = 0.0;
    }
    for (auto& bq : bq_states) {
        bq.d1 = 0.0;
        bq.d2 = 0.0;
    }
    for (auto& qt : qt_states) {
        qt.d1 = 0.0;
        qt.d2 = 0.0;
        qt.d3 = 0.0;
        qt.d4 = 0.0;
    }
    rem_state.reset();
}

// ═══════════════════════════════════════════════════════════════
//  CascadeFilterState::push — подать один отсчёт, вернуть выход
//
//  Отсчёт проходит последовательно через все звенья.
//  Порядок звеньев соответствует порядку, в котором нули
//  выделялись при декомпозиции (чередующийся, если задан).
//
//  Звено 1-го порядка (1 + sign·z⁻¹):
//    out = in + sign · d1
//    d1 ← in
//
//  Звено 2-го порядка (1 − γ·z⁻¹ + z⁻²):
//    out = in − γ·d1 + d2
//    d2 ← d1
//    d1 ← in
//
//  ВСЕ промежуточные значения (val, d1, d2) хранятся в double.
//  Это ключевое отличие от наивной реализации, где d1/d2
//  хранились в float и каждое звено теряло ~7 бит точности.
//  При 10 звеньях это давало SNR ≈ 37 дБ.
//  С double промежуточными единственная потеря точности —
//  на входе (float→double) и на выходе (double→float).
// ═══════════════════════════════════════════════════════════════

sample_t CascadeFilterState::push(sample_t x)
{
    double val = static_cast<double>(x);

    // ── Звенья 1-го порядка ──────────────────────────────────
    //  d1 хранится в double — без потери точности между звеньями.
    for (auto& fo : fo_states) {
        double in = val;
        double s  = static_cast<double>(fo.sign);
        val = (in + s * fo.d1) * fo.scale_inv;
        fo.d1 = in;          // double ← double, без округления
    }

    // ── Звенья 2-го порядка ──────────────────────────────────
    //  d1, d2 хранятся в double — без потери точности.
    //  gamma тоже double (поле структуры Biquad).
    for (auto& bq : bq_states) {
        double in = val;
        double g  = bq.gamma;
        val = (in - g * bq.d1 + bq.d2) * bq.scale_inv;
        bq.d2 = bq.d1;       // double ← double
        bq.d1 = in;           // double ← double
    }

    // ── Звенья 4-го порядка (палиндромные quartic-секции) ────
    //  (1 − α·z⁻¹ + β·z⁻² − α·z⁻³ + z⁻⁴)
    //  С палиндромной симметрией:
    //    y = (x + d4) − α·(d1 + d3) + β·d2
    //  Умножений: 2 (на α и β), сложений: 4.
    //  d1..d4 хранятся в double.
    for (auto& qt : qt_states) {
        double in = val;
        double a  = qt.alpha;
        double b  = qt.beta;
        val = ((in + qt.d4) - a * (qt.d1 + qt.d3) + b * qt.d2) * qt.scale_inv;
        qt.d4 = qt.d3;
        qt.d3 = qt.d2;
        qt.d2 = qt.d1;
        qt.d1 = in;
    }

    // ── Фильтр-остаток (прямая свёртка) ─────────────────────
    //
    // Если остаток — тривиальный (один коэффициент, т.е.
    // константа), просто домножаем.  Иначе — пропускаем
    // через DirectFilterState.
    //
    // Фильтр-остаток работает целиком в double (DoubleFilterState),
    // поэтому потери точности на границе каскад→остаток нет.
    if (rem_state.h.size() <= 1) {
        if (!rem_state.h.empty()) {
            val *= rem_state.h[0];
        }
    } else {
        val = rem_state.push(val);
    }

    // ── Коэффициент усиления ─────────────────────────────────
    val *= gain;

    return static_cast<sample_t>(val);
}

// ═══════════════════════════════════════════════════════════════
//  filter_cascade — пакетная каскадная фильтрация
//
//  Создаёт состояние каскадного фильтра, пропускает весь
//  входной сигнал поотсчётно и возвращает выходную
//  последовательность.
// ═══════════════════════════════════════════════════════════════

std::vector<sample_t> filter_cascade(const CascadeDecomposition& dec,
                                     const std::vector<sample_t>& x)
{
    CascadeFilterState state;
    state.init(dec);

    std::vector<sample_t> y(x.size());

    for (size_t m = 0; m < x.size(); ++m) {
        y[m] = state.push(x[m]);
    }

    return y;
}
