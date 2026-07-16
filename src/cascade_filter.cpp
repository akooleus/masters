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
//  Обычный путь использует double/long double. Для декомпозиций с
//  явно указанным runtime_decimal_digits=50 создаётся отдельное
//  Boost.Multiprecision-состояние; выбор точности не скрывается от API.
//
//  Собственная реализация настоящей работы.
// ═══════════════════════════════════════════════════════════════

#include "cascade_fir.h"

#include <boost/multiprecision/cpp_bin_float.hpp>

#include <stdexcept>

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

using runtime_mp = boost::multiprecision::number<
    boost::multiprecision::cpp_bin_float<50>>;

class MultiprecisionCascadeState final : public CascadeExtendedState {
public:
    explicit MultiprecisionCascadeState(const CascadeDecomposition& dec)
        : execution_order_(dec.execution_order), gain_(dec.runtime_gain)
    {
        blocks_.resize(dec.blocks.size());
        for (size_t i = 0; i < dec.blocks.size(); ++i) {
            for (double coefficient : dec.blocks[i].coefficients) {
                blocks_[i].coefficients.emplace_back(coefficient);
            }
            const size_t order = dec.blocks[i].coefficients.empty()
                ? 0 : dec.blocks[i].coefficients.size() - 1;
            blocks_[i].delay.assign(order, runtime_mp(0));
        }

        for (double coefficient : dec.remainder) {
            remainder_coefficients_.emplace_back(coefficient);
        }
        const size_t remainder_order = dec.remainder.empty()
            ? 0 : dec.remainder.size() - 1;
        remainder_delay_.assign(remainder_order, runtime_mp(0));

        for (const CascadeSectionPlacement& placement : execution_order_) {
            if (placement.kind != CascadeSectionKind::Block
                || placement.index >= blocks_.size()) {
                throw std::invalid_argument(
                    "multiprecision runtime requires a block-only execution order");
            }
        }
    }

    double push(double x) override
    {
        runtime_mp value(x);
        update_peak(value);
        for (const CascadeSectionPlacement& placement : execution_order_) {
            Block& block = blocks_[placement.index];
            runtime_mp output = block.coefficients.empty()
                ? value : block.coefficients[0] * value;
            for (size_t i = 1; i < block.coefficients.size(); ++i) {
                output += block.coefficients[i] * block.delay[i - 1];
            }
            for (size_t i = block.delay.size(); i-- > 1;) {
                block.delay[i] = block.delay[i - 1];
            }
            if (!block.delay.empty()) {
                block.delay[0] = value;
            }
            value = output * runtime_mp(placement.scale);
            update_peak(value);
        }

        if (remainder_coefficients_.size() == 1) {
            value *= remainder_coefficients_[0];
        } else if (!remainder_coefficients_.empty()) {
            runtime_mp output = remainder_coefficients_[0] * value;
            for (size_t i = 1; i < remainder_coefficients_.size(); ++i) {
                output += remainder_coefficients_[i] * remainder_delay_[i - 1];
            }
            for (size_t i = remainder_delay_.size(); i-- > 1;) {
                remainder_delay_[i] = remainder_delay_[i - 1];
            }
            if (!remainder_delay_.empty()) {
                remainder_delay_[0] = value;
            }
            value = output;
        }

        value *= gain_;
        update_peak(value);
        return static_cast<double>(value);
    }

    void reset() override
    {
        for (Block& block : blocks_) {
            std::fill(block.delay.begin(), block.delay.end(), runtime_mp(0));
        }
        std::fill(
            remainder_delay_.begin(), remainder_delay_.end(), runtime_mp(0));
        peak_ = 0.0L;
    }

    long double peak() const override
    {
        return peak_;
    }

private:
    struct Block {
        std::vector<runtime_mp> coefficients;
        std::vector<runtime_mp> delay;
    };

    void update_peak(const runtime_mp& value)
    {
        const long double magnitude =
            static_cast<long double>(abs(value));
        peak_ = std::max(peak_, magnitude);
    }

    std::vector<Block> blocks_;
    std::vector<CascadeSectionPlacement> execution_order_;
    std::vector<runtime_mp> remainder_coefficients_;
    std::vector<runtime_mp> remainder_delay_;
    runtime_mp gain_;
    long double peak_ = 0.0L;
};

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
    execution_order = dec.execution_order;
    peak_internal = 0.0L;
    extended_state.reset();
    fo_states.clear();
    bq_states.clear();
    qt_states.clear();
    block_states.clear();
    if (dec.diagnostics.runtime_decimal_digits != 0) {
        if (dec.diagnostics.runtime_decimal_digits != 50) {
            throw std::invalid_argument(
                "unsupported cascade runtime precision");
        }
        extended_state =
            std::make_unique<MultiprecisionCascadeState>(dec);
        gain = 1.0;
        return;
    }
    const bool has_explicit_order = !execution_order.empty();
    long double gain_acc = static_cast<long double>(dec.gain);

    // Звенья 1-го порядка
    fo_states.resize(dec.first_order.size());
    for (size_t i = 0; i < dec.first_order.size(); ++i) {
        fo_states[i].sign = dec.first_order[i].sign;
        const real_t scale = has_explicit_order
            ? 1.0 : stage_scale_first_order(dec.first_order[i].sign);
        fo_states[i].scale_inv = 1.0 / scale;
        fo_states[i].d1   = 0.0;
        if (!has_explicit_order) {
            gain_acc *= static_cast<long double>(scale);
        }
    }

    // Звенья 2-го порядка
    bq_states.resize(dec.biquads.size());
    for (size_t i = 0; i < dec.biquads.size(); ++i) {
        bq_states[i].gamma = dec.biquads[i].gamma;
        const real_t scale = has_explicit_order
            ? 1.0 : stage_scale_biquad(dec.biquads[i].gamma);
        bq_states[i].scale_inv = 1.0 / scale;
        bq_states[i].d1    = 0.0;
        bq_states[i].d2    = 0.0;
        if (!has_explicit_order) {
            gain_acc *= static_cast<long double>(scale);
        }
    }

    // Звенья 4-го порядка (палиндромные quartic-секции)
    qt_states.resize(dec.quartics.size());
    for (size_t i = 0; i < dec.quartics.size(); ++i) {
        qt_states[i].alpha = dec.quartics[i].alpha;
        qt_states[i].beta  = dec.quartics[i].beta;
        const real_t scale = has_explicit_order
            ? 1.0 : stage_scale_quartic(dec.quartics[i].alpha,
                                        dec.quartics[i].beta);
        qt_states[i].scale_inv = 1.0 / scale;
        qt_states[i].d1 = 0.0;
        qt_states[i].d2 = 0.0;
        qt_states[i].d3 = 0.0;
        qt_states[i].d4 = 0.0;
        if (!has_explicit_order) {
            gain_acc *= static_cast<long double>(scale);
        }
    }

    // Фильтр-остаток (прямая свёртка)
    block_states.resize(dec.blocks.size());
    for (size_t i = 0; i < dec.blocks.size(); ++i) {
        block_states[i].coefficients = dec.blocks[i].coefficients;
        const size_t order = dec.blocks[i].coefficients.empty()
            ? 0 : dec.blocks[i].coefficients.size() - 1;
        block_states[i].delay.assign(order, 0.0);
    }

    // Фильтр-остаток (прямая свёртка)
    rem_state.init(dec.remainder);

    // Общий коэффициент усиления
    gain = has_explicit_order
        ? dec.runtime_gain
        : static_cast<real_t>(gain_acc);
}

// ═══════════════════════════════════════════════════════════════
//  CascadeFilterState — сброс состояния (обнуление задержек)
// ═══════════════════════════════════════════════════════════════

void CascadeFilterState::reset()
{
    peak_internal = 0.0L;
    if (extended_state) {
        extended_state->reset();
        return;
    }
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
    for (auto& block : block_states) {
        std::fill(block.delay.begin(), block.delay.end(), 0.0);
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

double CascadeFilterState::push_double(double x)
{
    if (extended_state) {
        const double output = extended_state->push(x);
        peak_internal = extended_state->peak();
        return output;
    }
    long double val = static_cast<long double>(x);
    peak_internal = std::max(peak_internal, std::abs(val));

    if (!execution_order.empty()) {
        for (const auto& placement : execution_order) {
            switch (placement.kind) {
            case CascadeSectionKind::FirstOrder: {
                auto& fo = fo_states[placement.index];
                const double in = static_cast<double>(val);
                val = (in + static_cast<double>(fo.sign) * fo.d1)
                    * placement.scale;
                fo.d1 = in;
                break;
            }
            case CascadeSectionKind::Biquad: {
                auto& bq = bq_states[placement.index];
                const double in = static_cast<double>(val);
                val = (in - bq.gamma * bq.d1 + bq.d2)
                    * placement.scale;
                bq.d2 = bq.d1;
                bq.d1 = in;
                break;
            }
            case CascadeSectionKind::Quartic: {
                auto& qt = qt_states[placement.index];
                const double in = static_cast<double>(val);
                val = ((in + qt.d4) - qt.alpha * (qt.d1 + qt.d3)
                    + qt.beta * qt.d2) * placement.scale;
                qt.d4 = qt.d3;
                qt.d3 = qt.d2;
                qt.d2 = qt.d1;
                qt.d1 = in;
                break;
            }
            case CascadeSectionKind::Block: {
                auto& block = block_states[placement.index];
                long double out = block.coefficients.empty()
                    ? val : static_cast<long double>(block.coefficients[0]) * val;
                for (size_t i = 1; i < block.coefficients.size(); ++i) {
                    out += static_cast<long double>(block.coefficients[i])
                         * block.delay[i - 1];
                }
                for (size_t i = block.delay.size(); i-- > 1;) {
                    block.delay[i] = block.delay[i - 1];
                }
                if (!block.delay.empty()) {
                    block.delay[0] = val;
                }
                val = out * static_cast<long double>(placement.scale);
                break;
            }
            }
            peak_internal = std::max(peak_internal, std::abs(val));
        }
    } else {

    // ── Звенья 1-го порядка ──────────────────────────────────
    //  d1 хранится в double — без потери точности между звеньями.
    for (auto& fo : fo_states) {
        double in = static_cast<double>(val);
        double s  = static_cast<double>(fo.sign);
        val = (in + s * fo.d1) * fo.scale_inv;
        fo.d1 = in;          // double ← double, без округления
    }

    // ── Звенья 2-го порядка ──────────────────────────────────
    //  d1, d2 хранятся в double — без потери точности.
    //  gamma тоже double (поле структуры Biquad).
    for (auto& bq : bq_states) {
        double in = static_cast<double>(val);
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
        double in = static_cast<double>(val);
        double a  = qt.alpha;
        double b  = qt.beta;
        val = ((in + qt.d4) - a * (qt.d1 + qt.d3) + b * qt.d2) * qt.scale_inv;
        qt.d4 = qt.d3;
        qt.d3 = qt.d2;
        qt.d2 = qt.d1;
        qt.d1 = in;
    }
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
            val *= static_cast<long double>(rem_state.h[0]);
        }
    } else {
        val = rem_state.push(static_cast<double>(val));
    }

    // ── Коэффициент усиления ─────────────────────────────────
    val *= static_cast<long double>(gain);
    peak_internal = std::max(peak_internal, std::abs(val));

    return static_cast<double>(val);
}

sample_t CascadeFilterState::push(sample_t x)
{
    return static_cast<sample_t>(push_double(static_cast<double>(x)));
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

std::vector<double> filter_cascade_double(const CascadeDecomposition& dec,
                                          const std::vector<double>& x)
{
    CascadeFilterState state;
    state.init(dec);

    std::vector<double> y(x.size(), 0.0);
    for (size_t n = 0; n < x.size(); ++n) {
        y[n] = state.push_double(x[n]);
    }
    return y;
}
