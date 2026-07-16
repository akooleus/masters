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
//  явно указанным runtime_precision создаётся отдельное состояние binary128
//  либо Boost.Multiprecision; поля диагностики backend не выбирают.
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

#if defined(__SIZEOF_FLOAT128__)
using runtime_mp34 = __float128;
#else
using runtime_mp34 = boost::multiprecision::number<
    boost::multiprecision::cpp_bin_float<34>>;
#endif
using runtime_mp50 = boost::multiprecision::number<
    boost::multiprecision::cpp_bin_float<50>>;

void validate_runtime_decomposition(const CascadeDecomposition& dec)
{
    if (!std::isfinite(dec.gain) || !std::isfinite(dec.runtime_gain)) {
        throw std::invalid_argument("cascade gain is not finite");
    }
    if (dec.remainder.empty()) {
        throw std::invalid_argument("cascade remainder is empty");
    }
    for (double coefficient : dec.remainder) {
        if (!std::isfinite(coefficient)) {
            throw std::invalid_argument(
                "cascade remainder coefficient is not finite");
        }
    }
    for (const FirstOrder& section : dec.first_order) {
        if (section.sign != -1 && section.sign != 1) {
            throw std::invalid_argument("invalid first-order section sign");
        }
    }
    for (const Biquad& section : dec.biquads) {
        if (!std::isfinite(section.gamma)) {
            throw std::invalid_argument("biquad coefficient is not finite");
        }
    }
    for (const Quartic& section : dec.quartics) {
        if (!std::isfinite(section.alpha) || !std::isfinite(section.beta)) {
            throw std::invalid_argument("quartic coefficient is not finite");
        }
    }
    for (const CascadeBlock& block : dec.blocks) {
        if (block.coefficients.empty()) {
            throw std::invalid_argument("cascade block is empty");
        }
        for (double coefficient : block.coefficients) {
            if (!std::isfinite(coefficient)) {
                throw std::invalid_argument(
                    "cascade block coefficient is not finite");
            }
        }
    }

    const size_t section_count = dec.first_order.size() + dec.biquads.size()
                               + dec.quartics.size() + dec.blocks.size();
    if (dec.execution_order.empty()) {
        if (!dec.blocks.empty()) {
            throw std::invalid_argument(
                "block cascade has no explicit execution order");
        }
    } else {
        if (dec.execution_order.size() != section_count) {
            throw std::invalid_argument(
                "execution order does not cover every cascade section");
        }
        std::vector<bool> seen_first(dec.first_order.size(), false);
        std::vector<bool> seen_biquad(dec.biquads.size(), false);
        std::vector<bool> seen_quartic(dec.quartics.size(), false);
        std::vector<bool> seen_block(dec.blocks.size(), false);
        for (const CascadeSectionPlacement& placement : dec.execution_order) {
            if (!std::isfinite(placement.scale) || placement.scale <= 0.0) {
                throw std::invalid_argument(
                    "cascade execution scale is invalid");
            }
            std::vector<bool>* seen = nullptr;
            switch (placement.kind) {
            case CascadeSectionKind::FirstOrder:
                seen = &seen_first;
                break;
            case CascadeSectionKind::Biquad:
                seen = &seen_biquad;
                break;
            case CascadeSectionKind::Quartic:
                seen = &seen_quartic;
                break;
            case CascadeSectionKind::Block:
                seen = &seen_block;
                break;
            default:
                throw std::invalid_argument("invalid cascade section kind");
            }
            if (placement.index >= seen->size()) {
                throw std::invalid_argument(
                    "cascade execution index is out of range");
            }
            if ((*seen)[placement.index]) {
                throw std::invalid_argument(
                    "cascade execution order contains a duplicate section");
            }
            (*seen)[placement.index] = true;
        }
    }

    switch (dec.runtime_precision) {
    case CascadeRuntimePrecision::Native:
        break;
    case CascadeRuntimePrecision::Extended34:
    case CascadeRuntimePrecision::Extended50:
        if (!dec.first_order.empty() || !dec.biquads.empty()
            || !dec.quartics.empty() || dec.execution_order.empty()) {
            throw std::invalid_argument(
                "extended runtime requires an explicit block-only cascade");
        }
        for (const CascadeSectionPlacement& placement : dec.execution_order) {
            if (placement.kind != CascadeSectionKind::Block) {
                throw std::invalid_argument(
                    "extended runtime requires a block-only execution order");
            }
        }
        break;
    default:
        throw std::invalid_argument("unsupported cascade runtime precision");
    }
}

template <class RuntimeReal>
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
            blocks_[i].delay.assign(order, RuntimeReal(0));
        }

        for (double coefficient : dec.remainder) {
            remainder_coefficients_.emplace_back(coefficient);
        }
        const size_t remainder_order = dec.remainder.empty()
            ? 0 : dec.remainder.size() - 1;
        remainder_delay_.assign(remainder_order, RuntimeReal(0));

        for (const CascadeSectionPlacement& placement : execution_order_) {
            if (placement.kind != CascadeSectionKind::Block
                || placement.index >= blocks_.size()) {
                throw std::invalid_argument(
                    "multiprecision runtime requires a block-only execution order");
            }
            for (RuntimeReal& coefficient
                 : blocks_[placement.index].coefficients) {
                coefficient *= RuntimeReal(placement.scale);
            }
        }
    }

    double push(double x) override
    {
        RuntimeReal value(x);
        update_peak(value);
        for (const CascadeSectionPlacement& placement : execution_order_) {
            Block& block = blocks_[placement.index];
            RuntimeReal output;
            if (block.coefficients.size() == 9 && block.delay.size() == 8) {
                output = block.coefficients[0] * value
                       + block.coefficients[1] * block.delay[0]
                       + block.coefficients[2] * block.delay[1]
                       + block.coefficients[3] * block.delay[2]
                       + block.coefficients[4] * block.delay[3]
                       + block.coefficients[5] * block.delay[4]
                       + block.coefficients[6] * block.delay[5]
                       + block.coefficients[7] * block.delay[6]
                       + block.coefficients[8] * block.delay[7];
                block.delay[7] = block.delay[6];
                block.delay[6] = block.delay[5];
                block.delay[5] = block.delay[4];
                block.delay[4] = block.delay[3];
                block.delay[3] = block.delay[2];
                block.delay[2] = block.delay[1];
                block.delay[1] = block.delay[0];
                block.delay[0] = value;
            } else {
                output = block.coefficients.empty()
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
            }
            value = output;
            update_peak(value);
        }

        if (remainder_coefficients_.size() == 1) {
            value *= remainder_coefficients_[0];
        } else if (!remainder_coefficients_.empty()) {
            RuntimeReal output = remainder_coefficients_[0] * value;
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
            std::fill(block.delay.begin(), block.delay.end(), RuntimeReal(0));
        }
        std::fill(
            remainder_delay_.begin(), remainder_delay_.end(), RuntimeReal(0));
        peak_ = 0.0L;
    }

    long double peak() const override
    {
        return peak_;
    }

private:
    struct Block {
        std::vector<RuntimeReal> coefficients;
        std::vector<RuntimeReal> delay;
    };

    void update_peak(const RuntimeReal& value)
    {
        const long double magnitude =
            std::abs(static_cast<long double>(value));
        peak_ = std::max(peak_, magnitude);
    }

    std::vector<Block> blocks_;
    std::vector<CascadeSectionPlacement> execution_order_;
    std::vector<RuntimeReal> remainder_coefficients_;
    std::vector<RuntimeReal> remainder_delay_;
    RuntimeReal gain_;
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
    initialized = false;
    if (coeffs.empty()) {
        throw std::invalid_argument("double FIR coefficients are empty");
    }
    h   = coeffs;
    buf.assign(coeffs.size(), 0.0);
    idx = 0;
    initialized = true;
}

void DoubleFilterState::reset()
{
    if (!initialized) {
        throw std::logic_error("double FIR state is not initialized");
    }
    std::fill(buf.begin(), buf.end(), 0.0);
    idx = 0;
}

double DoubleFilterState::push(double x)
{
    if (!initialized) {
        throw std::logic_error("double FIR state is not initialized");
    }
    const unsigned N = static_cast<unsigned>(h.size());

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
    initialized = false;
    validate_runtime_decomposition(dec);
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
    if (dec.runtime_precision != CascadeRuntimePrecision::Native) {
        if (dec.runtime_precision == CascadeRuntimePrecision::Extended34) {
            extended_state =
                std::make_unique<MultiprecisionCascadeState<runtime_mp34>>(dec);
        } else if (dec.runtime_precision == CascadeRuntimePrecision::Extended50) {
            extended_state =
                std::make_unique<MultiprecisionCascadeState<runtime_mp50>>(dec);
        } else {
            throw std::invalid_argument(
                "unsupported cascade runtime precision");
        }
        gain = 1.0;
        initialized = true;
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
    initialized = true;
}

// ═══════════════════════════════════════════════════════════════
//  CascadeFilterState — сброс состояния (обнуление задержек)
// ═══════════════════════════════════════════════════════════════

void CascadeFilterState::reset()
{
    if (!initialized) {
        throw std::logic_error("cascade FIR state is not initialized");
    }
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
    if (!initialized) {
        throw std::logic_error("cascade FIR state is not initialized");
    }
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
