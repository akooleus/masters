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
//  Обычный путь использует double/long double. Для декомпозиций с явно
//  указанным runtime_precision создаётся состояние double-double, binary128
//  либо Boost.Multiprecision. ISA-kernel выбирается отдельно от численной
//  точности; поля диагностики backend не выбирают.
//
//  Собственная реализация настоящей работы.
// ═══════════════════════════════════════════════════════════════

#include "cascade_fir.h"

#include <boost/multiprecision/cpp_bin_float.hpp>

#include <stdexcept>

#if (defined(__x86_64__) || defined(__i386__)) \
    && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define CASCADE_HAS_X86_AVX2_TARGET 1
#else
#define CASCADE_HAS_X86_AVX2_TARGET 0
#endif

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
    case CascadeRuntimePrecision::DoubleDouble:
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
    explicit MultiprecisionCascadeState(const CascadeDecomposition& dec,
                                        bool track_peak)
        : execution_order_(dec.execution_order), gain_(dec.runtime_gain),
          track_peak_(track_peak)
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
        if (!track_peak_) {
            return;
        }
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
    bool track_peak_ = false;
};

struct DoubleDouble {
    double high;
    double low;
};

DoubleDouble quick_two_sum(double a, double b)
{
    const double sum = a + b;
    return {sum, b - (sum - a)};
}

DoubleDouble two_sum(double a, double b)
{
    const double sum = a + b;
    const double b_virtual = sum - a;
    const double error = (a - (sum - b_virtual)) + (b - b_virtual);
    return {sum, error};
}

DoubleDouble add_double_double(DoubleDouble a, DoubleDouble b)
{
    DoubleDouble high = two_sum(a.high, b.high);
    DoubleDouble low = two_sum(a.low, b.low);
    high.low += low.high;
    high = quick_two_sum(high.high, high.low);
    high.low += low.low;
    return quick_two_sum(high.high, high.low);
}

double product_error(double a, double b, double product)
{
    // Dekker split: exact product residual without requiring FMA support.
    static constexpr double SPLITTER = 134217729.0; // 2^27 + 1
    const double a_split = SPLITTER * a;
    const double a_high = a_split - (a_split - a);
    const double a_low = a - a_high;
    const double b_split = SPLITTER * b;
    const double b_high = b_split - (b_split - b);
    const double b_low = b - b_high;
    return ((a_high * b_high - product) + a_high * b_low
            + a_low * b_high) + a_low * b_low;
}

DoubleDouble multiply_double_double(DoubleDouble value, double factor)
{
    const double product = value.high * factor;
    const double error = product_error(value.high, factor, product)
                       + value.low * factor;
    return quick_two_sum(product, error);
}

bool cpu_supports_avx2_fma()
{
#if CASCADE_HAS_X86_AVX2_TARGET
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
    return false;
#endif
}

#if CASCADE_HAS_X86_AVX2_TARGET

struct DoubleDoubleVector {
    __m256d high;
    __m256d low;
};

__attribute__((target("avx2,fma")))
DoubleDoubleVector quick_two_sum_vector(__m256d a, __m256d b)
{
    const __m256d sum = _mm256_add_pd(a, b);
    return {sum, _mm256_sub_pd(b, _mm256_sub_pd(sum, a))};
}

__attribute__((target("avx2,fma")))
DoubleDoubleVector two_sum_vector(__m256d a, __m256d b)
{
    const __m256d sum = _mm256_add_pd(a, b);
    const __m256d b_virtual = _mm256_sub_pd(sum, a);
    const __m256d error = _mm256_add_pd(
        _mm256_sub_pd(a, _mm256_sub_pd(sum, b_virtual)),
        _mm256_sub_pd(b, b_virtual));
    return {sum, error};
}

__attribute__((target("avx2,fma")))
DoubleDoubleVector add_double_double_vector(DoubleDoubleVector a,
                                           DoubleDoubleVector b)
{
    DoubleDoubleVector high = two_sum_vector(a.high, b.high);
    DoubleDoubleVector low = two_sum_vector(a.low, b.low);
    high.low = _mm256_add_pd(high.low, low.high);
    high = quick_two_sum_vector(high.high, high.low);
    high.low = _mm256_add_pd(high.low, low.low);
    return quick_two_sum_vector(high.high, high.low);
}

__attribute__((target("avx2,fma")))
DoubleDoubleVector multiply_double_double_vector(
    const double* high,
    const double* low,
    const double* factors)
{
    const __m256d value_high = _mm256_loadu_pd(high);
    const __m256d value_low = _mm256_loadu_pd(low);
    const __m256d factor = _mm256_loadu_pd(factors);
    const __m256d product = _mm256_mul_pd(value_high, factor);
    const __m256d error = _mm256_add_pd(
        _mm256_fmsub_pd(value_high, factor, product),
        _mm256_mul_pd(value_low, factor));
    return quick_two_sum_vector(product, error);
}

__attribute__((target("avx2,fma")))
DoubleDouble reduce_double_double_vector(DoubleDoubleVector value)
{
    DoubleDoubleVector swapped{
        _mm256_permute2f128_pd(value.high, value.high, 0x01),
        _mm256_permute2f128_pd(value.low, value.low, 0x01)};
    value = add_double_double_vector(value, swapped);
    swapped = {
        _mm256_permute_pd(value.high, 0x05),
        _mm256_permute_pd(value.low, 0x05)};
    value = add_double_double_vector(value, swapped);
    return {
        _mm_cvtsd_f64(_mm256_castpd256_pd128(value.high)),
        _mm_cvtsd_f64(_mm256_castpd256_pd128(value.low))};
}

__attribute__((target("avx2,fma")))
DoubleDouble multiply_double_double_fma(DoubleDouble value, double factor)
{
    const double product = value.high * factor;
    const double error = std::fma(value.high, factor, -product)
                       + value.low * factor;
    return quick_two_sum(product, error);
}

__attribute__((target("avx2,fma")))
DoubleDouble dot_order_8_avx2(const double* coefficients,
                              const double* delay_high,
                              const double* delay_low,
                              DoubleDouble current)
{
    DoubleDoubleVector products = multiply_double_double_vector(
        delay_high, delay_low, coefficients + 1u);
    const DoubleDoubleVector upper = multiply_double_double_vector(
        delay_high + 4u, delay_low + 4u, coefficients + 5u);
    products = add_double_double_vector(products, upper);
    const DoubleDouble delayed_sum = reduce_double_double_vector(products);
    return add_double_double(
        multiply_double_double_fma(current, coefficients[0]), delayed_sum);
}

__attribute__((target("avx2,fma")))
DoubleDouble dot_order_4_avx2(const double* coefficients,
                              const double* delay_high,
                              const double* delay_low,
                              DoubleDouble current)
{
    const DoubleDoubleVector products = multiply_double_double_vector(
        delay_high, delay_low, coefficients + 1u);
    const DoubleDouble delayed_sum = reduce_double_double_vector(products);
    return add_double_double(
        multiply_double_double_fma(current, coefficients[0]), delayed_sum);
}

#endif

class DoubleDoubleCascadeState final : public CascadeExtendedState {
public:
    explicit DoubleDoubleCascadeState(const CascadeDecomposition& dec,
                                      bool track_peak,
                                      bool use_avx2)
        : execution_order_(dec.execution_order), remainder_(dec.remainder),
          gain_(dec.runtime_gain), track_peak_(track_peak),
          use_avx2_(use_avx2)
    {
        blocks_.resize(dec.blocks.size());
        for (size_t i = 0; i < dec.blocks.size(); ++i) {
            blocks_[i].coefficients = dec.blocks[i].coefficients;
            const size_t order = dec.blocks[i].coefficients.size() - 1u;
            blocks_[i].delay_high.assign(order, 0.0);
            blocks_[i].delay_low.assign(order, 0.0);
        }
        const size_t remainder_order = remainder_.size() - 1u;
        remainder_delay_.assign(
            remainder_order, DoubleDouble{0.0, 0.0});
    }

    double push(double x) override
    {
        DoubleDouble value{x, 0.0};
        update_peak(value);
        for (const CascadeSectionPlacement& placement : execution_order_) {
            Block& block = blocks_[placement.index];
            DoubleDouble output;
            if (block.coefficients.size() == 9u) {
#if CASCADE_HAS_X86_AVX2_TARGET
                output = use_avx2_
                    ? dot_order_8_avx2(
                        block.coefficients.data(),
                        block.delay_high.data(),
                        block.delay_low.data(), value)
                    : dot_order_8(block, value);
#else
                output = dot_order_8(block, value);
#endif
                shift_order_8(block, value);
            } else if (block.coefficients.size() == 5u) {
#if CASCADE_HAS_X86_AVX2_TARGET
                output = use_avx2_
                    ? dot_order_4_avx2(
                        block.coefficients.data(),
                        block.delay_high.data(),
                        block.delay_low.data(), value)
                    : dot_order_4(block, value);
#else
                output = dot_order_4(block, value);
#endif
                shift_order_4(block, value);
            } else {
                output = multiply_double_double(
                    value, block.coefficients.front());
                for (size_t i = 1; i < block.coefficients.size(); ++i) {
                    output = add_double_double(
                        output,
                        multiply_double_double(
                            delay(block, i - 1u), block.coefficients[i]));
                }
                for (size_t i = block.delay_high.size(); i-- > 1u;) {
                    block.delay_high[i] = block.delay_high[i - 1u];
                    block.delay_low[i] = block.delay_low[i - 1u];
                }
                if (!block.delay_high.empty()) {
                    set_delay(block, 0u, value);
                }
            }
            value = multiply_double_double(output, placement.scale);
            update_peak(value);
        }

        if (remainder_.size() == 1u) {
            value = multiply_double_double(value, remainder_.front());
        } else {
            DoubleDouble output = multiply_double_double(
                value, remainder_.front());
            for (size_t i = 1; i < remainder_.size(); ++i) {
                output = add_double_double(
                    output,
                    multiply_double_double(
                        remainder_delay_[i - 1u], remainder_[i]));
            }
            for (size_t i = remainder_delay_.size(); i-- > 1u;) {
                remainder_delay_[i] = remainder_delay_[i - 1u];
            }
            if (!remainder_delay_.empty()) {
                remainder_delay_[0] = value;
            }
            value = output;
        }

        value = multiply_double_double(value, gain_);
        update_peak(value);
        return value.high + value.low;
    }

    void reset() override
    {
        for (Block& block : blocks_) {
            std::fill(block.delay_high.begin(), block.delay_high.end(), 0.0);
            std::fill(block.delay_low.begin(), block.delay_low.end(), 0.0);
        }
        std::fill(
            remainder_delay_.begin(), remainder_delay_.end(),
            DoubleDouble{0.0, 0.0});
        peak_ = 0.0L;
    }

    long double peak() const override
    {
        return peak_;
    }

private:
    struct Block {
        std::vector<double> coefficients;
        std::vector<double> delay_high;
        std::vector<double> delay_low;
    };

    static DoubleDouble delay(const Block& block, size_t index)
    {
        return {block.delay_high[index], block.delay_low[index]};
    }

    static void set_delay(Block& block,
                          size_t index,
                          DoubleDouble value)
    {
        block.delay_high[index] = value.high;
        block.delay_low[index] = value.low;
    }

    static DoubleDouble dot_order_4(const Block& block,
                                    DoubleDouble value)
    {
        DoubleDouble output = multiply_double_double(
            value, block.coefficients[0]);
        output = add_double_double(output, multiply_double_double(
            delay(block, 0u), block.coefficients[1]));
        output = add_double_double(output, multiply_double_double(
            delay(block, 1u), block.coefficients[2]));
        output = add_double_double(output, multiply_double_double(
            delay(block, 2u), block.coefficients[3]));
        return add_double_double(output, multiply_double_double(
            delay(block, 3u), block.coefficients[4]));
    }

    static DoubleDouble dot_order_8(const Block& block,
                                    DoubleDouble value)
    {
        DoubleDouble output = multiply_double_double(
            value, block.coefficients[0]);
        output = add_double_double(output, multiply_double_double(
            delay(block, 0u), block.coefficients[1]));
        output = add_double_double(output, multiply_double_double(
            delay(block, 1u), block.coefficients[2]));
        output = add_double_double(output, multiply_double_double(
            delay(block, 2u), block.coefficients[3]));
        output = add_double_double(output, multiply_double_double(
            delay(block, 3u), block.coefficients[4]));
        output = add_double_double(output, multiply_double_double(
            delay(block, 4u), block.coefficients[5]));
        output = add_double_double(output, multiply_double_double(
            delay(block, 5u), block.coefficients[6]));
        output = add_double_double(output, multiply_double_double(
            delay(block, 6u), block.coefficients[7]));
        return add_double_double(output, multiply_double_double(
            delay(block, 7u), block.coefficients[8]));
    }

    static void shift_order_4(Block& block, DoubleDouble value)
    {
        set_delay(block, 3u, delay(block, 2u));
        set_delay(block, 2u, delay(block, 1u));
        set_delay(block, 1u, delay(block, 0u));
        set_delay(block, 0u, value);
    }

    static void shift_order_8(Block& block, DoubleDouble value)
    {
        set_delay(block, 7u, delay(block, 6u));
        set_delay(block, 6u, delay(block, 5u));
        set_delay(block, 5u, delay(block, 4u));
        set_delay(block, 4u, delay(block, 3u));
        set_delay(block, 3u, delay(block, 2u));
        set_delay(block, 2u, delay(block, 1u));
        set_delay(block, 1u, delay(block, 0u));
        set_delay(block, 0u, value);
    }

    void update_peak(DoubleDouble value)
    {
        if (!track_peak_) {
            return;
        }
        const long double magnitude = std::abs(
            static_cast<long double>(value.high)
            + static_cast<long double>(value.low));
        peak_ = std::max(peak_, magnitude);
    }

    std::vector<Block> blocks_;
    std::vector<CascadeSectionPlacement> execution_order_;
    std::vector<double> remainder_;
    std::vector<DoubleDouble> remainder_delay_;
    double gain_;
    long double peak_ = 0.0L;
    bool track_peak_ = false;
    bool use_avx2_ = false;
};

} // namespace

bool cascade_runtime_kernel_available(CascadeRuntimeKernel kernel)
{
    switch (kernel) {
    case CascadeRuntimeKernel::Auto:
    case CascadeRuntimeKernel::Scalar:
        return true;
    case CascadeRuntimeKernel::Avx2Fma:
        return cpu_supports_avx2_fma();
    }
    return false;
}

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

void CascadeFilterState::init(const CascadeDecomposition& dec,
                              CascadeRuntimeOptions options)
{
    initialized = false;
    validate_runtime_decomposition(dec);
    track_peak = options.track_peak;
    selected_kernel = CascadeRuntimeKernel::Scalar;
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
        if (dec.runtime_precision == CascadeRuntimePrecision::DoubleDouble) {
            const bool avx2_available = cpu_supports_avx2_fma();
            if (options.kernel == CascadeRuntimeKernel::Avx2Fma
                && !avx2_available) {
                throw std::invalid_argument(
                    "AVX2/FMA cascade kernel is unavailable on this CPU");
            }
            const bool use_avx2 = avx2_available
                && options.kernel != CascadeRuntimeKernel::Scalar;
            extended_state =
                std::make_unique<DoubleDoubleCascadeState>(
                    dec, track_peak, use_avx2);
            selected_kernel = use_avx2
                ? CascadeRuntimeKernel::Avx2Fma
                : CascadeRuntimeKernel::Scalar;
        } else if (dec.runtime_precision == CascadeRuntimePrecision::Extended34) {
            if (options.kernel == CascadeRuntimeKernel::Avx2Fma) {
                throw std::invalid_argument(
                    "AVX2/FMA does not implement binary128 arithmetic");
            }
            extended_state =
                std::make_unique<MultiprecisionCascadeState<runtime_mp34>>(
                    dec, track_peak);
        } else if (dec.runtime_precision == CascadeRuntimePrecision::Extended50) {
            if (options.kernel == CascadeRuntimeKernel::Avx2Fma) {
                throw std::invalid_argument(
                    "AVX2/FMA does not implement Extended50 arithmetic");
            }
            extended_state =
                std::make_unique<MultiprecisionCascadeState<runtime_mp50>>(
                    dec, track_peak);
        } else {
            throw std::invalid_argument(
                "unsupported cascade runtime precision");
        }
        gain = 1.0;
        initialized = true;
        return;
    }
    if (options.kernel == CascadeRuntimeKernel::Avx2Fma) {
        throw std::invalid_argument(
            "AVX2/FMA kernel is only available for double-double blocks");
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
        if (track_peak) {
            peak_internal = extended_state->peak();
        }
        return output;
    }
    long double val = static_cast<long double>(x);
    if (track_peak) {
        peak_internal = std::max(peak_internal, std::abs(val));
    }

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
            if (track_peak) {
                peak_internal = std::max(peak_internal, std::abs(val));
            }
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
    if (track_peak) {
        peak_internal = std::max(peak_internal, std::abs(val));
    }

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
