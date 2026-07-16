#ifndef CASCADE_FIR_H
#define CASCADE_FIR_H

#include <cstddef>
#include <cstdint>
#include <vector>
#include <string>
#include <cmath>
#include <complex>
#include <numeric>
#include <algorithm>
#include <cassert>
#include <array>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <limits>
#include <memory>

// ═══════════════════════════════════════════════════════════════
//  Базовые типы
// ═══════════════════════════════════════════════════════════════

using real_t   = double;
using sample_t = float;   // формат отсчётов сигнала

static constexpr real_t PI = 3.14159265358979323846;

// ═══════════════════════════════════════════════════════════════
//  Спецификация фильтра частотной выборки
// ═══════════════════════════════════════════════════════════════

struct FilterSpec {
    unsigned N;          // длина фильтра (число коэффициентов)
    real_t   fs;         // частота дискретизации, Гц
    real_t   f_pass;     // верхняя граница полосы пропускания, Гц
    real_t   f_stop;     // нижняя граница полосы заграждения, Гц

    // Вычислить номера выборок, попадающих в полосу пропускания,
    // переходную и полосу заграждения
    unsigned k_pass() const {
        return static_cast<unsigned>(std::floor(f_pass * N / fs));
    }
    unsigned k_stop() const {
        return static_cast<unsigned>(std::ceil(f_stop * N / fs));
    }
};

// ═══════════════════════════════════════════════════════════════
//  Результат проектирования (прямая форма)
// ═══════════════════════════════════════════════════════════════

struct DirectFIR {
    std::vector<real_t> h;                  // коэффициенты h[0..N-1]
    FilterSpec          spec;

    // Исходные вещественные амплитудные выборки метода частотной
    // выборки. Это источник истины для аналитически известных нулей:
    // нулевая выборка задаёт корень точно, без обратного распознавания
    // нуля по уже округлённым коэффициентам h[n].
    std::vector<real_t> frequency_samples;

    unsigned order()  const {
        return h.empty() ? 0u : static_cast<unsigned>(h.size() - 1u);
    }
    unsigned length() const { return static_cast<unsigned>(h.size()); }
};

// ═══════════════════════════════════════════════════════════════
//  Каскадное звено 2-го порядка: (1 − γ·z⁻¹ + z⁻²)
// ═══════════════════════════════════════════════════════════════

struct Biquad {
    real_t gamma;        // γ = 2·cos(ω), где ω — частота нуля
    unsigned k_index;    // номер частотной выборки, породившей этот нуль

    // Нули звена: z = exp(±jω), ω = acos(γ/2)
    real_t omega() const { return std::acos(gamma / 2.0); }
};

// ═══════════════════════════════════════════════════════════════
//  Каскадное звено 4-го порядка (палиндромное):
//  (1 − α·z⁻¹ + β·z⁻² − α·z⁻³ + z⁻⁴)
//
//  Соответствует четвёрке нулей: z₀, z₀*, 1/z₀, 1/z₀*
//  или паре комплексно-сопряжённых u-корней u₀ = a+jb, u₀*.
//
//  Фильтрация с палиндромной симметрией:
//    y[n] = (x[n] + x[n−4]) − α·(x[n−1] + x[n−3]) + β·x[n−2]
//
//  Умножений: 2 (на α и β), Сложений: 4, Регистров: 4
// ═══════════════════════════════════════════════════════════════

struct Quartic {
    real_t alpha;        // α = 2·Re(u)
    real_t beta;         // β = 2 + |u|²
};

// ═══════════════════════════════════════════════════════════════
//  Каскадное звено 1-го порядка: (1 ± z⁻¹)
// ═══════════════════════════════════════════════════════════════

struct FirstOrder {
    int sign;            // +1 → (1 + z⁻¹), нуль z = −1
                         // −1 → (1 − z⁻¹), нуль z = +1
};

// Низкопорядковый подфильтр, полученный устойчивым объединением нескольких
// корневых сомножителей в повышенной точности. Такие блоки нужны, когда
// отдельное округление каждого близкого корня в double разрушает каскад.
struct CascadeBlock {
    std::vector<real_t> coefficients;
};

enum class CascadeSectionKind {
    FirstOrder,
    Biquad,
    Quartic,
    Block
};

// Ссылка на типизированное звено и его runtime-масштаб. Поля
// first_order/biquads/quartics сохраняют математическое представление,
// а этот список задаёт устойчивый порядок фактического исполнения.
struct CascadeSectionPlacement {
    CascadeSectionKind kind;
    std::size_t index;
    real_t scale;
};

enum class CascadeBuildStatus {
    Unspecified,
    AnalyticalReduction,
    ShortBlockCascade,
    HighPrecisionShortBlockCascade,
    FullOrderFactorBlock,
    DirectFormFallback,
    Failed
};

// Арифметика, которую должен использовать runtime. Это часть исполняемого
// представления каскада, а не диагностическое поле.
enum class CascadeRuntimePrecision {
    Native = 0,
    Extended34 = 34,
    Extended50 = 50
};

// Численные свидетельства, по которым вызывающий код может отличить
// короткий каскад от безопасного отката к прямой форме.
struct CascadeDiagnostics {
    CascadeBuildStatus status = CascadeBuildStatus::Unspecified;

    bool complement_verified = false;
    real_t complement_identity_error =
        std::numeric_limits<real_t>::infinity();

    bool roots_verified = false;
    unsigned root_iterations = 0;
    real_t root_backward_error =
        std::numeric_limits<real_t>::infinity();
    real_t root_rebuild_error =
        std::numeric_limits<real_t>::infinity();

    bool runtime_verified = false;
    real_t runtime_impulse_error =
        std::numeric_limits<real_t>::infinity();
    real_t runtime_peak_internal =
        std::numeric_limits<real_t>::infinity();
    unsigned selected_max_block_order = 0;
    unsigned runtime_decimal_digits = 0;
};

// ═══════════════════════════════════════════════════════════════
//  Результат каскадной декомпозиции
// ═══════════════════════════════════════════════════════════════

struct CascadeDecomposition {
    std::vector<FirstOrder> first_order;   // звенья 1-го порядка
    std::vector<Biquad>     biquads;       // звенья 2-го порядка
    std::vector<Quartic>    quartics;      // звенья 4-го порядка
    std::vector<CascadeBlock> blocks;      // адаптивно объединённые подфильтры
    std::vector<real_t>     remainder;     // полином-остаток h_rem[0..M-1]
    real_t                  gain = 1.0;    // коэффициент математического представления

    // Если список непуст, CascadeFilterState применяет звенья именно
    // в этом порядке и с указанными масштабами. runtime_gain уже
    // учитывает эти масштабы и не обязан совпадать с gain.
    std::vector<CascadeSectionPlacement> execution_order;
    real_t runtime_gain = 1.0;
    CascadeRuntimePrecision runtime_precision =
        CascadeRuntimePrecision::Native;
    CascadeDiagnostics diagnostics;

    // Общее число выделенных нулей
    unsigned zeros_extracted() const {
        unsigned count = static_cast<unsigned>(first_order.size())
             + static_cast<unsigned>(biquads.size()) * 2
             + static_cast<unsigned>(quartics.size()) * 4;
        for (const auto& block : blocks) {
            if (!block.coefficients.empty()) {
                count += static_cast<unsigned>(block.coefficients.size() - 1);
            }
        }
        return count;
    }
};

// Версионированный артефакт готового каскада. Отпечаток связывает его с
// конкретными h[n], частотными выборками и спецификацией исходного фильтра.
// Сам формат файла реализован без внешних библиотек.
static constexpr std::uint32_t CASCADE_ARTIFACT_FORMAT_VERSION = 1u;

struct CascadeArtifactMetadata {
    std::uint32_t format_version = CASCADE_ARTIFACT_FORMAT_VERSION;
    FilterSpec source_spec{0u, 0.0, 0.0, 0.0};
    std::uint32_t source_coefficient_count = 0u;
    std::uint64_t source_fingerprint = 0u;
};

struct CascadeArtifact {
    CascadeArtifactMetadata metadata;
    CascadeDecomposition decomposition;
};

std::uint64_t cascade_source_fingerprint(const DirectFIR& fir);
CascadeArtifact make_cascade_artifact(
    const DirectFIR& fir,
    const CascadeDecomposition& decomposition);
bool cascade_artifact_matches(const CascadeArtifact& artifact,
                              const DirectFIR& fir);
void validate_cascade_artifact(const CascadeArtifact& artifact);
void save_cascade_artifact(const std::string& path,
                           const CascadeArtifact& artifact);
CascadeArtifact load_cascade_artifact(const std::string& path);

// ═══════════════════════════════════════════════════════════════
//  Состояние прямого КИХ-фильтра (для пооотсчётной фильтрации)
// ═══════════════════════════════════════════════════════════════

struct DirectFilterState {
    std::vector<real_t> h;         // коэффициенты
    std::vector<sample_t> buf;     // кольцевой буфер входных отсчётов
    unsigned idx = 0;              // текущая позиция в буфере
    bool initialized = false;

    void init(const std::vector<real_t>& coeffs);
    sample_t push(sample_t x);     // подать отсчёт, вернуть выход
    void reset();
};

// ═══════════════════════════════════════════════════════════════
//  Состояние прямого КИХ-фильтра в double (для остатка каскада)
//
//  Идентичен DirectFilterState, но буфер и аккумулятор в double.
//  Используется внутри CascadeFilterState для фильтрации
//  полинома-остатка без потери точности float→double→float
//  на каждом отсчёте.
// ═══════════════════════════════════════════════════════════════

struct DoubleFilterState {
    std::vector<real_t> h;         // коэффициенты (double)
    std::vector<double> buf;       // кольцевой буфер (double!)
    unsigned idx = 0;
    bool initialized = false;

    void init(const std::vector<real_t>& coeffs);
    double push(double x);         // вход/выход в double
    void reset();
};

// Внутренний полиморфный backend для редких каскадов, которым недостаточно
// аппаратного long double. Конкретный multiprecision-тип скрыт в .cpp.
struct CascadeExtendedState {
    virtual ~CascadeExtendedState() = default;
    virtual double push(double x) = 0;
    virtual void reset() = 0;
    virtual long double peak() const = 0;
};

// ═══════════════════════════════════════════════════════════════
//  Состояние каскадного КИХ-фильтра
// ═══════════════════════════════════════════════════════════════

struct CascadeFilterState {
    // Для каждого звена 2-го порядка: два элемента задержки
    struct BiquadState {
        real_t gamma;
        real_t scale_inv;
        double d1, d2;             // x[n-1], x[n-2] — в double для точности
    };
    // Для каждого звена 1-го порядка: один элемент задержки
    struct FirstOrderState {
        int sign;
        real_t scale_inv;
        double d1;                 // в double для точности
    };
    // Для каждого звена 4-го порядка: четыре элемента задержки
    //  y[n] = (x[n] + x[n−4]) − α·(x[n−1] + x[n−3]) + β·x[n−2]
    struct QuarticState {
        real_t alpha, beta;
        real_t scale_inv;
        double d1, d2, d3, d4;     // x[n-1] ... x[n-4], в double
    };
    struct BlockState {
        std::vector<real_t> coefficients;
        std::vector<long double> delay;
    };

    std::vector<FirstOrderState> fo_states;
    std::vector<BiquadState>     bq_states;
    std::vector<QuarticState>    qt_states;
    std::vector<BlockState>      block_states;
    std::vector<CascadeSectionPlacement> execution_order;
    DoubleFilterState            rem_state;    // фильтр-остаток (double!)
    real_t                       gain = 1.0;
    long double                 peak_internal = 0.0L;
    std::unique_ptr<CascadeExtendedState> extended_state;
    bool initialized = false;

    CascadeFilterState() = default;
    CascadeFilterState(const CascadeFilterState&) = delete;
    CascadeFilterState& operator=(const CascadeFilterState&) = delete;
    CascadeFilterState(CascadeFilterState&&) = default;
    CascadeFilterState& operator=(CascadeFilterState&&) = default;

    void init(const CascadeDecomposition& dec);
    sample_t push(sample_t x);
    double push_double(double x);
    void reset();
};

// ═══════════════════════════════════════════════════════════════
//  Метрики сравнения двух выходных последовательностей
// ═══════════════════════════════════════════════════════════════

struct CompareMetrics {
    double max_abs_err;            // max |y1[n] − y2[n]|
    double rms_err;                // sqrt( mean( (y1−y2)² ) )
    double max_rel_err;            // max |y1−y2| / max(|y1|,|y2|)
    double snr_db;                 // 10·log10( Σy1² / Σ(y1−y2)² )
    size_t num_samples;

    void print(const std::string& label) const;
};

CompareMetrics compare_signals(const std::vector<sample_t>& y1,
                               const std::vector<sample_t>& y2);

// ═══════════════════════════════════════════════════════════════
//  Метрики сравнения коэффициентов (double)
// ═══════════════════════════════════════════════════════════════

struct CoeffMetrics {
    double max_abs_err;
    double rel_l2_err;             // ||h − ĥ||₂ / ||h||₂
    size_t length;

    void print(const std::string& label) const;
};

CoeffMetrics compare_coeffs(const std::vector<real_t>& h1,
                            const std::vector<real_t>& h2);

// ═══════════════════════════════════════════════════════════════
//  API: проектирование фильтра методом частотной выборки
// ═══════════════════════════════════════════════════════════════

//  Рассчитать коэффициенты h[n] по спецификации.
//  Выборки в полосе перехода задаются вектором transition_values
//  (от нижнего к верхнему индексу переходной полосы).
//  Если transition_values пуст — резкий переход (0/1).
DirectFIR design_freq_sampling(const FilterSpec& spec,
                               const std::vector<real_t>& transition_values = {});

// ═══════════════════════════════════════════════════════════════
//  API: каскадная декомпозиция
// ═══════════════════════════════════════════════════════════════

//  Базовая каскадная декомпозиция через последовательное выделение
//  известных нулей на единичной окружности.
//  Сохранена как компактная baseline-реализация и для сравнения.
//  Для текущей exact-first ветки и больших фильтров предпочтительнее
//  `decompose_exact_fs*`.
CascadeDecomposition decompose(const DirectFIR& fir,
                               bool interleave_order = true);

//  Current exact-first decomposition family.
//  These entry points are the main development path for large
//  frequency-sampling FIR filters.

//  Exact stage-1 extraction.
//  Берёт только структурно известные точные сомножители по H[k]
//  и оставляет сокращённый палиндромический остаток в `dec.remainder`.
CascadeDecomposition decompose_exact_fs(const DirectFIR& fir,
                                        bool interleave_order = true);

//  Совместимый псевдоним полного exact-first пути. Сохранён для старого API;
//  отдельного structured-алгоритма больше нет.
CascadeDecomposition decompose_exact_fs_structured(const DirectFIR& fir,
                                                   bool interleave_order = true);

//  Full exact-first path.
//  Строит остаток без последовательной дефляции, одновременно уточняет весь
//  набор его корней и адаптивно объединяет сомножители в устойчивые блоки.
CascadeDecomposition decompose_exact_fs_full(const DirectFIR& fir,
                                             bool interleave_order = true);

//  Пересобрать коэффициенты из каскада (для верификации):
//  перемножить все звенья и остаток обратно в один полином.
std::vector<real_t> recompose(const CascadeDecomposition& dec,
                              unsigned original_length);

// ═══════════════════════════════════════════════════════════════
//  API: фильтрация (поотсчётная)
// ═══════════════════════════════════════════════════════════════

//  Прямая свёртка — собственная реализация
std::vector<sample_t> filter_direct(const DirectFIR& fir,
                                    const std::vector<sample_t>& x);

//  Каскадная фильтрация — собственная реализация
std::vector<sample_t> filter_cascade(const CascadeDecomposition& dec,
                                     const std::vector<sample_t>& x);

// Double-варианты нужны для проверки самого каскада без маскировки
// ошибки финальным округлением каждого выхода в sample_t=float.
std::vector<double> filter_direct_double(const DirectFIR& fir,
                                         const std::vector<double>& x);
std::vector<double> filter_cascade_double(const CascadeDecomposition& dec,
                                          const std::vector<double>& x);

// ═══════════════════════════════════════════════════════════════
//  API: генерация сигналов
// ═══════════════════════════════════════════════════════════════

//  Белый шум, равномерное распределение [−1, +1]
std::vector<sample_t> generate_white_noise(size_t length, uint32_t seed = 42);

//  Сумма синусоид (для визуальной проверки АЧХ)
std::vector<sample_t> generate_multitone(size_t length,
                                         real_t fs,
                                         const std::vector<real_t>& freqs,
                                         real_t amplitude = 0.5);

//  Единичный импульс (для проверки импульсной характеристики)
std::vector<sample_t> generate_impulse(size_t length);

// ═══════════════════════════════════════════════════════════════
//  Вспомогательные функции
// ═══════════════════════════════════════════════════════════════

//  Полиномиальное умножение (свёртка двух последовательностей)
std::vector<real_t> poly_multiply(const std::vector<real_t>& a,
                                  const std::vector<real_t>& b);

//  Полиномиальное деление: a(z) / b(z) → quotient, remainder.
//  Стандартный алгоритм «в столбик» (synthetic division).
//  Предусловие: b[0] != 0 (старший коэффициент делителя ненулевой).
//  Возвращает пару (quotient, remainder).
//  Длина quotient = len(a) - len(b) + 1, длина remainder = len(b) - 1.
std::pair<std::vector<real_t>, std::vector<real_t>>
poly_divide(const std::vector<real_t>& a,
            const std::vector<real_t>& b);

//  Пересчёт остатка из оригинального h[n].
//
//  После декомпозиции (decompose) остаток содержит накопленную
//  ошибку от P последовательных делений.  Эта функция пересчитывает
//  остаток ОДНИМ делением: h_original / (произведение всех каскадов).
//
//  Произведение каскадов строится из ТОЧНЫХ γ_k (из спецификации),
//  поэтому ошибка определяется только одним делением длинного
//  полинома, а не P последовательными.
//
//  Вызывается после decompose(); модифицирует dec.remainder in-place.
void recompute_remainder(CascadeDecomposition& dec,
                         const std::vector<real_t>& h_original);

//  Вычислить АЧХ в nfft точках
std::vector<real_t> compute_magnitude_response(const std::vector<real_t>& h,
                                               unsigned nfft = 4096);

//  Напечатать информацию о декомпозиции
void print_decomposition(const CascadeDecomposition& dec,
                         const FilterSpec& spec);

#endif // CASCADE_FIR_H
