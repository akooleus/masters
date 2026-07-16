#include "cascade_fir.h"

#include <cerrno>
#include <charconv>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace {

constexpr char ARTIFACT_MAGIC[] = "CASCADE_FIR_ARTIFACT";
constexpr std::size_t MAX_ARTIFACT_BYTES = 64u * 1024u * 1024u;
constexpr std::size_t MAX_CONTAINER_ENTRIES = 1024u * 1024u;
constexpr std::uint64_t FNV_OFFSET = 14695981039346656037ULL;
constexpr std::uint64_t FNV_PRIME = 1099511628211ULL;

std::uint64_t real_bits(real_t value)
{
    static_assert(sizeof(real_t) == sizeof(std::uint64_t),
                  "artifact format requires a 64-bit real_t");
    static_assert(std::numeric_limits<real_t>::is_iec559,
                  "artifact format requires IEEE-754 real_t");
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

real_t bits_real(std::uint64_t bits)
{
    real_t value = 0.0;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

void hash_byte(std::uint64_t& hash, unsigned char value)
{
    hash ^= value;
    hash *= FNV_PRIME;
}

void hash_u64(std::uint64_t& hash, std::uint64_t value)
{
    for (unsigned shift = 0; shift < 64u; shift += 8u) {
        hash_byte(hash, static_cast<unsigned char>((value >> shift) & 0xffu));
    }
}

std::uint64_t checksum(const std::string& data)
{
    std::uint64_t hash = FNV_OFFSET;
    for (unsigned char byte : data) {
        hash_byte(hash, byte);
    }
    return hash;
}

bool valid_status(CascadeBuildStatus status)
{
    switch (status) {
    case CascadeBuildStatus::Unspecified:
    case CascadeBuildStatus::AnalyticalReduction:
    case CascadeBuildStatus::ShortBlockCascade:
    case CascadeBuildStatus::HighPrecisionShortBlockCascade:
    case CascadeBuildStatus::FullOrderFactorBlock:
    case CascadeBuildStatus::DirectFormFallback:
    case CascadeBuildStatus::Failed:
        return true;
    }
    return false;
}

bool executable_status(CascadeBuildStatus status)
{
    switch (status) {
    case CascadeBuildStatus::ShortBlockCascade:
    case CascadeBuildStatus::HighPrecisionShortBlockCascade:
    case CascadeBuildStatus::FullOrderFactorBlock:
    case CascadeBuildStatus::DirectFormFallback:
        return true;
    case CascadeBuildStatus::Unspecified:
    case CascadeBuildStatus::AnalyticalReduction:
    case CascadeBuildStatus::Failed:
        return false;
    }
    return false;
}

unsigned precision_digits(CascadeRuntimePrecision precision)
{
    switch (precision) {
    case CascadeRuntimePrecision::Native:
        return 0u;
    case CascadeRuntimePrecision::DoubleDouble:
        return 31u;
    case CascadeRuntimePrecision::Extended34:
        return 34u;
    case CascadeRuntimePrecision::Extended50:
        return 50u;
    }
    throw std::invalid_argument("invalid cascade runtime precision");
}

void require(bool condition, const char* message)
{
    if (!condition) {
        throw std::invalid_argument(message);
    }
}

void require_finite(real_t value, const char* message)
{
    require(std::isfinite(value), message);
}

void validate_source_spec(const CascadeArtifactMetadata& metadata)
{
    const FilterSpec& spec = metadata.source_spec;
    require(metadata.format_version == CASCADE_ARTIFACT_FORMAT_VERSION,
            "unsupported cascade artifact version");
    require(metadata.source_coefficient_count > 0u,
            "artifact source coefficient count is zero");
    require(spec.N == metadata.source_coefficient_count,
            "artifact source length does not match its specification");
    require_finite(spec.fs, "artifact sampling frequency is not finite");
    require_finite(spec.f_pass, "artifact passband edge is not finite");
    require_finite(spec.f_stop, "artifact stopband edge is not finite");
    require(spec.fs > 0.0, "artifact sampling frequency is not positive");
    require(spec.f_pass >= 0.0 && spec.f_pass <= spec.f_stop,
            "artifact passband and stopband edges are inconsistent");
    require(spec.f_stop <= 0.5 * spec.fs,
            "artifact stopband edge exceeds Nyquist frequency");
}

std::size_t checked_add(std::size_t lhs, std::size_t rhs)
{
    if (rhs > std::numeric_limits<std::size_t>::max() - lhs) {
        throw std::invalid_argument("cascade degree overflow");
    }
    return lhs + rhs;
}

void validate_decomposition(const CascadeDecomposition& dec,
                            std::uint32_t source_length)
{
    require(valid_status(dec.diagnostics.status),
            "invalid cascade build status");
    require(executable_status(dec.diagnostics.status),
            "artifact does not contain a verified executable cascade");
    require(dec.diagnostics.runtime_verified,
            "artifact runtime was not verified");
    require_finite(dec.gain, "cascade gain is not finite");
    require_finite(dec.runtime_gain, "cascade runtime gain is not finite");
    require_finite(dec.diagnostics.runtime_impulse_error,
                   "cascade runtime impulse error is not finite");
    require_finite(dec.diagnostics.runtime_peak_internal,
                   "cascade runtime peak is not finite");
    require(!std::isnan(dec.diagnostics.complement_identity_error),
            "cascade complement error is NaN");
    require(!std::isnan(dec.diagnostics.root_backward_error),
            "cascade root backward error is NaN");
    require(!std::isnan(dec.diagnostics.root_rebuild_error),
            "cascade root rebuild error is NaN");
    if (dec.diagnostics.complement_verified) {
        require_finite(dec.diagnostics.complement_identity_error,
                       "verified complement error is not finite");
    }
    if (dec.diagnostics.roots_verified) {
        require_finite(dec.diagnostics.root_backward_error,
                       "verified root backward error is not finite");
        require_finite(dec.diagnostics.root_rebuild_error,
                       "verified root rebuild error is not finite");
    }

    const unsigned digits = precision_digits(dec.runtime_precision);
    require(dec.diagnostics.runtime_decimal_digits == digits,
            "runtime precision and diagnostics disagree");

    const std::size_t section_count = checked_add(
        checked_add(dec.first_order.size(), dec.biquads.size()),
        checked_add(dec.quartics.size(), dec.blocks.size()));
    require(section_count <= MAX_CONTAINER_ENTRIES,
            "cascade contains too many sections");
    require(dec.remainder.size() <= MAX_CONTAINER_ENTRIES,
            "cascade remainder is too large");
    require(!dec.remainder.empty(), "cascade remainder is empty");

    std::size_t degree = dec.remainder.size() - 1u;
    for (const FirstOrder& section : dec.first_order) {
        require(section.sign == -1 || section.sign == 1,
                "invalid first-order section sign");
        degree = checked_add(degree, 1u);
    }
    for (const Biquad& section : dec.biquads) {
        require_finite(section.gamma, "biquad coefficient is not finite");
        degree = checked_add(degree, 2u);
    }
    for (const Quartic& section : dec.quartics) {
        require_finite(section.alpha, "quartic alpha is not finite");
        require_finite(section.beta, "quartic beta is not finite");
        degree = checked_add(degree, 4u);
    }
    std::size_t coefficient_count = dec.remainder.size();
    for (const CascadeBlock& block : dec.blocks) {
        require(!block.coefficients.empty(), "cascade block is empty");
        coefficient_count = checked_add(
            coefficient_count, block.coefficients.size());
        require(coefficient_count <= MAX_CONTAINER_ENTRIES,
                "cascade contains too many coefficients");
        degree = checked_add(degree, block.coefficients.size() - 1u);
        for (real_t coefficient : block.coefficients) {
            require_finite(coefficient,
                           "cascade block coefficient is not finite");
        }
    }
    for (real_t coefficient : dec.remainder) {
        require_finite(coefficient,
                       "cascade remainder coefficient is not finite");
    }
    require(degree + 1u == source_length,
            "cascade degree does not match source filter length");

    if (dec.execution_order.empty()) {
        require(dec.blocks.empty(),
                "block cascade has no explicit execution order");
    } else {
        require(dec.execution_order.size() == section_count,
                "execution order does not cover every cascade section");
        std::vector<bool> seen_first(dec.first_order.size(), false);
        std::vector<bool> seen_biquad(dec.biquads.size(), false);
        std::vector<bool> seen_quartic(dec.quartics.size(), false);
        std::vector<bool> seen_block(dec.blocks.size(), false);
        for (const CascadeSectionPlacement& placement : dec.execution_order) {
            require_finite(placement.scale,
                           "cascade execution scale is not finite");
            require(placement.scale > 0.0,
                    "cascade execution scale is not positive");
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
            require(placement.index < seen->size(),
                    "cascade execution index is out of range");
            require(!(*seen)[placement.index],
                    "cascade execution order contains a duplicate section");
            (*seen)[placement.index] = true;
        }
    }

    if (dec.runtime_precision != CascadeRuntimePrecision::Native) {
        require(dec.first_order.empty() && dec.biquads.empty()
                    && dec.quartics.empty(),
                "extended runtime supports block sections only");
        require(!dec.execution_order.empty(),
                "extended runtime has no execution order");
        for (const CascadeSectionPlacement& placement : dec.execution_order) {
            require(placement.kind == CascadeSectionKind::Block,
                    "extended runtime contains a non-block section");
        }
    }
}

void write_hex(std::ostream& out, std::uint64_t value)
{
    const std::ios::fmtflags previous_flags = out.flags();
    const char previous_fill = out.fill();
    out << std::hex << std::nouppercase << std::setfill('0') << std::setw(16)
        << value;
    out.flags(previous_flags);
    out.fill(previous_fill);
}

void write_real(std::ostream& out, real_t value)
{
    write_hex(out, real_bits(value));
}

void write_real_vector(std::ostream& out,
                       const char* label,
                       const std::vector<real_t>& values)
{
    out << label << ' ' << values.size();
    for (real_t value : values) {
        out << ' ';
        write_real(out, value);
    }
    out << '\n';
}

std::string encode_payload(const CascadeArtifact& artifact)
{
    const CascadeArtifactMetadata& metadata = artifact.metadata;
    const CascadeDecomposition& dec = artifact.decomposition;
    std::ostringstream out;
    out.imbue(std::locale::classic());

    out << "source " << metadata.source_spec.N << ' ';
    write_real(out, metadata.source_spec.fs);
    out << ' ';
    write_real(out, metadata.source_spec.f_pass);
    out << ' ';
    write_real(out, metadata.source_spec.f_stop);
    out << ' ' << metadata.source_coefficient_count << ' ';
    write_hex(out, metadata.source_fingerprint);
    out << '\n';

    out << "precision "
        << static_cast<unsigned>(dec.runtime_precision) << '\n';
    out << "first_order " << dec.first_order.size();
    for (const FirstOrder& section : dec.first_order) {
        out << ' ' << section.sign;
    }
    out << '\n';

    out << "biquads " << dec.biquads.size();
    for (const Biquad& section : dec.biquads) {
        out << ' ';
        write_real(out, section.gamma);
        out << ' ' << section.k_index;
    }
    out << '\n';

    out << "quartics " << dec.quartics.size();
    for (const Quartic& section : dec.quartics) {
        out << ' ';
        write_real(out, section.alpha);
        out << ' ';
        write_real(out, section.beta);
    }
    out << '\n';

    out << "blocks " << dec.blocks.size() << '\n';
    for (const CascadeBlock& block : dec.blocks) {
        write_real_vector(out, "block", block.coefficients);
    }
    write_real_vector(out, "remainder", dec.remainder);
    out << "gain ";
    write_real(out, dec.gain);
    out << ' ';
    write_real(out, dec.runtime_gain);
    out << '\n';

    out << "execution_order " << dec.execution_order.size() << '\n';
    for (const CascadeSectionPlacement& placement : dec.execution_order) {
        out << "placement " << static_cast<unsigned>(placement.kind)
            << ' ' << placement.index << ' ';
        write_real(out, placement.scale);
        out << '\n';
    }

    const CascadeDiagnostics& diagnostics = dec.diagnostics;
    out << "diagnostics " << static_cast<unsigned>(diagnostics.status)
        << ' ' << static_cast<unsigned>(diagnostics.complement_verified) << ' ';
    write_real(out, diagnostics.complement_identity_error);
    out << ' ' << static_cast<unsigned>(diagnostics.roots_verified)
        << ' ' << diagnostics.root_iterations << ' ';
    write_real(out, diagnostics.root_backward_error);
    out << ' ';
    write_real(out, diagnostics.root_rebuild_error);
    out << ' ' << static_cast<unsigned>(diagnostics.runtime_verified) << ' ';
    write_real(out, diagnostics.runtime_impulse_error);
    out << ' ';
    write_real(out, diagnostics.runtime_peak_internal);
    out << ' ' << diagnostics.selected_max_block_order
        << ' ' << diagnostics.runtime_decimal_digits << '\n';
    out << "end\n";
    return out.str();
}

class TokenReader {
public:
    explicit TokenReader(const std::string& payload) : input_(payload)
    {
        input_.imbue(std::locale::classic());
    }

    void expect(const char* expected)
    {
        const std::string actual = token(expected);
        if (actual != expected) {
            throw std::runtime_error(
                "invalid cascade artifact: expected " + std::string(expected));
        }
    }

    std::uint64_t decimal(const char* field)
    {
        const std::string value = token(field);
        std::uint64_t parsed = 0;
        const auto result = std::from_chars(
            value.data(), value.data() + value.size(), parsed, 10);
        if (result.ec != std::errc()
            || result.ptr != value.data() + value.size()) {
            throw std::runtime_error(
                "invalid cascade artifact decimal field: " + std::string(field));
        }
        return parsed;
    }

    std::size_t count(const char* field)
    {
        const std::uint64_t value = decimal(field);
        if (value > MAX_CONTAINER_ENTRIES
            || value > std::numeric_limits<std::size_t>::max()) {
            throw std::runtime_error(
                "cascade artifact count is too large: " + std::string(field));
        }
        return static_cast<std::size_t>(value);
    }

    std::uint64_t hex(const char* field)
    {
        const std::string value = token(field);
        if (value.size() != 16u) {
            throw std::runtime_error(
                "invalid cascade artifact hexadecimal field: "
                + std::string(field));
        }
        std::uint64_t parsed = 0;
        const auto result = std::from_chars(
            value.data(), value.data() + value.size(), parsed, 16);
        if (result.ec != std::errc()
            || result.ptr != value.data() + value.size()) {
            throw std::runtime_error(
                "invalid cascade artifact hexadecimal field: "
                + std::string(field));
        }
        return parsed;
    }

    real_t real(const char* field)
    {
        return bits_real(hex(field));
    }

    bool boolean(const char* field)
    {
        const std::uint64_t value = decimal(field);
        if (value > 1u) {
            throw std::runtime_error(
                "invalid cascade artifact boolean field: "
                + std::string(field));
        }
        return value != 0u;
    }

    int sign(const char* field)
    {
        const std::string value = token(field);
        if (value == "-1") {
            return -1;
        }
        if (value == "1") {
            return 1;
        }
        throw std::runtime_error(
            "invalid cascade artifact sign field: " + std::string(field));
    }

    void finish()
    {
        expect("end");
        std::string extra;
        if (input_ >> extra) {
            throw std::runtime_error("trailing data in cascade artifact payload");
        }
    }

private:
    std::string token(const char* field)
    {
        std::string value;
        if (!(input_ >> value)) {
            throw std::runtime_error(
                "truncated cascade artifact field: " + std::string(field));
        }
        return value;
    }

    std::istringstream input_;
};

template <class Integer>
Integer narrow_integer(std::uint64_t value, const char* field)
{
    if (value > static_cast<std::uint64_t>(
                    std::numeric_limits<Integer>::max())) {
        throw std::runtime_error(
            "cascade artifact integer is out of range: " + std::string(field));
    }
    return static_cast<Integer>(value);
}

std::vector<real_t> read_real_vector(TokenReader& reader,
                                     const char* label)
{
    reader.expect(label);
    const std::size_t count = reader.count(label);
    std::vector<real_t> values(count);
    for (real_t& value : values) {
        value = reader.real(label);
    }
    return values;
}

CascadeArtifact decode_payload(const std::string& payload,
                               std::uint32_t format_version)
{
    TokenReader reader(payload);
    CascadeArtifact artifact;
    artifact.metadata.format_version = format_version;

    reader.expect("source");
    artifact.metadata.source_spec.N =
        narrow_integer<unsigned>(reader.decimal("source N"), "source N");
    artifact.metadata.source_spec.fs = reader.real("source fs");
    artifact.metadata.source_spec.f_pass = reader.real("source f_pass");
    artifact.metadata.source_spec.f_stop = reader.real("source f_stop");
    artifact.metadata.source_coefficient_count = narrow_integer<std::uint32_t>(
        reader.decimal("source coefficient count"),
        "source coefficient count");
    artifact.metadata.source_fingerprint = reader.hex("source fingerprint");

    CascadeDecomposition& dec = artifact.decomposition;
    reader.expect("precision");
    const unsigned precision = narrow_integer<unsigned>(
        reader.decimal("runtime precision"), "runtime precision");
    switch (precision) {
    case 0u:
        dec.runtime_precision = CascadeRuntimePrecision::Native;
        break;
    case 31u:
        dec.runtime_precision = CascadeRuntimePrecision::DoubleDouble;
        break;
    case 34u:
        dec.runtime_precision = CascadeRuntimePrecision::Extended34;
        break;
    case 50u:
        dec.runtime_precision = CascadeRuntimePrecision::Extended50;
        break;
    default:
        throw std::runtime_error("unsupported cascade artifact precision");
    }

    reader.expect("first_order");
    dec.first_order.resize(reader.count("first-order count"));
    for (FirstOrder& section : dec.first_order) {
        section.sign = reader.sign("first-order sign");
    }

    reader.expect("biquads");
    dec.biquads.resize(reader.count("biquad count"));
    for (Biquad& section : dec.biquads) {
        section.gamma = reader.real("biquad gamma");
        section.k_index = narrow_integer<unsigned>(
            reader.decimal("biquad k index"), "biquad k index");
    }

    reader.expect("quartics");
    dec.quartics.resize(reader.count("quartic count"));
    for (Quartic& section : dec.quartics) {
        section.alpha = reader.real("quartic alpha");
        section.beta = reader.real("quartic beta");
    }

    reader.expect("blocks");
    dec.blocks.resize(reader.count("block count"));
    for (CascadeBlock& block : dec.blocks) {
        block.coefficients = read_real_vector(reader, "block");
    }
    dec.remainder = read_real_vector(reader, "remainder");

    reader.expect("gain");
    dec.gain = reader.real("gain");
    dec.runtime_gain = reader.real("runtime gain");

    reader.expect("execution_order");
    dec.execution_order.resize(reader.count("execution-order count"));
    for (CascadeSectionPlacement& placement : dec.execution_order) {
        reader.expect("placement");
        const unsigned kind = narrow_integer<unsigned>(
            reader.decimal("section kind"), "section kind");
        if (kind > static_cast<unsigned>(CascadeSectionKind::Block)) {
            throw std::runtime_error("invalid cascade artifact section kind");
        }
        placement.kind = static_cast<CascadeSectionKind>(kind);
        placement.index = reader.count("section index");
        placement.scale = reader.real("section scale");
    }

    reader.expect("diagnostics");
    CascadeDiagnostics& diagnostics = dec.diagnostics;
    const unsigned status = narrow_integer<unsigned>(
        reader.decimal("build status"), "build status");
    if (status > static_cast<unsigned>(CascadeBuildStatus::Failed)) {
        throw std::runtime_error("invalid cascade artifact build status");
    }
    diagnostics.status = static_cast<CascadeBuildStatus>(status);
    diagnostics.complement_verified = reader.boolean("complement verified");
    diagnostics.complement_identity_error = reader.real("complement error");
    diagnostics.roots_verified = reader.boolean("roots verified");
    diagnostics.root_iterations = narrow_integer<unsigned>(
        reader.decimal("root iterations"), "root iterations");
    diagnostics.root_backward_error = reader.real("root backward error");
    diagnostics.root_rebuild_error = reader.real("root rebuild error");
    diagnostics.runtime_verified = reader.boolean("runtime verified");
    diagnostics.runtime_impulse_error = reader.real("runtime impulse error");
    diagnostics.runtime_peak_internal = reader.real("runtime peak");
    diagnostics.selected_max_block_order = narrow_integer<unsigned>(
        reader.decimal("selected block order"), "selected block order");
    diagnostics.runtime_decimal_digits = narrow_integer<unsigned>(
        reader.decimal("runtime decimal digits"), "runtime decimal digits");
    reader.finish();
    return artifact;
}

std::string read_file(const std::string& path)
{
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error("cannot open cascade artifact: " + path);
    }
    const std::streamoff length = input.tellg();
    if (length < 0
        || static_cast<std::uint64_t>(length) > MAX_ARTIFACT_BYTES) {
        throw std::runtime_error("cascade artifact has an invalid size");
    }
    std::string data(static_cast<std::size_t>(length), '\0');
    input.seekg(0);
    if (!data.empty()) {
        input.read(data.data(), length);
    }
    if (!input) {
        throw std::runtime_error("cannot read cascade artifact: " + path);
    }
    return data;
}

} // namespace

std::uint64_t cascade_source_fingerprint(const DirectFIR& fir)
{
    if (fir.h.empty() || fir.h.size() > std::numeric_limits<std::uint32_t>::max()
        || fir.spec.N != fir.h.size()) {
        throw std::invalid_argument("invalid source FIR for cascade artifact");
    }
    if (!std::isfinite(fir.spec.fs) || !std::isfinite(fir.spec.f_pass)
        || !std::isfinite(fir.spec.f_stop)) {
        throw std::invalid_argument("source FIR specification is not finite");
    }
    if (!fir.frequency_samples.empty()
        && fir.frequency_samples.size() != fir.h.size()) {
        throw std::invalid_argument(
            "source FIR frequency sample count does not match its length");
    }

    std::uint64_t hash = FNV_OFFSET;
    hash_u64(hash, fir.spec.N);
    hash_u64(hash, real_bits(fir.spec.fs));
    hash_u64(hash, real_bits(fir.spec.f_pass));
    hash_u64(hash, real_bits(fir.spec.f_stop));
    hash_u64(hash, fir.h.size());
    for (real_t value : fir.h) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("source FIR coefficient is not finite");
        }
        hash_u64(hash, real_bits(value));
    }
    hash_u64(hash, fir.frequency_samples.size());
    for (real_t value : fir.frequency_samples) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument(
                "source FIR frequency sample is not finite");
        }
        hash_u64(hash, real_bits(value));
    }
    return hash;
}

CascadeArtifact make_cascade_artifact(
    const DirectFIR& fir,
    const CascadeDecomposition& decomposition)
{
    CascadeArtifact artifact;
    artifact.metadata.source_spec = fir.spec;
    artifact.metadata.source_coefficient_count =
        static_cast<std::uint32_t>(fir.h.size());
    artifact.metadata.source_fingerprint = cascade_source_fingerprint(fir);
    artifact.decomposition = decomposition;
    validate_cascade_artifact(artifact);
    return artifact;
}

bool cascade_artifact_matches(const CascadeArtifact& artifact,
                              const DirectFIR& fir)
{
    try {
        return artifact.metadata.format_version
                   == CASCADE_ARTIFACT_FORMAT_VERSION
            && artifact.metadata.source_spec.N == fir.spec.N
            && real_bits(artifact.metadata.source_spec.fs)
                   == real_bits(fir.spec.fs)
            && real_bits(artifact.metadata.source_spec.f_pass)
                   == real_bits(fir.spec.f_pass)
            && real_bits(artifact.metadata.source_spec.f_stop)
                   == real_bits(fir.spec.f_stop)
            && artifact.metadata.source_coefficient_count == fir.h.size()
            && artifact.metadata.source_fingerprint
                   == cascade_source_fingerprint(fir);
    } catch (const std::invalid_argument&) {
        return false;
    }
}

void validate_cascade_artifact(const CascadeArtifact& artifact)
{
    validate_source_spec(artifact.metadata);
    validate_decomposition(
        artifact.decomposition, artifact.metadata.source_coefficient_count);
}

void save_cascade_artifact(const std::string& path,
                           const CascadeArtifact& artifact)
{
    if (path.empty()) {
        throw std::invalid_argument("cascade artifact path is empty");
    }
    validate_cascade_artifact(artifact);
    const std::string payload = encode_payload(artifact);
    if (payload.size() > MAX_ARTIFACT_BYTES) {
        throw std::invalid_argument("cascade artifact payload is too large");
    }

    std::ostringstream header;
    header.imbue(std::locale::classic());
    header << ARTIFACT_MAGIC << ' ' << CASCADE_ARTIFACT_FORMAT_VERSION
           << ' ' << payload.size() << ' ';
    write_hex(header, checksum(payload));
    header << '\n';

    const std::string temporary_path = path + ".tmp";
    {
        std::ofstream output(
            temporary_path, std::ios::binary | std::ios::trunc);
        if (!output) {
            throw std::runtime_error(
                "cannot create temporary cascade artifact: " + temporary_path);
        }
        output << header.str();
        output.write(payload.data(), static_cast<std::streamsize>(payload.size()));
        output.flush();
        if (!output) {
            output.close();
            std::remove(temporary_path.c_str());
            throw std::runtime_error(
                "cannot write cascade artifact: " + temporary_path);
        }
    }

    if (std::rename(temporary_path.c_str(), path.c_str()) != 0) {
        const int error = errno;
        std::remove(temporary_path.c_str());
        throw std::runtime_error(
            "cannot publish cascade artifact: "
            + std::error_code(error, std::generic_category()).message());
    }
}

CascadeArtifact load_cascade_artifact(const std::string& path)
{
    const std::string data = read_file(path);
    const std::size_t newline = data.find('\n');
    if (newline == std::string::npos) {
        throw std::runtime_error("cascade artifact header is truncated");
    }

    std::istringstream header(data.substr(0, newline));
    header.imbue(std::locale::classic());
    std::string magic;
    std::string version_token;
    std::string size_token;
    std::string checksum_token;
    std::string extra;
    if (!(header >> magic >> version_token >> size_token >> checksum_token)
        || (header >> extra) || magic != ARTIFACT_MAGIC) {
        throw std::runtime_error("invalid cascade artifact header");
    }

    auto parse_header_decimal = [](const std::string& token,
                                   const char* field) -> std::uint64_t {
        std::uint64_t value = 0;
        const auto result = std::from_chars(
            token.data(), token.data() + token.size(), value, 10);
        if (result.ec != std::errc()
            || result.ptr != token.data() + token.size()) {
            throw std::runtime_error(
                "invalid cascade artifact header field: "
                + std::string(field));
        }
        return value;
    };
    auto parse_header_hex = [](const std::string& token) -> std::uint64_t {
        if (token.size() != 16u) {
            throw std::runtime_error("invalid cascade artifact checksum");
        }
        std::uint64_t value = 0;
        const auto result = std::from_chars(
            token.data(), token.data() + token.size(), value, 16);
        if (result.ec != std::errc()
            || result.ptr != token.data() + token.size()) {
            throw std::runtime_error("invalid cascade artifact checksum");
        }
        return value;
    };

    const std::uint64_t version = parse_header_decimal(
        version_token, "version");
    if (version != CASCADE_ARTIFACT_FORMAT_VERSION) {
        throw std::runtime_error("unsupported cascade artifact version");
    }
    const std::uint64_t expected_size = parse_header_decimal(
        size_token, "payload size");
    const std::size_t actual_size = data.size() - newline - 1u;
    if (expected_size != actual_size || actual_size > MAX_ARTIFACT_BYTES) {
        throw std::runtime_error("cascade artifact payload size mismatch");
    }

    const std::string payload = data.substr(newline + 1u);
    if (parse_header_hex(checksum_token) != checksum(payload)) {
        throw std::runtime_error("cascade artifact checksum mismatch");
    }

    CascadeArtifact artifact = decode_payload(
        payload, static_cast<std::uint32_t>(version));
    try {
        validate_cascade_artifact(artifact);
    } catch (const std::invalid_argument& error) {
        throw std::runtime_error(
            "invalid cascade artifact payload: " + std::string(error.what()));
    }
    return artifact;
}
