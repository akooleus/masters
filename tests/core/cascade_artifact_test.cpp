#include "test_utils.h"

#include <cstdio>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

template <class Exception, class Function>
bool throws(Function&& function)
{
    try {
        function();
    } catch (const Exception&) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}

std::string read_bytes(const std::string& path)
{
    std::ifstream input(path, std::ios::binary);
    return std::string(
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>());
}

void write_bytes(const std::string& path, const std::string& bytes)
{
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

CascadeDecomposition direct_block(const DirectFIR& fir)
{
    CascadeDecomposition dec;
    dec.blocks.push_back(CascadeBlock{fir.h});
    dec.remainder = {1.0};
    dec.execution_order.push_back(
        {CascadeSectionKind::Block, 0u, 1.0});
    dec.gain = 1.0;
    dec.runtime_gain = 1.0;
    dec.diagnostics.status = CascadeBuildStatus::DirectFormFallback;
    dec.diagnostics.runtime_verified = true;
    dec.diagnostics.runtime_impulse_error = 0.0;
    dec.diagnostics.runtime_peak_internal = 1.0;
    dec.diagnostics.selected_max_block_order = fir.order();
    return dec;
}

} // namespace

int main()
{
    const std::string path = "/tmp/cascade_artifact_test.cfir";
    const std::string corrupt_path = path + ".corrupt";
    const std::string truncated_path = path + ".truncated";
    const std::string version_path = path + ".version";
    const std::string extended_path = path + ".extended";
    const std::string double_double_path = path + ".double-double";
    std::remove(path.c_str());
    std::remove(corrupt_path.c_str());
    std::remove(truncated_path.c_str());
    std::remove(version_path.c_str());
    std::remove(extended_path.c_str());
    std::remove(double_double_path.c_str());

    bool ok = true;
    DirectFIR empty;
    ok &= expect(empty.order() == 0u,
                 "empty DirectFIR order should not underflow");
    ok &= expect(throws<std::logic_error>([] {
                     DirectFilterState state;
                     (void)state.push(0.0f);
                 }),
                 "DirectFilterState::push should reject an uninitialized state");
    ok &= expect(throws<std::logic_error>([] {
                     DoubleFilterState state;
                     (void)state.push(0.0);
                 }),
                 "DoubleFilterState::push should reject an uninitialized state");
    ok &= expect(throws<std::logic_error>([] {
                     CascadeFilterState state;
                     (void)state.push_double(0.0);
                 }),
                 "CascadeFilterState::push should reject an uninitialized state");

    const DirectFIR fir = smoke_fir();
    const CascadeDecomposition dec = direct_block(fir);
    const CascadeArtifact artifact = make_cascade_artifact(fir, dec);
    save_cascade_artifact(path, artifact);
    const CascadeArtifact loaded = load_cascade_artifact(path);
    ok &= expect(cascade_artifact_matches(loaded, fir),
                 "loaded artifact should match its source FIR");
    ok &= expect(loaded.decomposition.blocks.size() == 1u
                     && loaded.decomposition.blocks.front().coefficients
                            == fir.h,
                 "artifact should preserve coefficient bits");
    ok &= expect(loaded.decomposition.runtime_precision
                     == CascadeRuntimePrecision::Native,
                 "artifact should preserve runtime precision");

    DirectFIR changed = fir;
    changed.h.front() += 1e-12;
    ok &= expect(!cascade_artifact_matches(loaded, changed),
                 "artifact fingerprint should reject changed coefficients");

    const std::vector<double> probe(64u, 0.25);
    ok &= expect(filter_cascade_double(dec, probe)
                     == filter_cascade_double(loaded.decomposition, probe),
                 "loaded artifact should reproduce runtime output exactly");

    CascadeDecomposition diagnostics_only = dec;
    diagnostics_only.diagnostics.runtime_decimal_digits = 50u;
    ok &= expect(!throws<std::invalid_argument>([&] {
                     CascadeFilterState state;
                     state.init(diagnostics_only);
                 }),
                 "diagnostics must not select the runtime backend");

    CascadeDecomposition extended = dec;
    extended.runtime_precision = CascadeRuntimePrecision::Extended50;
    extended.diagnostics.runtime_decimal_digits = 50u;
    ok &= expect(!throws<std::invalid_argument>([&] {
                     CascadeFilterState state;
                     state.init(extended);
                     (void)state.push_double(0.25);
                 }),
                 "explicit Extended50 backend should initialize");
    const CascadeArtifact extended_artifact =
        make_cascade_artifact(fir, extended);
    save_cascade_artifact(extended_path, extended_artifact);
    ok &= expect(load_cascade_artifact(extended_path)
                         .decomposition.runtime_precision
                     == CascadeRuntimePrecision::Extended50,
                 "artifact should preserve an extended runtime backend");

    CascadeDecomposition double_double = dec;
    double_double.runtime_precision =
        CascadeRuntimePrecision::DoubleDouble;
    double_double.diagnostics.runtime_decimal_digits = 31u;
    const CascadeArtifact double_double_artifact =
        make_cascade_artifact(fir, double_double);
    save_cascade_artifact(double_double_path, double_double_artifact);
    const CascadeArtifact loaded_double_double =
        load_cascade_artifact(double_double_path);
    ok &= expect(loaded_double_double.decomposition.runtime_precision
                     == CascadeRuntimePrecision::DoubleDouble,
                 "artifact should preserve double-double precision");
    CascadeFilterState scalar_double_double;
    scalar_double_double.init(
        loaded_double_double.decomposition,
        CascadeRuntimeOptions{false, CascadeRuntimeKernel::Scalar});
    ok &= expect(std::isfinite(scalar_double_double.push_double(0.25)),
                 "scalar double-double backend should be finite");
    CascadeFilterState automatic_double_double;
    automatic_double_double.init(loaded_double_double.decomposition);
    const CascadeRuntimeKernel automatic_kernel =
        cascade_runtime_kernel_available(CascadeRuntimeKernel::Avx2Fma)
        ? CascadeRuntimeKernel::Avx2Fma
        : CascadeRuntimeKernel::Scalar;
    ok &= expect(automatic_double_double.selected_kernel == automatic_kernel,
                 "double-double runtime dispatch mismatch");

    CascadeArtifact bad_index = artifact;
    bad_index.decomposition.execution_order.front().index = 1u;
    ok &= expect(throws<std::invalid_argument>([&] {
                     validate_cascade_artifact(bad_index);
                 }),
                 "artifact validation should reject an invalid section index");

    CascadeArtifact bad_precision = artifact;
    bad_precision.decomposition.diagnostics.runtime_decimal_digits = 34u;
    ok &= expect(throws<std::invalid_argument>([&] {
                     validate_cascade_artifact(bad_precision);
                 }),
                 "artifact validation should reject precision disagreement");

    const std::string bytes = read_bytes(path);
    std::string corrupt = bytes;
    const std::size_t block = corrupt.find("block ");
    if (block != std::string::npos && block + 8u < corrupt.size()) {
        corrupt[block + 8u] = (corrupt[block + 8u] == '0') ? '1' : '0';
    }
    write_bytes(corrupt_path, corrupt);
    ok &= expect(throws<std::runtime_error>([&] {
                     (void)load_cascade_artifact(corrupt_path);
                 }),
                 "artifact loader should reject a checksum mismatch");

    write_bytes(truncated_path, bytes.substr(0, bytes.size() - 1u));
    ok &= expect(throws<std::runtime_error>([&] {
                     (void)load_cascade_artifact(truncated_path);
                 }),
                 "artifact loader should reject a truncated payload");

    std::string wrong_version = bytes;
    const std::size_t version = wrong_version.find(" 1 ");
    if (version != std::string::npos) {
        wrong_version[version + 1u] = '2';
    }
    write_bytes(version_path, wrong_version);
    ok &= expect(throws<std::runtime_error>([&] {
                     (void)load_cascade_artifact(version_path);
                 }),
                 "artifact loader should reject an unsupported version");

    std::remove(path.c_str());
    std::remove(corrupt_path.c_str());
    std::remove(truncated_path.c_str());
    std::remove(version_path.c_str());
    std::remove(extended_path.c_str());
    std::remove(double_double_path.c_str());
    return ok ? 0 : 1;
}
