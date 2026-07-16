cmake_minimum_required(VERSION 3.16)
project(cascade_fir LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
include(CTest)

option(CASCADE_BUILD_EXACT_FS "Build exact-first diagnostics and tests" ON)

# -O2 for fair benchmarking, with debug info
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ── External dependencies ─────────────────────────────────────
if(CASCADE_BUILD_EXACT_FS)
    find_library(LAPACK_LIB lapack REQUIRED)
endif()

# ── Common interface and sources ─────────────────────────────
add_library(cascade_core INTERFACE)
target_include_directories(cascade_core INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

set(CASCADE_COMMON_SOURCES
    src/freq_sampling_design.cpp
    src/cascade_decompose.cpp
    src/cascade_filter.cpp
    src/direct_filter.cpp
    src/signal_generation.cpp
)

function(configure_cascade_target target)
    target_link_libraries(${target} PRIVATE
        cascade_core
        m
    )
endfunction()

function(link_extended_backend target)
    target_link_libraries(${target} PRIVATE
        ${LAPACK_LIB}
    )
endfunction()

function(add_cascade_executable target)
    add_executable(${target} ${ARGN} ${CASCADE_COMMON_SOURCES})
    configure_cascade_target(${target})
endfunction()

function(finalize_cascade_test target labels)
    target_include_directories(${target} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/tests
    )
    add_test(NAME ${target} COMMAND ${target})
    set_property(TEST ${target} PROPERTY LABELS "${labels}")
endfunction()

function(add_core_test target test_source)
    add_cascade_executable(${target}
        ${test_source}
    )
    finalize_cascade_test(${target} "core")
endfunction()

function(add_exact_fs_test target test_source)
    add_cascade_executable(${target}
        ${test_source}
        src/decompose_exact_fs.cpp
    )
    link_extended_backend(${target})
    finalize_cascade_test(${target} "exact_fs")
endfunction()

# ── Exact-first diagnostics and tests ────────────────────────
if(CASCADE_BUILD_EXACT_FS)
    add_cascade_executable(exact_fs_diagnostics
        examples/exact_fs_diagnostics.cpp
        src/decompose_exact_fs.cpp
    )
    link_extended_backend(exact_fs_diagnostics)

    add_cascade_executable(cascade_benchmark
        examples/cascade_benchmark.cpp
        src/decompose_exact_fs.cpp
    )
    link_extended_backend(cascade_benchmark)
endif()

if(BUILD_TESTING)
    # ── Core module smoke tests ──────────────────────────────
    add_core_test(cascade_fir_test
        tests/core/cascade_fir_test.cpp
    )

    add_core_test(cascade_decompose_test
        tests/core/cascade_decompose_test.cpp
    )

    add_core_test(freq_sampling_design_test
        tests/core/freq_sampling_design_test.cpp
    )

    add_core_test(signal_generation_test
        tests/core/signal_generation_test.cpp
    )

    # ── Exact-first tests ────────────────────────────────────
    if(CASCADE_BUILD_EXACT_FS)
        add_exact_fs_test(decompose_exact_fs_test
            tests/exact/decompose_exact_fs_test.cpp
        )
        add_exact_fs_test(decompose_exact_fs_full_test
            tests/exact/decompose_exact_fs_full_test.cpp
        )
    endif()
endif()
