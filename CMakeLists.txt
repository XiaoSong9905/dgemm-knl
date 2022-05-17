cmake_minimum_required(VERSION 3.14)
project(dgemm-knl LANGUAGES C CXX)

set(CMAKE_BUILD_TYPE Release)

find_package(BLAS REQUIRED)

# On Cori -- 1.4 GHz * 8 vector width * 2 vector pipelines * 2 flops for FMA = 44.8 GF/s
set(MAX_SPEED 44.8 CACHE STRING "The max speed of the CPU in GF/s")

# We require the GNU compiler for this assignment.
if (NOT "${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
    set(Prg-Intel "PrgEnv-intel")
    set(Prg-Clang "PrgEnv-cray")
    message(WARNING
            "Must use GNU Compiler for submission. Make sure you ran:\n"
            "module swap ${Prg-${CMAKE_C_COMPILER_ID}} PrgEnv-gnu")
endif ()

# Common library target for benchmarking.
add_library(benchmark OBJECT benchmark.cpp)
target_compile_definitions(benchmark PRIVATE MAX_SPEED=${MAX_SPEED})
target_compile_features(benchmark PRIVATE cxx_std_11)

# Benchmark executable
foreach (BENCHMARK IN ITEMS blocked blas)
    # Create benchmarking executable
    add_executable(benchmark-${BENCHMARK} dgemm-${BENCHMARK}.c)
    target_link_libraries(benchmark-${BENCHMARK} PRIVATE benchmark ${BLAS_LIBRARIES})
    target_compile_features(benchmark-${BENCHMARK} PRIVATE c_std_11 c_restrict)
    target_compile_options(benchmark-${BENCHMARK} PRIVATE -Wall -pedantic -march=knl -O3)

    # Generate job script
    configure_file(benchmark.in job-benchmark-${BENCHMARK})
endforeach ()

# Test
foreach (TESTTYPE IN ITEMS packa packb gemm )
    add_executable(test-${TESTTYPE} test-${TESTTYPE}.cpp)
    target_link_libraries(test-${TESTTYPE} PRIVATE ${BLAS_LIBRARIES})
    target_compile_features(test-${TESTTYPE} PRIVATE c_std_11 c_restrict)
    target_compile_options(test-${TESTTYPE} PRIVATE -Wall -pedantic -march=knl)

    # Generate job script
    configure_file(test.in job-test-${TESTTYPE})
endforeach ()
