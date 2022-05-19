/* 
 * Copyright (C) 2022 Xiao Song.
 * All Rights Reserved.
 * Content of this file is not for commertial use.
 */

#include <immintrin.h>
#include <stdlib.h>

const char* dgemm_desc = "Blocked dgemm.";

#define k_b 1620
#define m_b 31
#define m_r 31
#define n_r 8

// https://stackoverflow.com/questions/63404539/portable-loop-unrolling-with-template-parameter-in-c-with-gcc-icc
// Helper macros for stringification
#define TO_STRING_HELPER(X)   #X
#define TO_STRING(X)          TO_STRING_HELPER(X)

// Define loop unrolling depending on the compiler
#if defined(__ICC) || defined(__ICL)
  #define UNROLL_LOOP(n)      _Pragma(TO_STRING(unroll (n)))
#elif defined(__clang__)
  #define UNROLL_LOOP(n)      _Pragma(TO_STRING(unroll (n)))
#elif defined(__GNUC__) && !defined(__clang__)
  #define UNROLL_LOOP(n)      _Pragma(TO_STRING(GCC unroll (16)))
#elif defined(_MSC_BUILD)
  #pragma message ("Microsoft Visual C++ (MSVC) detected: Loop unrolling not supported!")
  #define UNROLL_LOOP(n)
#else
  #warning "Unknown compiler: Loop unrolling not supported!"
  #define UNROLL_LOOP(n)
#endif

void inner_kernel(  double* __restrict__ hat_a, \
                    double* __restrict__ hat_b, \
                    double* __restrict__ hat_c, \
                    int ldc );

void pack_b( double* __restrict__ src_b, double* __restrict__ hat_b, int ldb, int n );

void pack_a( double* __restrict__ src_a, double* __restrict__ hat_a, int lda );

/**
 * @brief Inner kernel for GEMM (row major order)
 * 
 *  \hat C : m_r * n_r
 *  \hat A : m_r * k_b
 *  \hat B : k_b * n_r
 * 
 *  \hat C += \hat A * \hat B
 * 
 *  m_r : 31
 *  n_r : 8
 *  k_b : not required to be hardcode by this function
 */
inline void inner_kernel( double* __restrict__ hat_a, \
                          double* __restrict__ hat_b, \
                          double* __restrict__ hat_c, \
                          int ldc )
{
    // TODO: replace this whole function using assembly code. 
    // Entering asm volatile is expensive
    __m512d R00, R01, R02, R03, R04, R05, R06, R07, R08, R09, \
            R10, R11, R12, R13, R14, R15, R16, R17, R18, R19, \
            R20, R21, R22, R23, R24, R25, R26, R27, R28, R29, \
            R30, R31;

    R00 = _mm512_setzero_pd();
    R01 = _mm512_setzero_pd();
    R02 = _mm512_setzero_pd();
    R03 = _mm512_setzero_pd();
    R04 = _mm512_setzero_pd();
    R05 = _mm512_setzero_pd();
    R06 = _mm512_setzero_pd();
    R07 = _mm512_setzero_pd();
    R08 = _mm512_setzero_pd();
    R09 = _mm512_setzero_pd();

    R10 = _mm512_setzero_pd();
    R11 = _mm512_setzero_pd();
    R12 = _mm512_setzero_pd();
    R13 = _mm512_setzero_pd();
    R14 = _mm512_setzero_pd();
    R15 = _mm512_setzero_pd();
    R16 = _mm512_setzero_pd();
    R17 = _mm512_setzero_pd();
    R18 = _mm512_setzero_pd();
    R19 = _mm512_setzero_pd();

    R20 = _mm512_setzero_pd();
    R21 = _mm512_setzero_pd();
    R22 = _mm512_setzero_pd();
    R23 = _mm512_setzero_pd();
    R24 = _mm512_setzero_pd();
    R25 = _mm512_setzero_pd();
    R26 = _mm512_setzero_pd();
    R27 = _mm512_setzero_pd();
    R28 = _mm512_setzero_pd();
    R29 = _mm512_setzero_pd();

    R30 = _mm512_setzero_pd();

    // R00 = _mm512_loadu_pd( (void*)(hat_c + 0 * ldc) );
    // R01 = _mm512_loadu_pd( (void*)(hat_c + 1 * ldc) );
    // R02 = _mm512_loadu_pd( (void*)(hat_c + 2 * ldc) );
    // R03 = _mm512_loadu_pd( (void*)(hat_c + 3 * ldc) );
    // R04 = _mm512_loadu_pd( (void*)(hat_c + 4 * ldc) );
    // R05 = _mm512_loadu_pd( (void*)(hat_c + 5 * ldc) );
    // R06 = _mm512_loadu_pd( (void*)(hat_c + 6 * ldc) );
    // R07 = _mm512_loadu_pd( (void*)(hat_c + 7 * ldc) );
    // R08 = _mm512_loadu_pd( (void*)(hat_c + 8 * ldc) );
    // R09 = _mm512_loadu_pd( (void*)(hat_c + 9 * ldc) );

    // R10 = _mm512_loadu_pd( (void*)(hat_c + 10 * ldc) );
    // R11 = _mm512_loadu_pd( (void*)(hat_c + 11 * ldc) );
    // R12 = _mm512_loadu_pd( (void*)(hat_c + 12 * ldc) );
    // R13 = _mm512_loadu_pd( (void*)(hat_c + 13 * ldc) );
    // R14 = _mm512_loadu_pd( (void*)(hat_c + 14 * ldc) );
    // R15 = _mm512_loadu_pd( (void*)(hat_c + 15 * ldc) );
    // R16 = _mm512_loadu_pd( (void*)(hat_c + 16 * ldc) );
    // R17 = _mm512_loadu_pd( (void*)(hat_c + 17 * ldc) );
    // R18 = _mm512_loadu_pd( (void*)(hat_c + 18 * ldc) );
    // R19 = _mm512_loadu_pd( (void*)(hat_c + 19 * ldc) );

    // R20 = _mm512_loadu_pd( (void*)(hat_c + 20 * ldc) );
    // R21 = _mm512_loadu_pd( (void*)(hat_c + 21 * ldc) );
    // R22 = _mm512_loadu_pd( (void*)(hat_c + 22 * ldc) );
    // R23 = _mm512_loadu_pd( (void*)(hat_c + 23 * ldc) );
    // R24 = _mm512_loadu_pd( (void*)(hat_c + 24 * ldc) );
    // R25 = _mm512_loadu_pd( (void*)(hat_c + 25 * ldc) );
    // R26 = _mm512_loadu_pd( (void*)(hat_c + 26 * ldc) );
    // R27 = _mm512_loadu_pd( (void*)(hat_c + 27 * ldc) );
    // R28 = _mm512_loadu_pd( (void*)(hat_c + 28 * ldc) );
    // R29 = _mm512_loadu_pd( (void*)(hat_c + 29 * ldc) );

    // R30 = _mm512_loadu_pd( (void*)(hat_c + 30 * ldc) );

    UNROLL_LOOP( 3 )
    for ( int i = 0; i < k_b; ++i )
    {
        // Software prefetch from L2 to L1 for \hat A \hat B
        // Each _mm_prefetch load one cache line of data
        // \hat A need to load m_r * 8 (size of double) / 64 (size of cache line) = 3.8 cache line
        // \hat B need to load n_r * 8 / 64 = 1 cache line
        _mm_prefetch( hat_a + 12 * m_r + 64 * 0, _MM_HINT_T0 );
        _mm_prefetch( hat_a + 12 * m_r + 64 * 1, _MM_HINT_T0 );
        _mm_prefetch( hat_a + 12 * m_r + 64 * 2, _MM_HINT_T0 );
        _mm_prefetch( hat_a + 12 * m_r + 64 * 3, _MM_HINT_T0 );
        _mm_prefetch( hat_b + 32 * n_r + 64 * 0, _MM_HINT_T0 );

        R31 = _mm512_load_pd( hat_b );

        // R00 = _mm512_fmadd_pd( _mm512_set1_pd( *(hat_a + 0) ), R31, R00 );
        // R01 = _mm512_fmadd_pd( _mm512_set1_pd( *(hat_a + 1) ), R31, R01 );
        // R02 = _mm512_fmadd_pd( _mm512_set1_pd( *(hat_a + 2) ), R31, R02 );
        // R03 = _mm512_fmadd_pd( _mm512_set1_pd( *(hat_a + 3) ), R31, R03 );
        // R04 = _mm512_fmadd_pd( _mm512_set1_pd( *(hat_a + 4) ), R31, R04 );
        // R05 = _mm512_fmadd_pd( _mm512_set1_pd( *(hat_a + 5) ), R31, R05 );
        // R06 = _mm512_fmadd_pd( _mm512_set1_pd( *(hat_a + 6) ), R31, R06 );
        // R07 = _mm512_fmadd_pd( _mm512_set1_pd( *(hat_a + 7) ), R31, R07 );
        // R08 = _mm512_fmadd_pd( _mm512_set1_pd( *(hat_a + 8) ), R31, R08 );
        // R09 = _mm512_fmadd_pd( _mm512_set1_pd( *(hat_a + 9) ), R31, R09 );

        // R10 = _mm512_fmadd_pd( _mm512_set1_pd( *(hat_a + 10) ), R31, R10 );
        // R11 = _mm512_fmadd_pd( _mm512_set1_pd( *(hat_a + 11) ), R31, R11 );
        // R12 = _mm512_fmadd_pd( _mm512_set1_pd( *(hat_a + 12) ), R31, R12 );
        // R13 = _mm512_fmadd_pd( _mm512_set1_pd( *(hat_a + 13) ), R31, R13 );
        // R14 = _mm512_fmadd_pd( _mm512_set1_pd( *(hat_a + 14) ), R31, R14 );
        // R15 = _mm512_fmadd_pd( _mm512_set1_pd( *(hat_a + 15) ), R31, R15 );
        // R16 = _mm512_fmadd_pd( _mm512_set1_pd( *(hat_a + 16) ), R31, R16 );
        // R17 = _mm512_fmadd_pd( _mm512_set1_pd( *(hat_a + 17) ), R31, R17 );
        // R18 = _mm512_fmadd_pd( _mm512_set1_pd( *(hat_a + 18) ), R31, R18 );
        // R19 = _mm512_fmadd_pd( _mm512_set1_pd( *(hat_a + 19) ), R31, R19 );

        // R20 = _mm512_fmadd_pd( _mm512_set1_pd( *(hat_a + 20) ), R31, R20 );
        // R21 = _mm512_fmadd_pd( _mm512_set1_pd( *(hat_a + 21) ), R31, R21 );
        // R22 = _mm512_fmadd_pd( _mm512_set1_pd( *(hat_a + 22) ), R31, R22 );
        // R23 = _mm512_fmadd_pd( _mm512_set1_pd( *(hat_a + 23) ), R31, R23 );
        // R24 = _mm512_fmadd_pd( _mm512_set1_pd( *(hat_a + 24) ), R31, R24 );
        // R25 = _mm512_fmadd_pd( _mm512_set1_pd( *(hat_a + 25) ), R31, R25 );
        // R26 = _mm512_fmadd_pd( _mm512_set1_pd( *(hat_a + 26) ), R31, R26 );
        // R27 = _mm512_fmadd_pd( _mm512_set1_pd( *(hat_a + 27) ), R31, R27 );
        // R28 = _mm512_fmadd_pd( _mm512_set1_pd( *(hat_a + 28) ), R31, R28 );
        // R29 = _mm512_fmadd_pd( _mm512_set1_pd( *(hat_a + 29) ), R31, R29 );

        // R30 = _mm512_fmadd_pd( _mm512_set1_pd( *(hat_a + 30) ), R31, R30 );
                
        asm volatile(
            "vfmadd231pd   0(%[hat_a])%{1to8}, %[R31], %[R00]\n\t"
            "vfmadd231pd   8(%[hat_a])%{1to8}, %[R31], %[R01]\n\t"
            "vfmadd231pd  16(%[hat_a])%{1to8}, %[R31], %[R02]\n\t"
            "vfmadd231pd  24(%[hat_a])%{1to8}, %[R31], %[R03]\n\t"
            "vfmadd231pd  32(%[hat_a])%{1to8}, %[R31], %[R04]\n\t"
            "vfmadd231pd  40(%[hat_a])%{1to8}, %[R31], %[R05]\n\t"
            "vfmadd231pd  48(%[hat_a])%{1to8}, %[R31], %[R06]\n\t"
            "vfmadd231pd  56(%[hat_a])%{1to8}, %[R31], %[R07]\n\t"
            "vfmadd231pd  64(%[hat_a])%{1to8}, %[R31], %[R08]\n\t"
            "vfmadd231pd  72(%[hat_a])%{1to8}, %[R31], %[R09]\n\t"
            : [R00] "+v" (R00),
              [R01] "+v" (R01),
              [R02] "+v" (R02),
              [R03] "+v" (R03),
              [R04] "+v" (R04),
              [R05] "+v" (R05),
              [R06] "+v" (R06),
              [R07] "+v" (R07),
              [R08] "+v" (R08),
              [R09] "+v" (R09)
            : [R31] "v"  (R31),
              [hat_a] "r" (hat_a)
            : "memory"
        ); 

        asm volatile(
            "vfmadd231pd  80(%[hat_a])%{1to8}, %[R31], %[R10]\n\t"
            "vfmadd231pd  88(%[hat_a])%{1to8}, %[R31], %[R11]\n\t"
            "vfmadd231pd  96(%[hat_a])%{1to8}, %[R31], %[R12]\n\t"
            "vfmadd231pd 104(%[hat_a])%{1to8}, %[R31], %[R13]\n\t"
            "vfmadd231pd 112(%[hat_a])%{1to8}, %[R31], %[R14]\n\t"
            "vfmadd231pd 120(%[hat_a])%{1to8}, %[R31], %[R15]\n\t"
            "vfmadd231pd 128(%[hat_a])%{1to8}, %[R31], %[R16]\n\t"
            "vfmadd231pd 136(%[hat_a])%{1to8}, %[R31], %[R17]\n\t"
            "vfmadd231pd 144(%[hat_a])%{1to8}, %[R31], %[R18]\n\t"
            "vfmadd231pd 152(%[hat_a])%{1to8}, %[R31], %[R19]\n\t"
            : [R10] "+v" (R10),
              [R11] "+v" (R11),
              [R12] "+v" (R12),
              [R13] "+v" (R13),
              [R14] "+v" (R14),
              [R15] "+v" (R15),
              [R16] "+v" (R16),
              [R17] "+v" (R17),
              [R18] "+v" (R18),
              [R19] "+v" (R19)
            : [R31] "v"  (R31),
              [hat_a] "r" (hat_a)
            : "memory"
        ); 

        asm volatile(
            "vfmadd231pd 160(%[hat_a])%{1to8}, %[R31], %[R20]\n\t"
            "vfmadd231pd 168(%[hat_a])%{1to8}, %[R31], %[R21]\n\t"
            "vfmadd231pd 176(%[hat_a])%{1to8}, %[R31], %[R22]\n\t"
            "vfmadd231pd 184(%[hat_a])%{1to8}, %[R31], %[R23]\n\t"
            "vfmadd231pd 192(%[hat_a])%{1to8}, %[R31], %[R24]\n\t"
            "vfmadd231pd 200(%[hat_a])%{1to8}, %[R31], %[R25]\n\t"
            "vfmadd231pd 208(%[hat_a])%{1to8}, %[R31], %[R26]\n\t"
            "vfmadd231pd 216(%[hat_a])%{1to8}, %[R31], %[R27]\n\t"
            "vfmadd231pd 224(%[hat_a])%{1to8}, %[R31], %[R28]\n\t"
            "vfmadd231pd 232(%[hat_a])%{1to8}, %[R31], %[R29]\n\t"
            "vfmadd231pd 240(%[hat_a])%{1to8}, %[R31], %[R30]\n\t"
            : [R20] "+v" (R20),
              [R21] "+v" (R21),
              [R22] "+v" (R22),
              [R23] "+v" (R23),
              [R24] "+v" (R24),
              [R25] "+v" (R25),
              [R26] "+v" (R26),
              [R27] "+v" (R27),
              [R28] "+v" (R28),
              [R29] "+v" (R29),
              [R30] "+v" (R30)
            : [R31] "v"  (R31),
              [hat_a] "r" (hat_a)
            : "memory"
        ); 

        hat_a += m_r;
        hat_b += n_r;
    }

    _mm512_storeu_pd( (void*)(hat_c +  0 * ldc), _mm512_add_pd( _mm512_loadu_pd( (void*)(hat_c +  0 * ldc) ), R00 ) );
    _mm512_storeu_pd( (void*)(hat_c +  1 * ldc), _mm512_add_pd( _mm512_loadu_pd( (void*)(hat_c +  1 * ldc) ), R01 ) );
    _mm512_storeu_pd( (void*)(hat_c +  2 * ldc), _mm512_add_pd( _mm512_loadu_pd( (void*)(hat_c +  2 * ldc) ), R02 ) );
    _mm512_storeu_pd( (void*)(hat_c +  3 * ldc), _mm512_add_pd( _mm512_loadu_pd( (void*)(hat_c +  3 * ldc) ), R03 ) );
    _mm512_storeu_pd( (void*)(hat_c +  4 * ldc), _mm512_add_pd( _mm512_loadu_pd( (void*)(hat_c +  4 * ldc) ), R04 ) );
    _mm512_storeu_pd( (void*)(hat_c +  5 * ldc), _mm512_add_pd( _mm512_loadu_pd( (void*)(hat_c +  5 * ldc) ), R05 ) );
    _mm512_storeu_pd( (void*)(hat_c +  6 * ldc), _mm512_add_pd( _mm512_loadu_pd( (void*)(hat_c +  6 * ldc) ), R06 ) );
    _mm512_storeu_pd( (void*)(hat_c +  7 * ldc), _mm512_add_pd( _mm512_loadu_pd( (void*)(hat_c +  7 * ldc) ), R07 ) );
    _mm512_storeu_pd( (void*)(hat_c +  8 * ldc), _mm512_add_pd( _mm512_loadu_pd( (void*)(hat_c +  8 * ldc) ), R08 ) );
    _mm512_storeu_pd( (void*)(hat_c +  9 * ldc), _mm512_add_pd( _mm512_loadu_pd( (void*)(hat_c +  9 * ldc) ), R09 ) );

    _mm512_storeu_pd( (void*)(hat_c + 10 * ldc), _mm512_add_pd( _mm512_loadu_pd( (void*)(hat_c + 10 * ldc) ), R10 ) );
    _mm512_storeu_pd( (void*)(hat_c + 11 * ldc), _mm512_add_pd( _mm512_loadu_pd( (void*)(hat_c + 11 * ldc) ), R11 ) );
    _mm512_storeu_pd( (void*)(hat_c + 12 * ldc), _mm512_add_pd( _mm512_loadu_pd( (void*)(hat_c + 12 * ldc) ), R12 ) );
    _mm512_storeu_pd( (void*)(hat_c + 13 * ldc), _mm512_add_pd( _mm512_loadu_pd( (void*)(hat_c + 13 * ldc) ), R13 ) );
    _mm512_storeu_pd( (void*)(hat_c + 14 * ldc), _mm512_add_pd( _mm512_loadu_pd( (void*)(hat_c + 14 * ldc) ), R14 ) );
    _mm512_storeu_pd( (void*)(hat_c + 15 * ldc), _mm512_add_pd( _mm512_loadu_pd( (void*)(hat_c + 15 * ldc) ), R15 ) );
    _mm512_storeu_pd( (void*)(hat_c + 16 * ldc), _mm512_add_pd( _mm512_loadu_pd( (void*)(hat_c + 16 * ldc) ), R16 ) );
    _mm512_storeu_pd( (void*)(hat_c + 17 * ldc), _mm512_add_pd( _mm512_loadu_pd( (void*)(hat_c + 17 * ldc) ), R17 ) );
    _mm512_storeu_pd( (void*)(hat_c + 18 * ldc), _mm512_add_pd( _mm512_loadu_pd( (void*)(hat_c + 18 * ldc) ), R18 ) );
    _mm512_storeu_pd( (void*)(hat_c + 19 * ldc), _mm512_add_pd( _mm512_loadu_pd( (void*)(hat_c + 19 * ldc) ), R19 ) );

    _mm512_storeu_pd( (void*)(hat_c + 20 * ldc), _mm512_add_pd( _mm512_loadu_pd( (void*)(hat_c + 20 * ldc) ), R20 ) );
    _mm512_storeu_pd( (void*)(hat_c + 21 * ldc), _mm512_add_pd( _mm512_loadu_pd( (void*)(hat_c + 21 * ldc) ), R21 ) );
    _mm512_storeu_pd( (void*)(hat_c + 22 * ldc), _mm512_add_pd( _mm512_loadu_pd( (void*)(hat_c + 22 * ldc) ), R22 ) );
    _mm512_storeu_pd( (void*)(hat_c + 23 * ldc), _mm512_add_pd( _mm512_loadu_pd( (void*)(hat_c + 23 * ldc) ), R23 ) );
    _mm512_storeu_pd( (void*)(hat_c + 24 * ldc), _mm512_add_pd( _mm512_loadu_pd( (void*)(hat_c + 24 * ldc) ), R24 ) );
    _mm512_storeu_pd( (void*)(hat_c + 25 * ldc), _mm512_add_pd( _mm512_loadu_pd( (void*)(hat_c + 25 * ldc) ), R25 ) );
    _mm512_storeu_pd( (void*)(hat_c + 26 * ldc), _mm512_add_pd( _mm512_loadu_pd( (void*)(hat_c + 26 * ldc) ), R26 ) );
    _mm512_storeu_pd( (void*)(hat_c + 27 * ldc), _mm512_add_pd( _mm512_loadu_pd( (void*)(hat_c + 27 * ldc) ), R27 ) );
    _mm512_storeu_pd( (void*)(hat_c + 28 * ldc), _mm512_add_pd( _mm512_loadu_pd( (void*)(hat_c + 28 * ldc) ), R28 ) );
    _mm512_storeu_pd( (void*)(hat_c + 29 * ldc), _mm512_add_pd( _mm512_loadu_pd( (void*)(hat_c + 29 * ldc) ), R29 ) );

    _mm512_storeu_pd( (void*)(hat_c + 30 * ldc), _mm512_add_pd( _mm512_loadu_pd( (void*)(hat_c + 30 * ldc) ), R30 ) );

    // _mm512_storeu_pd( (void*)(hat_c +  0 * ldc), R00 );
    // _mm512_storeu_pd( (void*)(hat_c +  1 * ldc), R01 );
    // _mm512_storeu_pd( (void*)(hat_c +  2 * ldc), R02 );
    // _mm512_storeu_pd( (void*)(hat_c +  3 * ldc), R03 );
    // _mm512_storeu_pd( (void*)(hat_c +  4 * ldc), R04 );
    // _mm512_storeu_pd( (void*)(hat_c +  5 * ldc), R05 );
    // _mm512_storeu_pd( (void*)(hat_c +  6 * ldc), R06 );
    // _mm512_storeu_pd( (void*)(hat_c +  7 * ldc), R07 );
    // _mm512_storeu_pd( (void*)(hat_c +  8 * ldc), R08 );
    // _mm512_storeu_pd( (void*)(hat_c +  9 * ldc), R09 );

    // _mm512_storeu_pd( (void*)(hat_c + 10 * ldc), R10 );
    // _mm512_storeu_pd( (void*)(hat_c + 11 * ldc), R11 );
    // _mm512_storeu_pd( (void*)(hat_c + 12 * ldc), R12 );
    // _mm512_storeu_pd( (void*)(hat_c + 13 * ldc), R13 );
    // _mm512_storeu_pd( (void*)(hat_c + 14 * ldc), R14 );
    // _mm512_storeu_pd( (void*)(hat_c + 15 * ldc), R15 );
    // _mm512_storeu_pd( (void*)(hat_c + 16 * ldc), R16 );
    // _mm512_storeu_pd( (void*)(hat_c + 17 * ldc), R17 );
    // _mm512_storeu_pd( (void*)(hat_c + 18 * ldc), R18 );
    // _mm512_storeu_pd( (void*)(hat_c + 19 * ldc), R19 );

    // _mm512_storeu_pd( (void*)(hat_c + 20 * ldc), R20 );
    // _mm512_storeu_pd( (void*)(hat_c + 21 * ldc), R21 );
    // _mm512_storeu_pd( (void*)(hat_c + 22 * ldc), R22 );
    // _mm512_storeu_pd( (void*)(hat_c + 23 * ldc), R23 );
    // _mm512_storeu_pd( (void*)(hat_c + 24 * ldc), R24 );
    // _mm512_storeu_pd( (void*)(hat_c + 25 * ldc), R25 );
    // _mm512_storeu_pd( (void*)(hat_c + 26 * ldc), R26 );
    // _mm512_storeu_pd( (void*)(hat_c + 27 * ldc), R27 );
    // _mm512_storeu_pd( (void*)(hat_c + 28 * ldc), R28 );
    // _mm512_storeu_pd( (void*)(hat_c + 29 * ldc), R29 );

    // _mm512_storeu_pd( (void*)(hat_c + 30 * ldc), R30 );
}


/**
 * @brief Pack k_b * n_r of submatrix B (row major order)
 * 
 */
void pack_b( double* __restrict__ src_b, double* __restrict__ pak_b, int ldb, int n )
{
    for ( int row_i = 0; row_i < k_b; ++row_i )
    {
        double* src_b_row_i = src_b + row_i * ldb;
        double* pak_b_row_i = pak_b + row_i * n_r;

        UNROLL_LOOP( 4 )
        for ( int n_r_i = 0; n_r_i < (n / n_r); ++n_r_i )
        {
            double* src_b_row_n_r_i = src_b_row_i + n_r_i * n_r;
            double* pak_b_row_n_r_i = pak_b_row_i + n_r_i * k_b * n_r;

            UNROLL_LOOP( n_r )
            for ( int col_i = 0; col_i < n_r; ++col_i )
            {
                *(pak_b_row_n_r_i + col_i) = *(src_b_row_n_r_i + col_i);
            }
        }
    }
}


/**
 * @brief Pack m_b * k_b of submatrix A (row major order)
 * 
 */
void pack_a( double* __restrict__ src_a, double* __restrict__ pak_a, int lda )
{
    for ( int m_r_i = 0; m_r_i < (m_b / m_r); ++m_r_i )
    {
        double* src_a_row_m_r_i = src_a + m_r_i * m_r * lda;
        double* pak_a_row_m_r_i = pak_a + m_r_i * m_r * k_b;

        UNROLL_LOOP( 4 )
        for ( int row_i = 0; row_i < m_r; ++row_i )
        {
            double* src_a_row_i = src_a_row_m_r_i + row_i * lda;
            double* pak_a_row_i = pak_a_row_m_r_i + row_i;

            UNROLL_LOOP( 8 * 4 )
            for ( int col_i = 0; col_i < k_b; ++col_i )
            {
                *(pak_a_row_i + col_i * m_r) = *(src_a_row_i + col_i);
            }
        }
    }
}


/**
 * @brief DGEMM on KNL Node (row major order)
 *  A : m * k
 *  B : k * n
 *  C : m * n
 */
void dgemm_knl( int m, int k, int n, \
                double* src_a, double* src_b, double* src_c, \
                int lda, int ldb, int ldc )
{
    // Memory for \tilde a and \tilde b
    double* pak_a = (double*)_mm_malloc( m_b * k_b * sizeof( double ), 64 );
    double* pak_b = (double*)_mm_malloc( k_b * n   * sizeof( double ), 64 );

    for ( int k_b_i = 0; k_b_i < k / k_b; k_b_i++)
    {
        // Pack \tilde b
        pack_b( src_b + k_b_i * k_b * ldb, pak_b, ldb, n );

        for ( int m_b_i = 0; m_b_i < m / m_b; m_b_i++ )
        {
            // Pack \tilde a
            pack_a( src_a + m_b_i * m_b * lda + k_b_i * k_b, pak_a, lda );

            for ( int n_r_i = 0; n_r_i < n / n_r; n_r_i++ )
            {
                for ( int m_r_i = 0; m_r_i < m_b / m_r; m_r_i++ )
                {
                    // Inner Kernel (register blocking)
                    inner_kernel( pak_a + m_r_i * m_r * k_b, \
                                  pak_b + n_r_i * n_r * k_b, \
                                  src_c + m_b_i * m_b * ldc + m_r_i * m_r * ldc + n_r_i * n_r, \
                                  ldc );
                }
            }
        }
    }

    _mm_free( pak_a );
    _mm_free( pak_b );
}
