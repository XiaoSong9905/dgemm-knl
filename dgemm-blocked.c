/* 
 * Copyright (C) 2022 Xiao Song.
 * All Rights Reserved.
 * Content of this file is not for commertial use.
 */

#include <immintrin.h>
#include <stdlib.h>

const char* dgemm_desc = "Blocked dgemm.";

#define k_b 1624
#define m_b 31
#define m_r 31
#define n_r 8

/**
 * @brief Inner kernel for GEMM
 * 
 *  \hat C : m_r * n_r
 *  \hat A : m_r * k_b
 *  \hat B : k_b * n_r
 * 
 *  m_r : 31
 *  n_r : 8
 *  k_b : as argument
 */
inline void inner_kernel( const double* __restrict__ hat_a, \
                          const double* __restrict__ hat_b, \
                                double*              hat_c, \
                          int ldc )
{
    __m512d R00, R01, R02, R03, R04, R05, R06, R07, R08, R09, \
            R10, R11, R12, R13, R14, R15, R16, R17, R18, R19, \
            R20, R21, R22, R23, R24, R25, R26, R27, R28, R29, \
            R30, R31;

    R00 = _mm512_loadu_pd( (void*)(hat_c + 0 * ldc) );
    R01 = _mm512_loadu_pd( (void*)(hat_c + 1 * ldc) );
    R02 = _mm512_loadu_pd( (void*)(hat_c + 2 * ldc) );
    R03 = _mm512_loadu_pd( (void*)(hat_c + 3 * ldc) );
    R04 = _mm512_loadu_pd( (void*)(hat_c + 4 * ldc) );
    R05 = _mm512_loadu_pd( (void*)(hat_c + 5 * ldc) );
    R06 = _mm512_loadu_pd( (void*)(hat_c + 6 * ldc) );
    R07 = _mm512_loadu_pd( (void*)(hat_c + 7 * ldc) );
    R08 = _mm512_loadu_pd( (void*)(hat_c + 8 * ldc) );
    R09 = _mm512_loadu_pd( (void*)(hat_c + 9 * ldc) );

    R10 = _mm512_loadu_pd( (void*)(hat_c + 11 * ldc) );
    R11 = _mm512_loadu_pd( (void*)(hat_c + 11 * ldc) );
    R12 = _mm512_loadu_pd( (void*)(hat_c + 12 * ldc) );
    R13 = _mm512_loadu_pd( (void*)(hat_c + 13 * ldc) );
    R14 = _mm512_loadu_pd( (void*)(hat_c + 14 * ldc) );
    R15 = _mm512_loadu_pd( (void*)(hat_c + 15 * ldc) );
    R16 = _mm512_loadu_pd( (void*)(hat_c + 16 * ldc) );
    R17 = _mm512_loadu_pd( (void*)(hat_c + 17 * ldc) );
    R18 = _mm512_loadu_pd( (void*)(hat_c + 18 * ldc) );
    R19 = _mm512_loadu_pd( (void*)(hat_c + 19 * ldc) );

    R20 = _mm512_loadu_pd( (void*)(hat_c + 22 * ldc) );
    R21 = _mm512_loadu_pd( (void*)(hat_c + 21 * ldc) );
    R22 = _mm512_loadu_pd( (void*)(hat_c + 22 * ldc) );
    R23 = _mm512_loadu_pd( (void*)(hat_c + 23 * ldc) );
    R24 = _mm512_loadu_pd( (void*)(hat_c + 24 * ldc) );
    R25 = _mm512_loadu_pd( (void*)(hat_c + 25 * ldc) );
    R26 = _mm512_loadu_pd( (void*)(hat_c + 26 * ldc) );
    R27 = _mm512_loadu_pd( (void*)(hat_c + 27 * ldc) );
    R28 = _mm512_loadu_pd( (void*)(hat_c + 28 * ldc) );
    R29 = _mm512_loadu_pd( (void*)(hat_c + 29 * ldc) );

    R30 = _mm512_loadu_pd( (void*)(hat_c + 30 * ldc) );

    #pragma GCC unroll 3
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

        R00 = _mm512_fmadd_pd( _mm512_set1_pd( *(hat_a + 0) ), R31, R00 );
        R01 = _mm512_fmadd_pd( _mm512_set1_pd( *(hat_a + 1) ), R31, R01 );
        R02 = _mm512_fmadd_pd( _mm512_set1_pd( *(hat_a + 2) ), R31, R02 );
        R03 = _mm512_fmadd_pd( _mm512_set1_pd( *(hat_a + 3) ), R31, R03 );
        R04 = _mm512_fmadd_pd( _mm512_set1_pd( *(hat_a + 4) ), R31, R04 );
        R05 = _mm512_fmadd_pd( _mm512_set1_pd( *(hat_a + 5) ), R31, R05 );
        R06 = _mm512_fmadd_pd( _mm512_set1_pd( *(hat_a + 6) ), R31, R06 );
        R07 = _mm512_fmadd_pd( _mm512_set1_pd( *(hat_a + 7) ), R31, R07 );
        R08 = _mm512_fmadd_pd( _mm512_set1_pd( *(hat_a + 8) ), R31, R08 );
        R09 = _mm512_fmadd_pd( _mm512_set1_pd( *(hat_a + 9) ), R31, R09 );

        R10 = _mm512_fmadd_pd( _mm512_set1_pd( *(hat_a + 10) ), R31, R10 );
        R11 = _mm512_fmadd_pd( _mm512_set1_pd( *(hat_a + 11) ), R31, R11 );
        R12 = _mm512_fmadd_pd( _mm512_set1_pd( *(hat_a + 12) ), R31, R12 );
        R13 = _mm512_fmadd_pd( _mm512_set1_pd( *(hat_a + 13) ), R31, R13 );
        R14 = _mm512_fmadd_pd( _mm512_set1_pd( *(hat_a + 14) ), R31, R14 );
        R15 = _mm512_fmadd_pd( _mm512_set1_pd( *(hat_a + 15) ), R31, R15 );
        R16 = _mm512_fmadd_pd( _mm512_set1_pd( *(hat_a + 16) ), R31, R16 );
        R17 = _mm512_fmadd_pd( _mm512_set1_pd( *(hat_a + 17) ), R31, R17 );
        R18 = _mm512_fmadd_pd( _mm512_set1_pd( *(hat_a + 18) ), R31, R18 );
        R19 = _mm512_fmadd_pd( _mm512_set1_pd( *(hat_a + 19) ), R31, R19 );

        R20 = _mm512_fmadd_pd( _mm512_set1_pd( *(hat_a + 20) ), R31, R20 );
        R21 = _mm512_fmadd_pd( _mm512_set1_pd( *(hat_a + 21) ), R31, R21 );
        R22 = _mm512_fmadd_pd( _mm512_set1_pd( *(hat_a + 22) ), R31, R22 );
        R23 = _mm512_fmadd_pd( _mm512_set1_pd( *(hat_a + 23) ), R31, R23 );
        R24 = _mm512_fmadd_pd( _mm512_set1_pd( *(hat_a + 24) ), R31, R24 );
        R25 = _mm512_fmadd_pd( _mm512_set1_pd( *(hat_a + 25) ), R31, R25 );
        R26 = _mm512_fmadd_pd( _mm512_set1_pd( *(hat_a + 26) ), R31, R26 );
        R27 = _mm512_fmadd_pd( _mm512_set1_pd( *(hat_a + 27) ), R31, R27 );
        R28 = _mm512_fmadd_pd( _mm512_set1_pd( *(hat_a + 28) ), R31, R28 );
        R29 = _mm512_fmadd_pd( _mm512_set1_pd( *(hat_a + 29) ), R31, R29 );

        R30 = _mm512_fmadd_pd( _mm512_set1_pd( *(hat_a + 30) ), R31, R30 );

        hat_a += m_r;
        hat_b += n_r;
    }

    _mm512_storeu_pd( (void*)(hat_c +  0 * ldc), R00 );
    _mm512_storeu_pd( (void*)(hat_c +  1 * ldc), R01 );
    _mm512_storeu_pd( (void*)(hat_c +  2 * ldc), R02 );
    _mm512_storeu_pd( (void*)(hat_c +  3 * ldc), R03 );
    _mm512_storeu_pd( (void*)(hat_c +  4 * ldc), R04 );
    _mm512_storeu_pd( (void*)(hat_c +  5 * ldc), R05 );
    _mm512_storeu_pd( (void*)(hat_c +  6 * ldc), R06 );
    _mm512_storeu_pd( (void*)(hat_c +  7 * ldc), R07 );
    _mm512_storeu_pd( (void*)(hat_c +  8 * ldc), R08 );
    _mm512_storeu_pd( (void*)(hat_c +  9 * ldc), R09 );

    _mm512_storeu_pd( (void*)(hat_c + 10 * ldc), R10 );
    _mm512_storeu_pd( (void*)(hat_c + 11 * ldc), R11 );
    _mm512_storeu_pd( (void*)(hat_c + 12 * ldc), R12 );
    _mm512_storeu_pd( (void*)(hat_c + 13 * ldc), R13 );
    _mm512_storeu_pd( (void*)(hat_c + 14 * ldc), R14 );
    _mm512_storeu_pd( (void*)(hat_c + 15 * ldc), R15 );
    _mm512_storeu_pd( (void*)(hat_c + 16 * ldc), R16 );
    _mm512_storeu_pd( (void*)(hat_c + 17 * ldc), R17 );
    _mm512_storeu_pd( (void*)(hat_c + 18 * ldc), R18 );
    _mm512_storeu_pd( (void*)(hat_c + 19 * ldc), R19 );

    _mm512_storeu_pd( (void*)(hat_c + 20 * ldc), R20 );
    _mm512_storeu_pd( (void*)(hat_c + 21 * ldc), R21 );
    _mm512_storeu_pd( (void*)(hat_c + 22 * ldc), R22 );
    _mm512_storeu_pd( (void*)(hat_c + 23 * ldc), R23 );
    _mm512_storeu_pd( (void*)(hat_c + 24 * ldc), R24 );
    _mm512_storeu_pd( (void*)(hat_c + 25 * ldc), R25 );
    _mm512_storeu_pd( (void*)(hat_c + 26 * ldc), R26 );
    _mm512_storeu_pd( (void*)(hat_c + 27 * ldc), R27 );
    _mm512_storeu_pd( (void*)(hat_c + 28 * ldc), R28 );
    _mm512_storeu_pd( (void*)(hat_c + 29 * ldc), R29 );

    _mm512_storeu_pd( (void*)(hat_c + 30 * ldc), R30 );
}


/**
 * @brief Pack k_b * n_r of submatrix B
 * 
 */
void pack_b( const double* src_b, \
                   double* pack_b, \
             int ldb )
{
    // masks for unit 1 shuffle
    // 2 512 bits register used
    long unit1shuffle_mask0[8] __attribute__((aligned(64))) = {0, 8, 2, 10, 4, 12, 6, 14};
    long unit1shuffle_mask1[8] __attribute__((aligned(64))) = {1, 9, 3, 11, 5, 13, 7, 15};
    __m512i v_unit1shuffle_mask0 = _mm512_load_epi64( &unit1shuffle_mask0 );
    __m512i v_unit1shuffle_mask1 = _mm512_load_epi64( &unit1shuffle_mask1 );

    // masks for unit 2 shuffle
    // 2 512 bits register used
    long unit2shuffle_mask0[8] __attribute__((aligned(64))) = {0, 1,  8,  9, 4, 5, 12, 13};
    long unit2shuffle_mask1[8] __attribute__((aligned(64))) = {2, 3, 10, 11, 6, 7, 14, 15};
    __m512i v_unit2shuffle_mask0 = _mm512_load_epi64( &unit2shuffle_mask0 );
    __m512i v_unit2shuffle_mask1 = _mm512_load_epi64( &unit2shuffle_mask1 );

    // mask for unit 4 shuffle
    // 2 512 bits register used
    long unit4shuffle_mask0[8] __attribute__((aligned(64))) = {0, 1, 2, 3,  8,  9, 10, 11};
    long unit4shuffle_mask1[8] __attribute__((aligned(64))) = {4, 5, 6, 7, 12, 13, 14, 15};
    __m512i v_unit4shuffle_mask0 = _mm512_load_epi64( &unit4shuffle_mask0 );
    __m512i v_unit4shuffle_mask1 = _mm512_load_epi64( &unit4shuffle_mask1 );

    // 16 512 bits registers used
    __m512d x0, x1, x2, x3, x4, x5, x6, x7;
    __m512d y0, y1, y2, y3, y4, y5, y6, y7;

    // k_b (1624) * n_r (8) is packed by running multiple 8x8 transpose kernel
    // Set k_b to multiply of 8 for easier transpose
    for ( int iter_i = 0; iter_i < k_b / 8; iter_i++ )
    {
        const double* src_col0 = src_b + 0 * ldb;
        const double* src_col1 = src_b + 1 * ldb;
        const double* src_col2 = src_b + 2 * ldb;
        const double* src_col3 = src_b + 3 * ldb;

        const double* src_col4 = src_b + 4 * ldb;
        const double* src_col5 = src_b + 5 * ldb;
        const double* src_col6 = src_b + 6 * ldb;
        const double* src_col7 = src_b + 7 * ldb;

        // TODO: consider add prefetch here
        // maybe add prefetch for every column, prefetch next cache line (can still be place under L1 cache)

        x0 = _mm512_loadu_pd( src_col0 );
        x1 = _mm512_loadu_pd( src_col1 );
        x2 = _mm512_loadu_pd( src_col2 );
        x3 = _mm512_loadu_pd( src_col3 );

        x4 = _mm512_loadu_pd( src_col4 );
        x5 = _mm512_loadu_pd( src_col5 );
        x6 = _mm512_loadu_pd( src_col6 );
        x7 = _mm512_loadu_pd( src_col7 );

        y0 = _mm512_permutex2var_pd( x0, v_unit1shuffle_mask0, x1 );
        y1 = _mm512_permutex2var_pd( x0, v_unit1shuffle_mask1, x1 );
        y2 = _mm512_permutex2var_pd( x2, v_unit1shuffle_mask0, x3 );
        y3 = _mm512_permutex2var_pd( x2, v_unit1shuffle_mask1, x3 );

        y4 = _mm512_permutex2var_pd( x4, v_unit1shuffle_mask0, x5 );
        y5 = _mm512_permutex2var_pd( x4, v_unit1shuffle_mask1, x5 );
        y6 = _mm512_permutex2var_pd( x6, v_unit1shuffle_mask0, x7 );
        y7 = _mm512_permutex2var_pd( x6, v_unit1shuffle_mask1, x7 );

        x0 = _mm512_permutex2var_pd( y0, v_unit2shuffle_mask0, y2 );
        x1 = _mm512_permutex2var_pd( y0, v_unit2shuffle_mask1, y2 );
        x2 = _mm512_permutex2var_pd( y1, v_unit2shuffle_mask0, y3 );
        x3 = _mm512_permutex2var_pd( y1, v_unit2shuffle_mask1, y3 );

        x4 = _mm512_permutex2var_pd( y4, v_unit2shuffle_mask0, y6 );
        x5 = _mm512_permutex2var_pd( y4, v_unit2shuffle_mask1, y6 );
        x6 = _mm512_permutex2var_pd( y5, v_unit2shuffle_mask0, y7 );
        x7 = _mm512_permutex2var_pd( y5, v_unit2shuffle_mask1, y7 );

        y0 = _mm512_permutex2var_pd( x0, v_unit4shuffle_mask0, x4 );
        y1 = _mm512_permutex2var_pd( x2, v_unit4shuffle_mask0, x6 );
        y2 = _mm512_permutex2var_pd( x1, v_unit4shuffle_mask0, x5 );
        y3 = _mm512_permutex2var_pd( x3, v_unit4shuffle_mask0, x7 );

        y4 = _mm512_permutex2var_pd( x0, v_unit4shuffle_mask1, x4 );
        y5 = _mm512_permutex2var_pd( x2, v_unit4shuffle_mask1, x6 );
        y6 = _mm512_permutex2var_pd( x1, v_unit4shuffle_mask1, x5 );
        y7 = _mm512_permutex2var_pd( x3, v_unit4shuffle_mask1, x7 );

        _mm512_store_pd( pack_b + 0 * 8, y0 );
        _mm512_store_pd( pack_b + 1 * 8, y1 );
        _mm512_store_pd( pack_b + 2 * 8, y2 );
        _mm512_store_pd( pack_b + 3 * 8, y3 );

        _mm512_store_pd( pack_b + 4 * 8, y4 );
        _mm512_store_pd( pack_b + 5 * 8, y5 );
        _mm512_store_pd( pack_b + 6 * 8, y6 );
        _mm512_store_pd( pack_b + 7 * 8, y7 );

        src_b += 8;
        pack_b += 64;
    }
}


/**
 * @brief Pack m_b * k_b of submatrix A
 * 
 */
void pack_a( const double* src_a, \
                   double* pack_a, \
             int lda )
{
    __m512d v_a_col0_row07;
    __m512d v_a_col1_row07;
    __m512d v_a_col2_row07;
    __m512d v_a_col3_row07;

    __m512d v_a_col4_row07;
    __m512d v_a_col5_row07;
    __m512d v_a_col6_row07;
    __m512d v_a_col7_row07;

    // k_i += 8 is where s8 (step 8) come from
    for ( int k_i = 0; k_i < k_c; k_i += 8 )
    {
        const double* src_a_col_0 = src_a + 0 * lda; 
        const double* src_a_col_1 = src_a + 1 * lda;
        const double* src_a_col_2 = src_a + 2 * lda;
        const double* src_a_col_3 = src_a + 3 * lda;

        const double* src_a_col_4 = src_a + 4 * lda; 
        const double* src_a_col_5 = src_a + 5 * lda;
        const double* src_a_col_6 = src_a + 6 * lda;
        const double* src_a_col_7 = src_a + 7 * lda;

        // Move right along x axis by 8
        src_a += 8 * lda; // 8 is from s8

        double* pack_a_0 = pack_a;
        pack_a += 64; // 8x8

        // Process 8x8 block at a time
        // m_i += 8 is where v8 (vector of size 8) come from
        // Iterate down along y axis, this is why called contigious read
        for ( int m_i = 0; m_i < m_c; m_i += 8 )
        {
            // Since we can not require user to align matrix A to 32
            //      the load from a need to be non-aligned version
            // col 0 to 3
            // col 0 to 3
            v_a_col0_row07 = _mm512_loadu_pd( src_a_col_0 );
            v_a_col1_row07 = _mm512_loadu_pd( src_a_col_1 );
            v_a_col2_row07 = _mm512_loadu_pd( src_a_col_2 );
            v_a_col3_row07 = _mm512_loadu_pd( src_a_col_3 );

            _mm512_store_pd( pack_a_0 + 0 * 8, v_a_col0_row07 );
            _mm512_store_pd( pack_a_0 + 1 * 8, v_a_col1_row07 );
            _mm512_store_pd( pack_a_0 + 2 * 8, v_a_col2_row07 );
            _mm512_store_pd( pack_a_0 + 3 * 8, v_a_col3_row07 );

            // col 4 to 7
            v_a_col4_row07 = _mm512_loadu_pd( src_a_col_4 );
            v_a_col5_row07 = _mm512_loadu_pd( src_a_col_5 );
            v_a_col6_row07 = _mm512_loadu_pd( src_a_col_6 );
            v_a_col7_row07 = _mm512_loadu_pd( src_a_col_7 );

            _mm512_store_pd( pack_a_0 + 4 * 8, v_a_col4_row07 );
            _mm512_store_pd( pack_a_0 + 5 * 8, v_a_col5_row07 );
            _mm512_store_pd( pack_a_0 + 6 * 8, v_a_col6_row07 );
            _mm512_store_pd( pack_a_0 + 7 * 8, v_a_col7_row07 );

            // update for next iter
            src_a_col_0 += 8; // 8 is v8
            src_a_col_1 += 8;
            src_a_col_2 += 8;
            src_a_col_3 += 8;

            src_a_col_4 += 8;
            src_a_col_5 += 8;
            src_a_col_6 += 8;
            src_a_col_7 += 8;

            // Move to next m_r * k_c block
            pack_a_0 += 8 * k_c; // 8 is v8
        }
    }
}

/**
 * @brief Allocate require aligned memory.
 * 
 */
inline double* allocate_align_memory( int size, int align_bytes )
{
    void* ptr = NULL;
    posix_memalign( &ptr, align_bytes, size );
    return (double*)ptr;
}


/**
 * @brief DGEMM on KNL Node
 *  A : m * k
 *  B : k * n
 *  C : m * n
 */
void dgemm_knl( int m, int k, int n, \
                const double* src_a, const double* src_b, double* src_c, \
                int lda, int ldb, int ldc )
{
    // Memory for \tilde a and \tilde b
    double* tilde_a = allocate_align_memory( m_b * k_b, 64 );
    double* hat_b   = allocate_align_memory( k_b * n_r, 64 );

    for ( int k_b_i = 0; k_b_i < k / k_b; k_b_i++)
    {
        for ( int m_b_i = 0; m_b_i < m / m_b; m_b_i++ )
        {
            // Pack \tilde a

            for ( int n_r_i = 0; n_r_i < n / n_r; n_r_i++ )
            {
                // Pack \tilde b
                pack_b( src_b, hat_b, ldb ); // TODO: change src_b displacement
                inner_kernel( );
            }
        }
    }
}



/**
 * @brief Benchmark function
 * 
 */
void square_dgemm( int n, double* src_a, double* src_b, double* src_c )
{
    return dgemm_knl( n, n, n, src_a, src_b, src_c, n, n, n );
}
