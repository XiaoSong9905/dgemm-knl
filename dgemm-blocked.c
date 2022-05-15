#include <immintrin.h>
#include <stdlib.h>

const char* dgemm_desc = "Blocked dgemm.";

#define k_b 1620
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
        _mm_prefetch( hat_a + 12 * m_r, _MM_HINT_T0 );
        _mm_prefetch( hat_b + 32 * n_r , _MM_HINT_T0 );

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
    double* tilde_a = allocate_align_memory( k_b * m_b, 64 );
    double* tilde_b = allocate_align_memory(   n * k_b, 64 );

    for ( int k_b_i = 0; k_b_i < k / k_b; k_b_i++)
    {
        // Pack \tilde b

        for ( int m_b_i = 0; m_b_i < m / m_b; m_b_i++ )
        {
            // Pack \tilde a

            for ( int n_r_i = 0; n_r_i < n / n_r; n_r_i++ )
            {
                for ( int m_r_i = 0; m_r_i < m_b / m_r; m_r_i++ )
                {
                    inner_kernel( );
                }
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
