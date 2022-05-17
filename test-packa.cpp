/* 
 * Copyright (C) 2022 Xiao Song.
 * All Rights Reserved.
 * Content of this file is not for commertial use.
 */

#include <cstdio>
#include <cstdlib>
#include <immintrin.h>

#define n_r 8
#define k_b 24 // NOTE: this is different number compared with actual k_b
#define m_r 31 // NOTE: this is different number compared with actual m_r
#define m_b 31 // NOTE: this is different number compared with actual m_b

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


void pack_a( double* src_a, double* pak_a, int lda )
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

int main( int argc, char** argv )
{
    if ( argc != 4 )
    {
        printf("Invalid argv\n");
        return -1;
    }
    int m = m_b * atoi(argv[1]);
    int k = k_b * atoi(argv[2]);
    int n = n_r * atoi(argv[3]);
    int lda = k;
    int ldb = n;
    int ldc = n;

    double* src_a = (double*)_mm_malloc( m   * k   * sizeof( double ), 64 );
    double* pak_a = (double*)_mm_malloc( m_b * k_b * sizeof( double ), 64 );

    for ( int i = 0; i < m * k; ++i )
    {
        *( src_a + i ) = i;
    }

    // Print source matrix
    printf("\nSRC A\n");
    for ( int row_i = 0; row_i < m; ++row_i )
    {
        if ( row_i % m_r == 0 )
        {
            printf("\n");
        }
        for ( int col_i = 0; col_i < k; ++col_i )
        {
            if ( col_i % k_b == 0 )
            {
                printf("  ");
            }
            printf("%4.0f ", *(src_a + row_i * lda + col_i ));
        }
        printf("\n");
    }

    for ( int k_b_i = 0; k_b_i < k / k_b; k_b_i++)
    {
        for ( int m_b_i = 0; m_b_i < m / m_b; m_b_i++ )
        {
            // Pack \tilde a
            pack_a( src_a + m_b_i * m_b * lda + k_b_i * k_b, pak_a, lda );

            // Print hat A
            // hat A is stored in column major order
            printf("\nHAT A\n");
            for ( int col_i = 0; col_i < (k_b * m_b / m_r); ++col_i )
            {
                if ( col_i % k_b == 0 )
                {
                    printf("\n");
                }

                for ( int row_i = 0; row_i < m_r; ++row_i )
                {
                    printf("%4.0f ", *(pak_a + col_i * m_r + row_i ));
                }
                printf("\n");
            }
        }
    }

    _mm_free( src_a );
    _mm_free( pak_a );
}