/* 
 * Copyright (C) 2022 Xiao Song.
 * All Rights Reserved.
 * Content of this file is not for commertial use.
 */

#include <cstdio>
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

void pack_b( double* src_b, double* pak_b, int ldb, int n )
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


int main( int argc, char** argv )
{
    if ( argc != 4 )
    {
        printf("Invalid argv\n");
        exit(1);
    }
    int m = m_b * atoi(argv[1]);
    int k = k_b * atoi(argv[2]);
    int n = n_r * atoi(argv[3]);
    int lda = k;
    int ldb = n;
    int ldc = n;

    double* src_b = (double*)_mm_malloc( n * k   * sizeof( double ), 64 );
    double* pak_b = (double*)_mm_malloc( n * k_b * sizeof( double ), 64 );

    for ( int i = 0; i < n * k; ++i )
    {
        *( src_b + i ) = i;
    }

    // Print source matrix
    printf("\nSRC B\n");
    for ( int row_i = 0; row_i < k; ++row_i )
    {
        if ( row_i % k_b == 0 )
        {
            printf("\n");
        }
        for ( int col_i = 0; col_i < n; ++col_i )
        {
            if ( col_i % n_r == 0 )
            {
                printf("  ");
            }
            printf("%4.0f ", *(src_b + row_i * ldb + col_i));
        }
        printf("\n");
    }

    for ( int k_b_i = 0; k_b_i < k / k_b; k_b_i++)
    {
        // Pack \tilde b
        pack_b( src_b + k_b_i * k_b * ldb, pak_b, ldb, n );

        printf("\nHAT B\n");
        for ( int row_i = 0; row_i < (k_b * n / n_r); ++row_i )
        {
            if ( row_i % k_b == 0 )
            {
                printf("\n");
            }

            for ( int col_i = 0; col_i < n_r; ++col_i )
            {
                printf("%4.0f ", *(pak_b + row_i * n_r + col_i ));
            }
            printf("\n");
        }
    }

    _mm_free( src_b );
    _mm_free( pak_b );
}