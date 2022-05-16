#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include <cmath> // For: fabs

#include <cblas.h>

#ifndef MAX_SPEED
#error "Must set max speed with -DMAX_SPEED=... or similar"
#endif

extern "C" 
{

/* Your function must have the following signature: */
extern const char* dgemm_desc;
extern void dgemm_knl( int, int, int, double*, double*, double*, int, int, int );

}


void reference_dgemm( int m, int k, int n, double alpha, double* src_a, double* src_b, double* src_c, int lda, int ldb, int ldc ) 
{
    cblas_dgemm( CblasColMajor, CblasNoTrans, CblasNoTrans, \
                 m, n, k,  \
                 alpha, \
                 src_a, m, \
                 src_b, k, \
                 1., \
                 src_c, m );
}


void fill(double* p, int n) 
{
    static std::random_device rd;
    static std::default_random_engine gen(rd());
    static std::uniform_real_distribution<> dis(-1.0, 1.0);
    for (int i = 0; i < n; ++i)
        p[i] = 2 * dis(gen) - 1;
}


/* The benchmarking program */
int main(int argc, char** argv) 
{
    std::cout << "Description:\t" << dgemm_desc << std::endl << std::endl;
    std::cout << std::fixed << std::setprecision(2);

    std::vector<std::vector<int>> test_sizes
    {
        // m,k,n
        std::vector<int>{ 78*31, 1*1624, 300*8 },
        std::vector<int>{ 78*31, 2*1624, 300*8 },
        std::vector<int>{ 156*31, 3*1624, 605*8 },
    };

    int nsizes = test_sizes.size();

    // NOTE: need to always double check every dim of test_sizes is less than 5000
    int nmax = 5000;

    /* allocate memory for all problems */
    std::vector<double> buf(3 * nmax * nmax);
    std::vector<double> per;

    /* For each test size */
    for ( auto test_size_i : test_sizes )
    {
        /* Create and fill 3 random matrices A,B,C*/
        int m = test_size_i[ 0 ];
        int k = test_size_i[ 1 ];
        int n = test_size_i[ 2 ];

        double* A = buf.data() + 0;
        double* B = A + nmax * nmax;
        double* C = B + nmax * nmax;

        fill( A, m * k );
        fill( B, k * n );
        fill( C, m * n );

        /* Measure performance (in Gflops/s). */

        /* Time a "sufficiently long" sequence of calls to reduce noise */
        double Gflops_s = 0.0, seconds = -1.0;
        double timeout = 0.1; // "sufficiently long" := at least 1/10 second.
        for (int n_iterations = 1; seconds < timeout; n_iterations *= 2) 
        {
            /* Warm-up */
            dgemm_knl( m, k, n, A, B, C, m, k, m );

            /* Benchmark n_iterations runs of square_dgemm */
            auto start = std::chrono::steady_clock::now();
            for (int it = 0; it < n_iterations; ++it) 
            {
                dgemm_knl( m, k, n, A, B, C, m, k, m );
            }
            auto end = std::chrono::steady_clock::now();
            std::chrono::duration<double> diff = end - start;
            seconds = diff.count();

            /*  compute Gflop/s rate */
            Gflops_s = 2.e-9 * n_iterations * m * n * k / seconds;
        }

        /* Storing Mflop rate and calculating percentage of peak */
        double Mflops_s = Gflops_s * 1000;
        per.push_back(Gflops_s * 100 / MAX_SPEED);

        std::cout << "Size: " << n                  //
                  << "\tMflops/s: " << Mflops_s     //
                  << "\tPercentage: " << per.back() //
                  << std::endl;
       
        /* Ensure that error does not exceed the theoretical error bound. */

        /* C := A * B, computed with square_dgemm */
        std::fill(C, &C[m * n], 0.0);
        dgemm_knl( m, k, n, A, B, C, m, k, m );

        /* Do not explicitly check that A and B were unmodified on square_dgemm exit
         *  - if they were, the following will most likely detect it:
         * C := C - A * B, computed with reference_dgemm */
        reference_dgemm( m, k, n, -1, A, B, C, m, k, m );

        /* A := |A|, B := |B|, C := |C| */
        std::transform(A, &A[m * k], A, fabs);
        std::transform(B, &B[k * n], B, fabs);
        std::transform(C, &C[m * n], C, fabs);

        /* C := |C| - 3 * e_mach * n * |A| * |B|, computed with reference_dgemm */
        const auto e_mach = std::numeric_limits<double>::epsilon();
        reference_dgemm( m, k, n, -3. * e_mach * std::max(n, m), A, B, C, m, k, m );

        /* If any element in C is positive, then something went wrong in square_dgemm */
        for (int i = 0; i < m * n; ++i) 
        {
            if (C[i] > 0) 
            {
                std::cerr << "*** FAILURE *** Error in matrix multiply exceeds componentwise error "
                             "bounds."
                          << std::endl;
                return 1;
            }
        }
    }

    /* Calculating average percentage of peak reached by algorithm */
    double aveper = 0;
    for (int i = 0; i < nsizes; i++) {
        aveper += per[i];
    }
    aveper /= nsizes;

    /* Printing average percentage to screen */
    std::cout << "Average percentage of Peak = " << aveper << std::endl;

    return 0;
}