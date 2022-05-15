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

extern "C" {
/* Your function must have the following signature: */
extern const char* dgemm_desc;

extern void square_dgemm(int, double*, double*, double*);
}

void reference_dgemm(int n, double alpha, double* A, double* B, double* C) {
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, alpha, A, n, B, n, 1., C, n);
}

void fill(double* p, int n) {
    static std::random_device rd;
    static std::default_random_engine gen(rd());
    static std::uniform_real_distribution<> dis(-1.0, 1.0);
    for (int i = 0; i < n; ++i)
        p[i] = 2 * dis(gen) - 1;
}

/* The benchmarking program */
int main(int argc, char** argv) {
    std::cout << "Description:\t" << dgemm_desc << std::endl << std::endl;

    std::cout << std::fixed << std::setprecision(2);

    /* Test sizes should highlight performance dips at multiples of certain powers-of-two */
#define ALL_SIZES
#ifdef ALL_SIZES
    /* Multiples-of-32, +/- 1. */
    std::vector<int> test_sizes{
        31,  32,  33,  63,  64,  65,  95,  96,  97,  127, 128, 129, 159, 160,  161,  191,
        192, 193, 223, 224, 225, 255, 256, 257, 287, 288, 289, 319, 320, 321,  351,  352,
        353, 383, 384, 385, 415, 416, 417, 447, 448, 449, 479, 480, 481, 511,  512,  513,
        543, 544, 545, 575, 576, 577, 607, 608, 609, 639, 640, 641, 671, 672,  673,  703,
        704, 705, 735, 736, 737, 767, 768, 769, 799, 800, 801, 831, 832, 833,  863,  864,
        865, 895, 896, 897, 927, 928, 929, 959, 960, 961, 991, 992, 993, 1023, 1024, 1025};
#else
    /* A representative subset of the first list. */
    std::vector<int> test_sizes{31,  32,  96,  97,  127, 128, 129, 191, 192, 229, 255, 256, 257, \
                                319, 320, 321, 417, 479, 480, 511, 512, 639, 640, 767, 768, 769};

#endif

    if (argc > 1) {
        test_sizes.clear();
        std::transform(&argv[1], &argv[argc], std::back_inserter(test_sizes), [](char* arg) {
            size_t end;
            int size = std::stoi(arg, &end);
            if (arg[end] != '\0' || size < 1) {
                throw std::invalid_argument("all arguments must be positive numbers");
            }
            return size;
        });
    }

    std::sort(test_sizes.begin(), test_sizes.end());

    int nsizes = test_sizes.size();

    /* assume last size is also the largest size */
    int nmax = test_sizes[nsizes - 1];

    /* allocate memory for all problems */
    std::vector<double> buf(3 * nmax * nmax);
    std::vector<double> per;

    /* For each test size */
    for (int n : test_sizes) {
        //printf("@ test for size %d\n", n );
        /* Create and fill 3 random matrices A,B,C*/
        double* A = buf.data() + 0;
        double* B = A + nmax * nmax;
        double* C = B + nmax * nmax;

        fill(A, n * n);
        fill(B, n * n);
        fill(C, n * n);

        /* Measure performance (in Gflops/s). */

        /* Time a "sufficiently long" sequence of calls to reduce noise */
        double Gflops_s = 0.0, seconds = -1.0;
        double timeout = 0.1; // "sufficiently long" := at least 1/10 second.
        for (int n_iterations = 1; seconds < timeout; n_iterations *= 2) {
            /* Warm-up */
            square_dgemm(n, A, B, C);

            /* Benchmark n_iterations runs of square_dgemm */
            auto start = std::chrono::steady_clock::now();
            for (int it = 0; it < n_iterations; ++it) {
                square_dgemm(n, A, B, C);
            }
            auto end = std::chrono::steady_clock::now();
            std::chrono::duration<double> diff = end - start;
            seconds = diff.count();

            /*  compute Gflop/s rate */
            Gflops_s = 2.e-9 * n_iterations * n * n * n / seconds;
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
        std::fill(C, &C[n * n], 0.0);
        square_dgemm(n, A, B, C);

        /* Do not explicitly check that A and B were unmodified on square_dgemm exit
         *  - if they were, the following will most likely detect it:
         * C := C - A * B, computed with reference_dgemm */
        //printf("@ before ref \n" );fflush( stdout );
        reference_dgemm(n, -1., A, B, C);
        //printf("@ after ref \n" ); fflush( stdout );

        /* A := |A|, B := |B|, C := |C| */
        std::transform(A, &A[n * n], A, fabs);
        std::transform(B, &B[n * n], B, fabs);
        std::transform(C, &C[n * n], C, fabs);

        /* C := |C| - 3 * e_mach * n * |A| * |B|, computed with reference_dgemm */
        const auto e_mach = std::numeric_limits<double>::epsilon();
        reference_dgemm(n, -3. * e_mach * n, A, B, C);

        /* If any element in C is positive, then something went wrong in square_dgemm */
        for (int i = 0; i < n * n; ++i) {
            if (C[i] > 0) {
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