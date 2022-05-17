#include <cblas.h>

const char* dgemm_desc = "Reference dgemm.";

/*
 * This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values.
 * This function wraps a call to the BLAS-3 routine DGEMM,
 * via the standard FORTRAN interface - hence the reference semantics.
 */
void dgemm_knl( int m, int k, int n, \
                double* src_a, double* src_b, double* src_c, \
                int lda, int ldb, int ldc )
{
    // cblas_dgemm (const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE transa, const CBLAS_TRANSPOSE transb, \
    //                 const MKL_INT m, const MKL_INT n, const MKL_INT k, \
    //                 const double alpha, \
    //                 const double *a, const MKL_INT lda, \
    //                 const double *b, const MKL_INT ldb, \
    //                 const double beta, \
    //                 double *c, const MKL_INT ldc);

    cblas_dgemm( CblasRowMajor, CblasNoTrans, CblasNoTrans, \
                 m, n, k, \
                 1., \
                 src_a, lda, \
                 src_b, ldb, \
                 1., \
                 src_c, ldc );
}
