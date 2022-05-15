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
void square_dgemm(int n, double* A, double* B, double* C) {
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1., A, n, B, n, 1., C, n);
}
