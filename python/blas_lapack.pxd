cimport numpy as np

#
# BLAS
#

ctypedef int dgemm_t(
    # Compute C := alpha*A*B + beta*C
    char *transa,        # {'T','C'}: o(A)=A'; {'N'}: o(A)=A
    char *transb,        # {'T','C'}: o(B)=B'; {'N'}: o(B)=B
    int *m,              # Rows of o(A)    (and of C)
    int *n,              # Columns of o(B) (and of C)
    int *k,              # Columns of o(A) / Rows of o(B)
    np.float64_t *alpha, # Scalar multiple
    np.float64_t *a,     # Matrix A: mxk
    int *lda,            # The size of the first dimension of A (in memory)
    np.float64_t *b,     # Matrix B: kxn
    int *ldb,            # The size of the first dimension of B (in memory)
    np.float64_t *beta,  # Scalar multiple
    np.float64_t *c,     # Matrix C: mxn
    int *ldc             # The size of the first dimension of C (in memory)
)

ctypedef int dgemv_t(
    # Compute y := alpha*A*x + beta*y
    char *trans,         # {'T','C'}: o(A)=A'; {'N'}: o(A)=A
    int *m,              # Rows of A (prior to transpose from *trans)
    int *n,              # Columns of A / min(len(x))
    np.float64_t *alpha, # Scalar multiple
    np.float64_t *a,     # Matrix A: mxn
    int *lda,            # The size of the first dimension of A (in memory)
    np.float64_t *x,     # Vector x, min(len(x)) = n
    int *incx,           # The increment between elements of x (usually 1)
    np.float64_t *beta,  # Scalar multiple
    np.float64_t *y,     # Vector y, min(len(y)) = m
    int *incy            # The increment between elements of y (usually 1)
)

ctypedef int dsymm_t(
    # Compute C := alpha*A*B + beta*C,
    char *side,    # {'L','R'}
    char *uplo,    # {'U','L'}
    int *m,        # Rows of C
    int *n,        # Columns C
    double *alpha, # Scalar multiple
    double *a,     # Matrix A: mxk
    int *lda,      # The size of the first dimension of A (in memory)
    double *b,     # Matrix B: kxn
    int *ldb,      # The size of the first dimension of B (in memory)
    double *beta,  # Scalar multiple
    double *c,     # Matrix C: mxn
    int *ldc       # The size of the first dimension of C (in memory)
)

ctypedef int dsymv_t(
    # y := alpha*A*x + beta*y, where A is symmetric
    # Compute C := alpha*A*B + beta*C,
    char *uplo,    # {'U','L'}
    int *n,        # order of A
    double *alpha, # Scalar multiple
    double *a,     # Matrix A: mxk
    int *lda,      # The size of the first dimension of A (in memory)
    double *x,     # Vector x, min(len(x)) = n
    int *incx,     # The increment between elements of x (usually 1)
    double *beta,  # Scalar multiple
    double *y,     # Vector y, min(len(y)) = m
    int *incy      # The increment between elements of y (usually 1)
)

ctypedef int dtrtrs_t(
    # op( A )*X = B
    # Compute C := alpha*A*B + beta*C,
    char *uplo,    # {'U','L'} is A upper or lower triangular
    char *trans,   # {'N','T','C'} specifies op(A)
    char *diag,    # {'U','N'} is A unit triangular?
    int *n,        # order of A
    int *nrhs,     # columns of B
    double *a,     # Matrix A: lda x n
    int *lda,      # The size of the first dimension of A (in memory)
    double *b,     # Matrix B: ldb x NRHS
    int *ldb,      # The size of the first dimension of B (in memory)
    int *info,     # 0 if success, otherwise an error code (integer)
)

ctypedef int dtrmm_t(
    # DTRMM - perform one of the matrix-matrix operations   B :=
    # alpha*op( A )*B, or B := alpha*B*op( A ),
    char *side,           # {'L', 'R'}: left, right
    char *uplo,           # {'U','L'}, upper, lower
    char *transa,         # {'T','C'}: o(A)=A'; {'N'}: o(A)=A
    char *diag,           # {'U','N'}: unit triangular or not
    int *m,               # Rows of B
    int *n,               # Columns of B
    np.float64_t *alpha,  # Scalar multiple
    np.float64_t *a,      # Matrix A
    int *lda,             # The size of the first dimension of A (in memory)
    np.float64_t *b,      # Matrix B
    int *ldb,             # The size of the first dimension of B (in memory)
)

ctypedef int dtrmv_t(
    # DTRMV - perform one of the matrix-vector operations   x :=
    # A*x, or x := A'*x,
    char *uplo,           # {'U','L'}, upper, lower
    char *trans,          # {'T','C'}: o(A)=A'; {'N'}: o(A)=A
    char *diag,           # {'U','N'}: unit triangular or not
    int *n,               # Order of matrix A
    np.float64_t *a,      # Matrix A: mxn
    int *lda,             # The size of the first dimension of A (in memory)
    np.float64_t *x,      # Vector x, min(len(x)) = n
    int *incx,            # The increment between elements of x (usually 1)
)

ctypedef int dsyrk_t(
    # DSYRK - perform one of the symmetric rank k operations   C
    # := alpha*A*A' + beta*C,
    char *uplo,           # {'U','L'}, upper, lower
    char *trans,          # {'T','C'}: o(A)=A'; {'N'}: o(A)=A
    int *n,               # Order of matrix C
    int *k,               # 'T' => rows of A; 'N' => cols of A
    np.float64_t *alpha,  # Scalar multiple
    np.float64_t *a,      # Matrix A
    int *lda,             # The size of the first dimension of A (in memory)
    np.float64_t *beta,  # Scalar multiple
    np.float64_t *c,      # Matrix C
    int *ldc,             # The size of the first dimension of C (in memory)
)

ctypedef int dcopy_t(
    int *n,              # Number of vector elements to be copied.
    np.float64_t *x,     # Vector from which to copy.
    int *incx,           # Increment between elements of x.
    np.float64_t *y,     # array of dimension (n-1) * |incy| + 1, result vector.
    int *incy            # Increment between elements of y.
)

ctypedef int dscal_t(
    # DSCAL - BLAS level one, scales a double precision vector
    int *n,               # Number of elements in the vector.
    np.float64_t *alpha,  # scalar alpha
    np.float64_t *x,      # Array of dimension (n-1) * |incx| + 1. Vector to be scaled.
    int *incx             # Increment between elements of x.
)

ctypedef int daxpy_t(
    # Compute y := alpha*x + y
    int *n,              # Columns of o(A) / min(len(x))
    np.float64_t *alpha, # Scalar multiple
    np.float64_t *x,     # Vector x, min(len(x)) = n
    int *incx,           # The increment between elements of x (usually 1)
    np.float64_t *y,     # Vector y, min(len(y)) = m
    int *incy            # The increment between elements of y (usually 1)
)

ctypedef double ddot_t(
    # Compute DDOT := x.T * y
    int *n,              # Length of vectors
    np.float64_t *x,     # Vector x, min(len(x)) = n
    int *incx,           # The increment between elements of x (usually 1)
    np.float64_t *y,     # Vector y, min(len(y)) = m
    int *incy            # The increment between elements of y (usually 1)
)

#
# LAPACK
#

ctypedef int dgetrf_t(
    # DGETRF - compute an LU factorization of a general M-by-N
    # matrix A using partial pivoting with row interchanges
    int *m,          # Rows of A
    int *n,          # Columns of A
    np.float64_t *a, # Matrix A: mxn
    int *lda,        # The size of the first dimension of A (in memory)
    int *ipiv,       # Matrix P: mxn (the pivot indices)
    int *info        # 0 if success, otherwise an error code (integer)
)

ctypedef int dgetri_t(
    # DGETRI - compute the inverse of a matrix using the LU fac-
    # torization computed by DGETRF
    int *n,              # Order of A
    np.float64_t *a,     # Matrix A: nxn (the LUP decomposed matrix from dgetrf)
    int *lda,            # The size of the first dimension of A (in memory)
    int *ipiv,           # Matrix P: nxn (the pivot indices from the LUP decomposition)
    np.float64_t *work,  # Matrix: nxn (a workspace for the inversion, optimal size=nxn)
    int *lwork,          # Number of elements in the workspace: optimal is n**2
    int *info            # 0 if success, otherwise an error code (integer)
)

ctypedef int dgetrs_t(
    # DGETRS - solve a system of linear equations  A * X = B or A'
    # * X = B with a general N-by-N matrix A using the LU factori-
    # zation computed by DGETRF
    char *trans,        # Specifies the form of the system of equations
    int *n,             # Order of A
    int *nrhs,          # The number of right hand sides
    np.float64_t *a,    # Matrix A: nxn (the LUP decomposed matrix from dgetrf)
    int *lda,           # The size of the first dimension of A (in memory)
    int *ipiv,          # Matrix P: nxn (the pivot indices from the LUP decomposition)
    np.float64_t *b,    # Matrix B: nxnrhs
    int *ldb,           # The size of the first dimension of B (in memory)
    int *info           # 0 if success, otherwise an error code (integer)
)

ctypedef int dpotrf_t(
    # Compute the Cholesky factorization of a
    # real  symmetric positive definite matrix A
    char *uplo,      # 'U':  A = U'U and U is stored, 'L': A = LL' and L is stored
    int *n,          # The order of the matrix A.  n >= 0.
    np.float64_t *a, # Matrix A: nxn
    int *lda,        # The size of the first dimension of A (in memory)
    int *info        # 0 if success, otherwise an error code (integer)
)

ctypedef int dpotri_t(
    # DPOTRI - compute the inverse of a real symmetric positive
    # definite matrix A using the Cholesky factorization A =
    # U**T*U or A = L*L**T computed by DPOTRF
    char *uplo,      # 'U':  A = U'U and U is stored, 'L': A = LL' and L is stored
    int *n,          # The order of the matrix A.  n >= 0.
    np.float64_t *a, # Matrix A: nxn
    int *lda,        # The size of the first dimension of A (in memory)
    int *info        # 0 if success, otherwise an error code (integer)
)

ctypedef int dpotrs_t(
    # DPOTRS - solve a system of linear equations A*X = B with a
    # symmetric positive definite matrix A using the Cholesky fac-
    # torization A = U**T*U or A = L*L**T computed by DPOTRF
    char *uplo,       # 'U':  A = U'U and U is stored, 'L': A = LL' and L is stored
    int *n,           # The order of the matrix A.  n >= 0.
    int *nrhs,        # The number of right hand sides
    np.float64_t *a,  # Matrix A: nxn
    int *lda,         # The size of the first dimension of A (in memory)
    np.float64_t *b,  # Matrix B: nxnrhs
    int *ldb,         # The size of the first dimension of B (in memory)
    int *info         # 0 if success, otherwise an error code (integer)
)