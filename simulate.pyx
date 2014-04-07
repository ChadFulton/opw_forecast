import numpy as np
from scipy.misc import comb
from scipy import stats
cimport numpy as np
cimport cython
from rtnorm.pyrtnorm import rtnorm

from cpython cimport PyCObject_AsVoidPtr
import scipy
__import__('scipy.linalg.blas')
__import__('scipy.linalg.lapack')

from libc.math cimport sqrt, exp, log

from blas_lapack cimport *

cdef dsymm_t *dsymm = <dsymm_t*>PyCObject_AsVoidPtr(scipy.linalg.blas.dsymm._cpointer)
cdef dsymv_t *dsymv = <dsymv_t*>PyCObject_AsVoidPtr(scipy.linalg.blas.dsymv._cpointer)
cdef dgemm_t *dgemm = <dgemm_t*>PyCObject_AsVoidPtr(scipy.linalg.blas.dgemm._cpointer)
cdef dgemv_t *dgemv = <dgemv_t*>PyCObject_AsVoidPtr(scipy.linalg.blas.dgemv._cpointer)
cdef dcopy_t *dcopy = <dcopy_t*>PyCObject_AsVoidPtr(scipy.linalg.blas.dcopy._cpointer)
cdef daxpy_t *daxpy = <daxpy_t*>PyCObject_AsVoidPtr(scipy.linalg.blas.daxpy._cpointer)
cdef ddot_t *ddot = <ddot_t*>PyCObject_AsVoidPtr(scipy.linalg.blas.ddot._cpointer)
cdef dgetrf_t *dgetrf = <dgetrf_t*>PyCObject_AsVoidPtr(scipy.linalg.lapack.dgetrf._cpointer)
cdef dgetri_t *dgetri = <dgetri_t*>PyCObject_AsVoidPtr(scipy.linalg.lapack.dgetri._cpointer)
cdef dpotrf_t *dpotrf = <dpotrf_t*>PyCObject_AsVoidPtr(scipy.linalg.lapack.dpotrf._cpointer)

cpdef int [:] draw_gamma(int [:] gamma, int rvs):
    cdef:
        int [:] gamma_star
        int n = gamma.shape[0]

    gamma_star = np.copy(gamma)

    if rvs > 0:
        if gamma_star[rvs] == 1:
            gamma_star[rvs] = 0
        else:
            gamma_star[rvs] = 1
    return gamma_star

cpdef int get_draw_rho_NB(double [::1, :] M0,  # k_gamma x k_gamma
                          double [:] endog,    # T-h x 0
                          double [::1,:] exog, # T-h x k_gamma
                          double [:] rvs):     # k_gamma x 0
    cdef:
        int T_h = exog.shape[0]
        int k_gamma = exog.shape[1] # in terms of OPW this value is k_gamma+1
        int k_gamma2 = k_gamma**2
    cdef:
        double [::1, :] M1
        double [:] m1, tmp
    cdef:
        double alpha = 1.0
        double beta = 0.0
        double delta = -1.0
        int inc = 1
        int i,j
    cdef:
        double [::1,:] work
        int [::1,:] ipiv
        int lwork = k_gamma   # size of work array for dgetri
        int info
    
    M1 = np.empty((k_gamma,k_gamma), float, order="F")
    m1 = np.empty((k_gamma,), float, order="F")
    tmp = np.empty((k_gamma,), float, order="F")
    work = np.empty((lwork,lwork), float, order="F")
    ipiv = np.zeros((k_gamma,k_gamma), np.int32, order="F")
    
    # Posterior covariance matrix
    # M1 = M0
    dcopy(&k_gamma2, &M0[0,0], &inc, &M1[0,0], &inc)
    # M1 = M1 + exog.T.dot(exog)
    dgemm("T", "N", &k_gamma, &k_gamma, &T_h, &alpha, &exog[0,0], &T_h, &exog[0,0], &T_h, &alpha, &M1[0,0], &k_gamma)
    # M1 = np.linalg.inv(M1)
    # TODO cache?
    dgetrf(&k_gamma, &k_gamma, &M1[0,0], &k_gamma, &ipiv[0,0], &info)
    dgetri(&k_gamma, &M1[0,0], &k_gamma, &ipiv[0,0], &work[0,0], &lwork, &info)

    return <int>work[0,0]

cpdef double [:] draw_rho(double [::1, :] M0,  # k_gamma x k_gamma
                          double [:] endog,    # T-h x 0
                          double [::1,:] exog, # T-h x k_gamma
                          double [:] rvs,
                          double [::1,:] work):     # k_gamma x 0
    cdef:
        int T_h = exog.shape[0]
        int k_gamma = exog.shape[1] # in terms of OPW this value is k_gamma+1
        int k_gamma2 = k_gamma**2
    cdef:
        double [::1, :] M1
        double [:] m1, tmp
    cdef:
        double alpha = 1.0
        double beta = 0.0
        double delta = -1.0
        int inc = 1
        int i,j
    cdef:
        #double [::1,:] work
        int [::1,:] ipiv
        #int ldwork = k_gamma**2 # dimension of work array for dgetri
        int lwork = work.shape[0]   # size of work array for dgetri
        int info
    
    M1 = np.empty((k_gamma,k_gamma), float, order="F")
    m1 = np.empty((k_gamma,), float, order="F")
    tmp = np.empty((k_gamma,), float, order="F")
    #work = np.empty((ldwork,ldwork), float, order="F")
    ipiv = np.zeros((k_gamma,k_gamma), np.int32, order="F")
    
    # Posterior covariance matrix
    # M1 = M0
    dcopy(&k_gamma2, &M0[0,0], &inc, &M1[0,0], &inc)
    # M1 = M1 + exog.T.dot(exog)
    dgemm("T", "N", &k_gamma, &k_gamma, &T_h, &alpha, &exog[0,0], &T_h, &exog[0,0], &T_h, &alpha, &M1[0,0], &k_gamma)
    # M1 = np.linalg.inv(M1)
    # TODO cache?
    dgetrf(&k_gamma, &k_gamma, &M1[0,0], &k_gamma, &ipiv[0,0], &info)
    dgetri(&k_gamma, &M1[0,0], &k_gamma, &ipiv[0,0], &work[0,0], &lwork, &info)
    
    # Posterior mean
    #m1 = M1.dot(exog.T.dot(endog))
    dgemv("T", &T_h, &k_gamma, &alpha, &exog[0,0], &T_h, &endog[0], &inc, &beta, &tmp[0], &inc)
    dgemv("N", &k_gamma, &k_gamma, &alpha, &M1[0,0], &k_gamma, &tmp[0], &inc, &beta, &m1[0], &inc)
    
    # Transform the variate from standard normal to the actual posterior
    # M1 = np.linalg.cholesky(M1) # TODO can probably prefetch/cache these too
    dpotrf("U", &k_gamma, &M1[0,0], &k_gamma, &info)
    for i in range(k_gamma):
        for j in range(0,i):
            M1[i,j] = 0

    # m1 = m1 + M1.dot(rvs)
    dgemv("T", &k_gamma, &k_gamma, &alpha, &M1[0,0], &k_gamma, &rvs[0], &inc, &alpha, &m1[0], &inc)

    return m1

cpdef int get_mvn_density_NB(double [::1, :] M0,   # T-h x T-h
                             double sigma2,
                             double [:] endog,     # T-h x 0
                             double [::1,:] exog): # T-h x k_gamma
    cdef:
        int T_h = exog.shape[0]
        int T_h2 = T_h**2
        int k_gamma = exog.shape[1] # in terms of OPW this value is k_gamma+1
    cdef:
        double [::1, :] Sigma
        double [:] tmp
        double val
    cdef:
        double alpha = 1.0
        double beta = 0.0
        double delta = -1.0
        int inc = 1
    cdef:
        double [::1,:] work
        int [::1,:] ipiv
        int lwork = T_h   # size of work array for dgetri
        int info
        double det
        int i
    
    Sigma = np.empty((T_h,T_h), float, order="F")
    tmp = np.empty((T_h,), float, order="F")
    work = np.empty((lwork,lwork), float, order="F")
    ipiv = np.zeros((T_h,T_h), np.int32, order="F")

    # Sigma = M0
    dcopy(&T_h2, &M0[0,0], &inc, &Sigma[0,0], &inc)

    # Sigma = (np.eye(exog.shape[0]) + sigma2*exog.dot(exog.T))/sigma2
    #       = Sigma + sigma2*exog.dot(exog.T)
    dgemm("N", "T", &T_h, &T_h, &k_gamma, &sigma2, &exog[0,0], &T_h, &exog[0,0], &T_h, &alpha, &Sigma[0,0], &T_h)

    # det = np.linalg.det(Sigma)
    # Sigma = np.linalg.inv(Sigma)
    dgetrf(&T_h, &T_h, &Sigma[0,0], &T_h, &ipiv[0,0], &info)
    det = 1
    for i in range(T_h):
        if not ipiv[i,0] == i+1:
            det *= -1*Sigma[i,i]
        else:
            det *= Sigma[i,i]
    det = sqrt(det)
    dgetri(&T_h, &Sigma[0,0], &T_h, &ipiv[0,0], &work[0,0], &lwork, &info)

    return <int>work[0,0]

cpdef mvn_density(double [::1, :] M0,   # T-h x T-h
                         double sigma2,
                         double [:] endog,     # T-h x 0
                         double [::1,:] exog,  # T-h x k_gamma
                         double [::1,:] work,
                         key, cache):
    cdef:
        int T_h = exog.shape[0]
        int T_h2 = T_h**2
        int k_gamma = exog.shape[1] # in terms of OPW this value is k_gamma+1
    cdef:
        double [::1, :] Sigma
        double [:] tmp
        double val
    cdef:
        double alpha = 1.0
        double beta = 0.0
        double delta = -1.0
        int inc = 1
    cdef:
        #double [::1,:] work
        int [::1,:] ipiv
        #int lwork = T_h*NB   # dimension of work array for dgetri
        int lwork = work.shape[0]
        int info
        double det
        int i
    
    Sigma = np.empty((T_h,T_h), float, order="F")
    tmp = np.empty((T_h,), float, order="F")
    #work = np.empty((lwork,lwork), float, order="F")
    ipiv = np.zeros((T_h,T_h), np.int32, order="F")

    if key in cache:
        Sigma = cache[key]['Sigma']
        det = cache[key]['det']
        cache[key]['count'] += 1
    else:
        # Sigma = M0
        dcopy(&T_h2, &M0[0,0], &inc, &Sigma[0,0], &inc)

        # Sigma = (np.eye(exog.shape[0]) + sigma2*exog.dot(exog.T))/sigma2
        #       = Sigma + sigma2*exog.dot(exog.T)
        dgemm("N", "T", &T_h, &T_h, &k_gamma, &sigma2, &exog[0,0], &T_h, &exog[0,0], &T_h, &alpha, &Sigma[0,0], &T_h)

        # det = np.linalg.det(Sigma)
        # Sigma = np.linalg.inv(Sigma)
        dgetrf(&T_h, &T_h, &Sigma[0,0], &T_h, &ipiv[0,0], &info)
        det = 1
        for i in range(T_h):
            if not ipiv[i,0] == i+1:
                det *= -1*Sigma[i,i]
            else:
                det *= Sigma[i,i]
        dgetri(&T_h, &Sigma[0,0], &T_h, &ipiv[0,0], &work[0,0], &lwork, &info)
        cache[key] = {
            'Sigma':Sigma,
            'det':det,
            'count':0
        }

    # tmp = Sigma.dot(endog)
    #dgemv("N", &T_h, &T_h, &alpha, &Sigma[0,0], &T_h, &endog[0], &inc, &beta, &tmp[0], &inc)
    dsymv("U", &T_h, &alpha, &Sigma[0,0], &T_h, &endog[0], &inc, &beta, &tmp[0], &inc)

    return exp(-0.5*ddot(&T_h, &endog[0], &inc, &tmp[0], &inc))/sqrt(det)

cpdef ln_mvn_density(double [::1, :] M0,   # T-h x T-h
                         double sigma2,
                         double [:] endog,     # T-h x 0
                         double [::1,:] exog,  # T-h x k_gamma
                         double [::1,:] work,  # T-h x k_gamma
                         key, cache):
    cdef:
        int T_h = exog.shape[0]
        int T_h2 = T_h**2
        int k_gamma = exog.shape[1] # in terms of OPW this value is k_gamma+1
    cdef:
        double [::1, :] Sigma
        double [:] tmp
        double val
    cdef:
        double alpha = 1.0
        double beta = 0.0
        double delta = -1.0
        int inc = 1
    cdef:
        #double [::1,:] work
        int [::1,:] ipiv
        #int lwork = T_h*NB   # dimension of work array for dgetri
        int lwork = work.shape[0]
        int info
        double det
        int i
    
    Sigma = np.empty((T_h,T_h), float, order="F")
    tmp = np.empty((T_h,), float, order="F")
    #work = np.empty((lwork,lwork), float, order="F")
    ipiv = np.zeros((T_h,T_h), np.int32, order="F")

    if key in cache:
        Sigma = cache[key]['Sigma']
        det = cache[key]['det']
        cache[key]['count'] += 1
    else:
        # Sigma = M0
        dcopy(&T_h2, &M0[0,0], &inc, &Sigma[0,0], &inc)

        # Sigma = (np.eye(exog.shape[0]) + sigma2*exog.dot(exog.T))/sigma2
        #       = Sigma + sigma2*exog.dot(exog.T)
        dgemm("N", "T", &T_h, &T_h, &k_gamma, &sigma2, &exog[0,0], &T_h, &exog[0,0], &T_h, &alpha, &Sigma[0,0], &T_h)

        # det = np.linalg.det(Sigma)
        # Sigma = np.linalg.inv(Sigma)
        dgetrf(&T_h, &T_h, &Sigma[0,0], &T_h, &ipiv[0,0], &info)
        det = 1
        for i in range(T_h):
            if not ipiv[i,0] == i+1:
                det *= -1*Sigma[i,i]
            else:
                det *= Sigma[i,i]
        dgetri(&T_h, &Sigma[0,0], &T_h, &ipiv[0,0], &work[0,0], &lwork, &info)
        cache[key] = {
           'Sigma':Sigma,
           'det':det,
           'count':0
        }

    # tmp = Sigma.dot(endog)
    #dgemv("N", &T_h, &T_h, &alpha, &Sigma[0,0], &T_h, &endog[0], &inc, &beta, &tmp[0], &inc)
    dsymv("U", &T_h, &alpha, &Sigma[0,0], &T_h, &endog[0], &inc, &beta, &tmp[0], &inc)

    return - 0.5*log(det) - 0.5*ddot(&T_h, &endog[0], &inc, &tmp[0], &inc)

cpdef mn_mass(gamma):
    return 1/comb(gamma.shape[0], np.sum(gamma))

cpdef ln_mn_mass(gamma):
    return log(mn_mass(gamma))

cpdef draw_y(double [:] rho,       # k_gamma x 0
             int [:] s,            # T_h x 0
             double [::1,:] exog,  # T_h x k_gamma
             double [::1,:] rvs):  # T_h x I
    cdef double [:] y, xB, _rvs
    cdef:
        int T_h = exog.shape[0]
        int k_gamma = exog.shape[1]
        int inc = 1
        int t
        int i
        int j
        int I = rvs.shape[1]
    cdef:
        double alpha = 1.0
        double beta = 0.0
    
    xB = np.zeros((T_h,), float, order="F")
    y = np.zeros((T_h,), float, order="F")
    _rvs = np.zeros((I,), float, order="F")
    
    # xB = np.dot(x, rho)
    dgemv('N', &T_h, &k_gamma, &alpha, &exog[0,0], &T_h, &rho[0], &inc, &beta, &xB[0], &inc)
    #print np.asarray(xB)
    #print np.asarray(xB).sum()
    
    # y = xB + rvs[:,0]
    dcopy(&T_h, &rvs[0,0], &inc, &y[0], &inc)
    daxpy(&T_h, &alpha, &xB[0], &inc, &y[0], &inc)
    #print np.asarray(y)
    
    for t in range(T_h):
        i = 0
        j = 0
        if s[t] == 1 and y[t] < 0:
            dcopy(&I, &rvs[t,0], &T_h, &_rvs[0], &inc)
            while y[t] < 0:
                # Increment
                i += 1
                j += 1
                # If we're not moving, just draw from the truncated normal
                if j > 50:
                    #y[t] = stats.truncnorm.rvs(-xB[t], np.Inf, loc=xB[t])
                    y[t] = rtnorm((1,), 0, np.Inf, xB[t], 1)
                    continue
                # Make sure we have enough variates
                if i == I:
                    _rvs = np.random.normal(size=(I,))
                    i = 0
                # Set new value
                y[t] = xB[t] + rvs[t,i]
                
        elif s[t] == 0 and y[t] > 0:
            dcopy(&I, &rvs[t,0], &T_h, &_rvs[0], &inc)
            while y[t] > 0:
                # Increment
                i += 1
                j += 1
                # If we're not moving, just draw from the truncated normal
                if j > 50:
                    #y[t] = stats.truncnorm.rvs(-np.Inf, -xB[t], loc=xB[t])
                    y[t] = rtnorm((1,), -np.Inf, 0, xB[t], 1)
                    continue
                # Make sure we have enough variates
                if i == I:
                    _rvs = np.random.normal(size=(I,))
                    i = 0
                # Set new value
                y[t] = xB[t] + rvs[t,i]
    return y

cpdef mh(s, x, G0, G, h, sigma2=10):
    pass