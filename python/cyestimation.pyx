# cython: profile=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=False

cimport numpy as cnp
cnp.import_array()
cimport cpython

cdef extern from *:
    int unlikely(int)

# Typical imports
import cython
import numpy as np

# Statistical functions imports
from scipy.misc import comb
from scipy import stats

# BLAS / LAPACK
from cpython cimport PyCObject_AsVoidPtr
import scipy
from scipy.linalg import blas
from scipy.linalg import lapack

from libc.math cimport sqrt, exp, log
cdef extern from "math.h":
    cnp.float64_t NPY_PI

from blas_lapack cimport *

cdef dcopy_t *dcopy = <dcopy_t*>PyCObject_AsVoidPtr(scipy.linalg.blas.dcopy._cpointer)
cdef daxpy_t *daxpy = <daxpy_t*>PyCObject_AsVoidPtr(scipy.linalg.blas.daxpy._cpointer)
cdef ddot_t *ddot = <ddot_t*>PyCObject_AsVoidPtr(scipy.linalg.blas.ddot._cpointer)
cdef dgemm_t *dgemm = <dgemm_t*>PyCObject_AsVoidPtr(scipy.linalg.blas.dgemm._cpointer)
cdef dgemv_t *dgemv = <dgemv_t*>PyCObject_AsVoidPtr(scipy.linalg.blas.dgemv._cpointer)
cdef dtrtrs_t *dtrtrs = <dtrtrs_t*>PyCObject_AsVoidPtr(scipy.linalg.lapack.dtrtrs._cpointer)
cdef dsyrk_t *dsyrk = <dsyrk_t*>PyCObject_AsVoidPtr(scipy.linalg.blas.dsyrk._cpointer)
cdef dtrmv_t *dtrmv = <dtrmv_t*>PyCObject_AsVoidPtr(scipy.linalg.blas.dtrmv._cpointer)
cdef dsymv_t *dsymv = <dsymv_t*>PyCObject_AsVoidPtr(scipy.linalg.blas.dsymv._cpointer)
cdef dpotrf_t *dpotrf = <dpotrf_t*>PyCObject_AsVoidPtr(scipy.linalg.lapack.dpotrf._cpointer)
cdef dpotri_t *dpotri = <dpotri_t*>PyCObject_AsVoidPtr(scipy.linalg.lapack.dpotri._cpointer)
cdef dpotrs_t *dpotrs = <dpotrs_t*>PyCObject_AsVoidPtr(scipy.linalg.lapack.dpotrs._cpointer)

# Constants
cdef double alpha = 1.0
cdef double beta = 0.0
cdef double gamma = -1.0

cdef int inc = 1
cdef int info

cdef long get_cache_key(int nexog, int * indicators):
    """
    Generate integer cache key

    Converts an array of indicators (e.g. [1,0,1, ..., 0]) to a
    integer representing the corresponding binary number (e.g. 0b101...0)

    Returns a long, meaning that it can only accept an indicator array
    of maximum length 63.
    """
    cdef int i
    cdef long y = 0

    for i in range(nexog):
        y = (y << 1) + indicators[i]

    return y

cdef int get_cache_location(int cache_size, int nexog, long * cache_locations, int * indicators):
    """
    Generate a cache location.
    """
    cdef long cache_key
    cdef int i

    cache_key = get_cache_key(nexog, indicators)

    # Try to find an existing cache location
    for i in range(cache_size):
        # Check if i is the cache location
        if cache_locations[i] == cache_key:
            return i

    return -1

cdef int get_next_cache_location(int cache_size, long * cache_locations):
    cdef int i
    for i in range(cache_size):
        if cache_locations[i] == -1:
            # Free the next cache location
            cache_locations[i+1] = -1
            return i

    # If everything has been assigned, loop around
    cache_locations[1] = -1
    return 0

cdef draw_augmented(int nobs, int * endog, double * draw, double * predicted):
    cdef double [:] rvs
    cdef int t, i, j
    cdef int ndraws = nobs * 10, maxdraws = 20

    # Copy the predicted values to the draw
    #dcopy(&nobs, predicted, &inc, draw, &inc)

    i = ndraws
    for t in range(nobs):
        j = 0
        while True:
            if i >= ndraws:
                rvs = np.random.normal(size=(ndraws,))
                i = 0

            draw[t] = predicted[t] + rvs[i]

            j += 1
            i += 1

            if (draw[t] > 0 and endog[t] == 1) or (draw[t] < 0 and endog[t] == 0):
                break

            if j > maxdraws:
                if endog[t] == 1:
                    draw[t] = stats.truncnorm.rvs(-predicted[t], np.Inf, loc=predicted[t])
                else:
                    draw[t] = stats.truncnorm.rvs(-np.Inf, -predicted[t], loc=predicted[t])
                break

cdef draw_coefficients(int nobs, int nexog, int selected_nexog, double variance, double * variates,
                       double * augmented, int * indicators, 
                       double * selected_exog,
                       double * factorization, double * tmp,
                       double * coefficients):
    cdef int i, j
    cdef double precision = 1.0 / variance

    # print 'indicators'
    # x = np.zeros((nexog))
    # for i in range(nexog):
    #     x[i] = indicators[i]
    # print x

    # print 'selected'
    # x = np.zeros((nobs, selected_nexog))
    # for i in range(selected_nexog):
    #     for j in range(nobs):
    #         x[j, i] = selected_exog[j + i*nobs]
    # print x

    # Posterior covariance matrix
    dsyrk("L", "T", &selected_nexog, &nobs, &alpha, selected_exog, &nobs, &beta, factorization, &selected_nexog)
    for i in range(selected_nexog):
        factorization[i + i*selected_nexog] += precision

    #for i in range(selected_nexog):
    #    for j in range(i, selected_nexog):
    #        factorization[i + j*selected_nexog] = factorization[j + i*selected_nexog]

    # print "I/s2 + x'x"
    # x = np.zeros((selected_nexog, selected_nexog))
    # for i in range(selected_nexog):
    #     for j in range(selected_nexog):
    #         x[j, i] = factorization[j + i*selected_nexog]
    # print x
    # print factorization[1]
    # print factorization[2]
    # print factorization[3]
    # print factorization[4]

    dpotrf("L", &selected_nexog, factorization, &selected_nexog, &info)
    dpotri("L", &selected_nexog, factorization, &selected_nexog, &info)

    # print 'inverse'
    # x = np.zeros((selected_nexog, selected_nexog))
    # for i in range(selected_nexog):
    #     for j in range(selected_nexog):
    #         x[j, i] = factorization[j + i*selected_nexog]
    # print x

    # Posterior mean
    dgemv("T", &nobs, &selected_nexog, &alpha, selected_exog, &nobs, augmented, &inc, &beta, tmp, &inc)
    #dgemv("N", &selected_nexog, &selected_nexog, &alpha, factorization, &selected_nexog, tmp, &inc, &beta, coefficients, &inc)
    dsymv("L", &selected_nexog, &alpha, factorization, &selected_nexog, tmp, &inc, &beta, coefficients, &inc)

    # print 'posterior mean'
    # x = np.zeros((selected_nexog,))
    # for i in range(selected_nexog):
    #         x[i] = coefficients[i]
    # print x

    # Cholesky decomposition
    dpotrf("L", &selected_nexog, factorization, &selected_nexog, &info)

    # print 'cholesky_factorization'
    # x = np.zeros((selected_nexog, selected_nexog))
    # for i in range(selected_nexog):
    #     for j in range(selected_nexog):
    #         x[j, i] = factorization[j + i*selected_nexog]
    # print x

    # Calculate coefficients (overwrites variates as tmp array)
    dtrmv("L", "T", "N", &selected_nexog, factorization, &selected_nexog, variates, &inc)
    # print 'coefs'
    # x = np.zeros((selected_nexog,))
    # for i in range(selected_nexog):
    #         x[i] = variates[i]
    # print x

    daxpy(&selected_nexog, &alpha, variates, &inc, coefficients, &inc)

    # print 
    # x = np.zeros((selected_nexog,))
    # for i in range(selected_nexog):
    #         x[i] = coefficients[i]
    # print x

    # Coefficients are in first selected_nexog spots, need to expand them to
    # the entire vector
    j = selected_nexog-1
    for i in range(nexog-1, -1, -1):
        if indicators[i] == 1:
            coefficients[i] = coefficients[j]
            j -= 1
        else:
            coefficients[i] = 0


cdef draw_indicators(int variate, int * indicators, int * nslopes):
    if variate > 0:
        if indicators[variate] == 1:
            indicators[variate] = 0
            nslopes[0] -= 1
        else:
            indicators[variate] = 1
            nslopes[0] += 1

# See http://www.johndcook.com/csharp_log_factorial.html
cdef double * lf = [0.000000000000000, 0.000000000000000, 0.693147180559945, 1.791759469228055, 3.178053830347946, 4.787491742782046, 6.579251212010101, 8.525161361065415, 10.604602902745251, 12.801827480081469, 15.104412573075516, 17.502307845873887, 19.987214495661885, 22.552163853123421, 25.191221182738683, 27.899271383840894, 30.671860106080675, 33.505073450136891, 36.395445208033053, 39.339884187199495, 42.335616460753485, 45.380138898476908, 48.471181351835227, 51.606675567764377, 54.784729398112319, 58.003605222980518, 61.261701761002001, 64.557538627006323, 67.889743137181526, 71.257038967168000, 74.658236348830158, 78.092223553315307, 81.557959456115029, 85.054467017581516, 88.580827542197682, 92.136175603687079, 95.719694542143202, 99.330612454787428, 102.968198614513810, 106.631760260643450, 110.320639714757390, 114.034211781461690, 117.771881399745060, 121.533081515438640, 125.317271149356880, 129.123933639127240, 132.952575035616290, 136.802722637326350, 140.673923648234250, 144.565743946344900, 148.477766951773020, 152.409592584497350, 156.360836303078800, 160.331128216630930, 164.320112263195170, 168.327445448427650, 172.352797139162820, 176.395848406997370, 180.456291417543780, 184.533828861449510, 188.628173423671600, 192.739047287844900, 196.866181672889980, 201.009316399281570, 205.168199482641200, 209.342586752536820, 213.532241494563270]
cdef double lnfact(int n):
    cdef double x
    if n > 66:
        x = n + 1;
        return (x - 0.5)*log(x) - x + 0.5*log(2*NPY_PI) + 1.0/(12.0*x);
    return lf[n]

cdef double ln_nchoosek(int n, int k):
    return lnfact(n) - lnfact(k) - lnfact(n-k)

cdef double ln_mn_mass(int nexog, int selected_nexog):
    return -ln_nchoosek(nexog, selected_nexog)

cdef cholesky_factorization(int nobs, int selected_nexog, double variance, double * selected_exog,
                            double * factorization, double * determinant):
    # $\\# = I_{nobs}$
    cdef int i

    # `factorization` $= # + exog exog'$
    dsyrk("L", "N", &nobs, &selected_nexog, &variance, selected_exog, &nobs, &beta, factorization, &nobs)
    for i in range(nobs):
        factorization[i + i*nobs] += 1

    # Cholesky decomposition
    dpotrf("L", &nobs, factorization, &nobs, &info)

    # Determinant
    determinant[0] = 1
    for i in range(nobs):
        determinant[0] *= factorization[i + i*nobs]
    determinant[0] *= determinant[0]

cdef double ln_mvn_density(int nobs, int selected_nexog, double determinant,
                    double * factorization, double * selected_exog,
                    double * augmented, double * tmp):
    

    dcopy(&nobs, augmented, &inc, tmp, &inc)
    dtrtrs('L', 'N', 'N', &nobs, &inc, factorization, &nobs, tmp, &nobs, &info)

    return - 0.5*log(determinant) - 0.5*ddot(&nobs, tmp, &inc, tmp, &inc)

cdef select_exog(int nobs, int nexog, int * indicators,
                 double * exog, double * selected_exog,
                 double * coefficients, double * selected_coefficients):
    cdef int i, j
    j = 0
    for i in range(nexog):
        if indicators[i]:

            # Copy the ith column of the exogenous matrix
            dcopy(&nobs, &exog[i * nobs], &inc,
                        &selected_exog[j * nobs], &inc)

            # Copy the ith coefficient
            selected_coefficients[j] = coefficients[i]

            j += 1

cdef class Simulate(object):

    # ### Sampling parameters
    cdef readonly int nburn
    cdef readonly int nsample
    cdef readonly int niterations
    cdef readonly int t
    cdef readonly double prior_variance

    # ### Dimensions
    # $(613)$
    cdef readonly int nobs
    # $(56)$
    cdef readonly int nexog

    # ### Data
    # $(nobs)$
    cdef readonly int [:] endog
    # $(nobs \times nexog)$
    cdef readonly double [::1,:] exog

    # ### Storage arrays

    # $y (nobs \times niterations)$
    cdef readonly double [::1, :] augmented
    # $\rho = [\alpha, \beta']' (nexog x niterations)$
    cdef readonly double [::1, :] coefficients
    # $\gamma (nexog \times niterations)$
    cdef readonly int [::1, :] indicators
    # $|\gamma|$
    cdef readonly int [:] selected_nexog

    # acceptances
    cdef readonly int [:] acceptances

    # ### Temporary arrays

    cdef double * tmp
    cdef double * tmp2

    # $exog \rho$
    cdef double * predicted

    # #### Cache
    cdef double * cache
    cdef double * determinants
    cdef readonly long [:] cache_locations
    cdef readonly int cache_size

    # #### Selected arrays
    # This holds the selected exogenous array
    # $(nobs \times selected_nexog)$
    cdef double * selected_exog
    # $(selected_nexog)$
    cdef double * selected_coefficients

    # ### Random variates
    cdef readonly double [::1, :] coefficient_variates
    cdef readonly int [:] indicator_variates
    cdef readonly double [:] metropolis_variates

    def __cinit__(self, int [:] endog, double [::1,:] exog, double prior_variance, int nburn, int nsample, int cache_size=2000):

        cdef int i
        cdef cnp.intp_t dims[2]

        # Setup iteration parameters
        self.nburn = nburn
        self.nsample = nsample
        self.niterations = nburn + nsample + 1
        # start at the t=1 iteration so that t-1 is a valid array location
        self.t = 1
        self.prior_variance = prior_variance
        self.cache_size = cache_size if cache_size < self.niterations else self.niterations

        # Save the dataset
        self.endog = endog
        self.exog = exog

        # Setup the problem dimensions
        self.nobs = self.exog.shape[0]
        self.nexog = self.exog.shape[1]

        # Setup the data storage arrays

        # Augmented data
        dims[0] = <cnp.intp_t> self.nobs; dims[1] = <cnp.intp_t> self.niterations;
        self.augmented = cnp.PyArray_EMPTY(2, &dims[0], cnp.NPY_DOUBLE, 1)

        # Indicators
        dims[0] = <cnp.intp_t> self.nexog; dims[1] = <cnp.intp_t> self.niterations;
        self.indicators = cnp.PyArray_ZEROS(2, &dims[0], cnp.NPY_INT, 1)

        # Coefficients
        dims[0] = <cnp.intp_t> self.nexog; dims[1] = <cnp.intp_t> self.niterations;
        self.coefficients = cnp.PyArray_ZEROS(2, &dims[0], cnp.NPY_DOUBLE, 1)

        # Number of selected exogenous variables
        dims[0] = <cnp.intp_t> self.niterations;
        self.selected_nexog = cnp.PyArray_ZEROS(1, &dims[0], cnp.NPY_INT32, 1)

        # Cache Locations
        dims[0] = <cnp.intp_t> self.cache_size;
        self.cache_locations = cnp.PyArray_EMPTY(1, &dims[0], cnp.NPY_INT64, 1)

        # Acceptances
        dims[0] = <cnp.intp_t> self.niterations;
        self.acceptances = cnp.PyArray_EMPTY(1, &dims[0], cnp.NPY_INT, 1)

        # The indicator for the constant term is always 1
        for i in range(self.niterations):
            self.indicators[0, i] = 1
        self.selected_nexog[0] = 1

        # Allocate memory
        # Predicted values
        self.predicted = <double *> cpython.PyMem_Malloc(self.nobs * sizeof(double))
        if unlikely(self.predicted == NULL): raise MemoryError()
        # Cache
        self.cache = <double *> cpython.PyMem_Malloc(self.nobs * self.nobs * self.cache_size * sizeof(double))
        if unlikely(self.cache == NULL): raise MemoryError()
        self.determinants = <double *> cpython.PyMem_Malloc(self.cache_size * sizeof(double))
        if unlikely(self.determinants == NULL): raise MemoryError()
        # Selected exogenous matrix
        self.selected_exog = <double *> cpython.PyMem_Malloc(self.nobs * self.nexog * sizeof(double))
        if unlikely(self.selected_exog == NULL): raise MemoryError()
        # Selected coefficients
        self.selected_coefficients = <double *> cpython.PyMem_Malloc(self.nexog * sizeof(double))
        if unlikely(self.selected_coefficients == NULL): raise MemoryError()
        # Temporary arrays
        self.tmp = <double *> cpython.PyMem_Malloc(self.nexog * self.nexog * sizeof(double))
        if unlikely(self.tmp == NULL): raise MemoryError()
        self.tmp2 = <double *> cpython.PyMem_Malloc(self.nexog * self.nexog * sizeof(double))
        if unlikely(self.tmp2 == NULL): raise MemoryError()

    def __init__(self, int [:] endog, double [::1,:] exog, double prior_variance, int nburn, int nsample, int cache_size=2000):

        # Generate random variates
        self.coefficient_variates = np.asfortranarray(np.random.normal(
            size=(self.nexog, self.niterations,)
        ))
        self.indicator_variates = np.random.random_integers(
            0, self.nexog-1, size=(self.niterations,)
        ).astype(np.int32)
        self.metropolis_variates = np.random.uniform(
            size=(self.niterations,)
        ).T

        # Initialize the first cache location as unassigned
        self.cache_locations[0] = -1


    def __dealloc__(self):
        # De-allocate memory
        cpython.PyMem_Free(self.tmp2)
        cpython.PyMem_Free(self.tmp)
        cpython.PyMem_Free(self.selected_coefficients)
        cpython.PyMem_Free(self.selected_exog)
        cpython.PyMem_Free(self.determinants)
        cpython.PyMem_Free(self.cache)
        cpython.PyMem_Free(self.predicted)

    def __iter__(self):
        return self

    def __next__(self):
        # Get time subscript, and stop the iterator if at the end
        if not self.t < self.niterations:
            raise StopIteration
        
        self.sample()

        self.t += 1

    cpdef advance(self, int n = 0):
        """
        advance(n)

        Advance a specific number of iterations.
        """
        cdef int i

        if n == 0:
            n = self.niterations-1

        for i in range(n):
            self.__next__()

    cdef sample(self):
        cdef int i, t = self.t
        cdef int cache_location
        cdef double acceptance_probability

        # print '------> ', t

        # ## 0. Setup

        # Select the exogenous array and coefficients
        # based on previous step
        select_exog(self.nobs, self.nexog, &self.indicators[0, t-1],
                    &self.exog[0, 0], &self.selected_exog[0],
                    &self.coefficients[0, t-1], &self.selected_coefficients[0])

        # print 'selected'

        # Generate predicted values for the augmented data (i.e. the mean)
        # based on the previous step
        dgemv('N', &self.nobs, &self.selected_nexog[t-1],
              &alpha, &self.selected_exog[0], &self.nobs,
                      &self.selected_coefficients[0], &inc,
              &beta, self.predicted, &inc)

        #for i in range(self.nobs):
        #    print self.predicted[i]
        # print 'predicted'

        # 1. Gibbs step: draw augmented data
        draw_augmented(self.nobs, &self.endog[0], &self.augmented[0, t], self.predicted)

        # print 'augmented'

        # 2. Metropolis step: draw indicators and coefficients
        # based on the drawn augmented data

        # Copy the previous iteration to the proposal
        #self.indicators[:, t] = self.indicators[:, t-1]
        for i in range(self.nexog):
            self.indicators[i, t] = self.indicators[i, t-1]
        self.selected_nexog[t] = self.selected_nexog[t-1]

        # Get the acceptance probability
        if self.indicator_variates[t] > 0:
            # print 'pre indicators'
            # Get the proposed indicators and new number of indicators
            draw_indicators(self.indicator_variates[t], &self.indicators[0, t], &self.selected_nexog[t])

            # print 'indicators'

            acceptance_probability = self.get_acceptance_probability()

            # print 'acceptance', acceptance_probability
        else:
            # print 'copy indicators'
            acceptance_probability = 1.0

        # Update the arrays based on acceptance or not
        self.acceptances[t] = acceptance_probability >= self.metropolis_variates[t]
        if self.acceptances[t]:
            # print 'drawing'
            draw_coefficients(self.nobs, self.nexog, self.selected_nexog[t], self.prior_variance,
                              &self.coefficient_variates[0, t],
                              &self.augmented[0, t], &self.indicators[0, t],
                              &self.selected_exog[0], 
                              &self.tmp[0], &self.tmp2[0],
                              &self.coefficients[0, t])

            # print 'coefficients'
        else:
            # If not accepted, reset everything to the previous iteration's values
            dcopy(&self.nexog, &self.coefficients[0, t-1], &inc, &self.coefficients[0, t], &inc)
            self.selected_nexog[t] = self.selected_nexog[t-1]
            # self.indicators[:, t] = self.indicators[:, t-1]
            for i in range(self.nexog):
                self.indicators[i, t] = self.indicators[i, t-1]
            # print 'copy previous'

    cdef double get_acceptance_probability(self):
        cdef int t = self.t
        cdef int cache_location
        cdef double numer, denom, ln_mass, ln_density
        cdef double * factorization
        cdef double * determinant

        # Calculate the density for the previous step
        # print 'numer (previous)'
        cache_location = self.init_factorization(t-1)
        # print 'cache', cache_location
        ln_mass = ln_mn_mass(self.nexog - 1, self.selected_nexog[t-1] - 1)
        # print 'mass', ln_mass
        ln_density = ln_mvn_density(self.nobs, self.selected_nexog[t-1], self.determinants[cache_location],
                                    &self.cache[self.nobs * self.nobs * cache_location],
                                    &self.selected_exog[0], &self.augmented[0, t], self.predicted) # using predicted as tmp array
        # print 'density', ln_density
        denom = ln_mass + ln_density

        # Update the selected exogenous array for the proposal
        select_exog(self.nobs, self.nexog, &self.indicators[0, t],
                    &self.exog[0, 0], &self.selected_exog[0],
                    self.predicted, self.predicted) # using predicted as tmp array

        # Calculate the density for proposal
        # print 'denom (proposal)'
        cache_location = self.init_factorization(t)
        # print 'cache', cache_location
        ln_mass = ln_mn_mass(self.nexog - 1, self.selected_nexog[t] - 1)
        # print 'mass', ln_mass
        ln_density = ln_mvn_density(self.nobs, self.selected_nexog[t], self.determinants[cache_location],
                                    &self.cache[self.nobs * self.nobs * cache_location],
                                    &self.selected_exog[0], &self.augmented[0, t], self.predicted) # using predicted as tmp array
        # print 'density', ln_density
        numer = ln_mass + ln_density

        return exp(numer - denom)

    cdef int init_factorization(self, int t):
        cdef int i, j, cache_location
        cdef int calculate = 0        

        cache_location = get_cache_location(self.cache_size, self.nexog, &self.cache_locations[0], &self.indicators[0, t])

        # If the indicators are unassigned, fill the cache and force calculation
        if cache_location == -1:
            cache_location = get_next_cache_location(self.cache_size, &self.cache_locations[0])
            self.cache_locations[cache_location] = get_cache_key(self.nexog, &self.indicators[0, t])
            calculate = 1
        # print '-> ', cache_location

        factorization = &self.cache[self.nobs * self.nobs * cache_location]
        determinant = &self.determinants[cache_location]

        if calculate:
            # print 'calced'
            # print 'calculation'
            # print 'nexog', self.selected_nexog[t]
            #for i in range(self.selected_nexog[t]):
            #    for j in range(self.nobs):
            #        print self.selected_exog[j + i*self.nobs]
            cholesky_factorization(
                self.nobs, self.selected_nexog[t], self.prior_variance,
                &self.selected_exog[0],
                factorization, determinant
            )
        # else:
            # print 'not calced'
        # x = np.zeros((self.nobs, self.nobs))
        # for i in range(self.nobs):
        #     for j in range(self.nobs):
        #         x[j, i] = factorization[j + i*self.nobs]
        # print x

        # print 'det', determinant[0]

        return cache_location
