# distutils: language = c++
# distutils: sources = call_rtnorm.cpp rtnorm.cpp

import numpy as np
cimport numpy as np
cimport cython
from cython_gsl cimport *

cdef extern from "call_rtnorm.hpp":
    void call_rtnorm(double result[], int K, double a, double b, double mu, double sigma)

cpdef rtnorm(shape, double a, double b, double mu, double sigma):
    cdef double [:] result
    cdef int K = np.product(shape)

    # Create the result array
    result = np.zeros((K,), float, order="C")

    # Generate the truncated normals
    call_rtnorm(&result[0], K, a, b, mu, sigma)

    return np.array(result).reshape(shape)

cpdef test(int K = 1, int N = 10000):
    cdef int i
    cdef double [:] result

    # Create the result array
    result = np.zeros((K,), float, order="C")

    # Generate the truncated normals
    for i in range(N):
        call_rtnorm(&result[0], K, 0.0, 1000000.0, 0.0, 1.0)

    return np.array(result)