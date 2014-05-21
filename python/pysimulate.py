from __future__ import division
import numpy as np
import pandas as pd
from scipy import stats, misc, linalg
import time

np.set_printoptions(precision=5, suppress=True)


def get_data():
    # Data
    data = pd.read_excel('data/recession probit data.xlsx',
                         'nber_indicator.csv')
    data.index = pd.date_range(start='1960-01-01', periods=data.shape[0],
                               freq='MS')

    growth_rate = lambda data, delta=3: (data / data.shift(delta) - 1)*100
    state_columns = data.columns[8:].tolist()

    data['sp500_return'] = growth_rate(data['S&P 500'])
    data['term_spread'] = (data['Ten Year Bond'] - data['Three Month Bill'])
    data['agg_emp_growth'] = growth_rate(data['U.S. Employment'])
    data['agg_ip_growth'] = growth_rate(data['U.S. Industrial Production'])
    data.ix[:, state_columns] = growth_rate(data.ix[:, state_columns])

    growth_columns = ['agg_emp_growth', 'agg_ip_growth'] + state_columns
    data.ix[:, growth_columns] = data.ix[:, growth_columns].shift(1)

    # Final datasets
    endog = np.asarray(data.ix[3+lag+1:, 'NBER Indicator'], np.int32)
    exog = np.c_[
        np.ones(endog.shape[0]),
        np.asarray(data.ix[3+1:-lag,
                   ['Federal Funds Rate', 'sp500_return', 'term_spread'] +
                   growth_columns])
    ]

    return endog, exog


# Utility functions
def tostring(array):
    return ''.join(['1' if value else '0' for value in array])


# MCMC Functions
def draw_gamma(gamma, rvs):
    gamma_star = np.copy(gamma)
    if rvs > 0:
        if gamma_star[rvs] == 1:
            gamma_star[rvs] = 0
        else:
            gamma_star[rvs] = 1
    return gamma_star


def draw_rho(M0, y, exog, rvs):
    M1 = np.linalg.inv(M0 + exog.T.dot(exog))
    m1 = M1.dot(exog.T.dot(y))
    A = np.linalg.cholesky(M1)
    return m1 + A.dot(rvs)


def ln_mvn_density(M0, sigma2, y, exog, cache_key):
    """
    Uses basic Python inversion and optional caching
    """
    if cache_key is not None and cache_key in ln_mvn_density.cache:
        Sigma = ln_mvn_density.cache[cache_key]['Sigma']
        det = ln_mvn_density.cache[cache_key]['det']
        ln_mvn_density.cache[cache_key]['count'] += 1
    else:
        Sigma = M0 + sigma2*exog.dot(exog.T)
        det = np.linalg.det(Sigma)
        if cache_key is not None:
            ln_mvn_density.cache[cache_key] = {
                'Sigma': Sigma,
                'det': det,
                'count': 0
            }

    return -0.5*np.log(det) - 0.5*y.dot(np.linalg.inv(Sigma).dot(y))
ln_mvn_density.cache = {}


def ln_mvn_density_ch(M0, sigma2, y, exog, cache_key):
    """
    Uses Cholesky decomposition with linear solver and optional caching
    """

    if cache_key is not None and cache_key in ln_mvn_density_ch.cache:
        A = ln_mvn_density_ch.cache[cache_key]['A']
        det = ln_mvn_density_ch.cache[cache_key]['det']
        ln_mvn_density_ch.cache[cache_key]['count'] += 1
    else:
        Sigma = M0 + sigma2*exog.dot(exog.T)
        # Cholesky decomposition
        A, info = linalg.lapack.dpotrf(Sigma, lower=1)
        # Determinant
        det = np.prod(A.diagonal())**2
        if cache_key is not None:
            ln_mvn_density_ch.cache[cache_key] = {
                'A': A,
                'det': det,
                'count': 0
            }

    # Solve the linear system
    res, info = linalg.lapack.dtrtrs(A, y, lower=1)

    return -0.5*np.log(det) - 0.5*res.dot(res)
ln_mvn_density_ch.cache = {}


def ln_mn_mass(gamma):
    return np.log(1/misc.comb(gamma.shape[0], np.sum(gamma)))


def draw_y(rho, endog, exog, rvs):
    T = exog.shape[0]
    I = rvs.shape[1]
    max_iter = I*3

    xB = np.dot(exog, rho)
    y = rvs[:, 0] + xB

    for t in range(T):
        i = 0
        j = 0
        if endog[t] == 1 and y[t] < 0:
            _rvs = rvs[t, :]
            while y[t] < 0:
                # Increment
                i += 1
                j += 1
                # If we're not moving, just draw from the truncated normal
                if j > max_iter:
                    y[t] = stats.truncnorm.rvs(-xB[t], np.Inf, loc=xB[t])
                    continue
                # Make sure we have enough variates
                if i == I:
                    _rvs = np.random.normal(size=(I,))
                    i = 0
                # Set new value
                y[t] = xB[t] + _rvs[i]
        elif endog[t] == 0 and y[t] > 0:
            _rvs = rvs[t, :]
            while y[t] > 0:
                # Increment
                i += 1
                j += 1
                # If we're not moving, just draw from the truncated normal
                if j > 50:
                    y[t] = stats.truncnorm.rvs(-np.Inf, -xB[t], loc=xB[t])
                    continue
                # Make sure we have enough variates
                if i == I:
                    _rvs = np.random.normal(size=(I,))
                    i = 0
                # Set new value
                y[t] = xB[t] + _rvs[i]
    return y


# Random Variate Functions
def draw_rvs_comparators(iterations):
    return np.random.uniform(size=(iterations))


def draw_rvs_rho(n, iterations):
    return np.random.multivariate_normal(
        mean=np.zeros(n),
        cov=np.eye(n),
        size=(iterations,)
    ).T


def draw_rvs_gamma(n, iterations):
    return np.random.random_integers(0, n-1, size=(iterations,))


def draw_rvs_y(T, shape):
    return np.array(
        np.random.normal(size=(T,)+shape),
        order="F"
    )


# MH Functions
def calculate_accept(y, exog, M0, gamma, gamma_star, use_cholesky, use_cache):
    gamma_indicators = np.array(gamma, bool)
    gamma_star_indicators = np.array(gamma_star, bool)
    cache_key = None

    _ln_mvn_density = ln_mvn_density_ch if use_cholesky else ln_mvn_density

    if use_cache:
        cache_key = tostring(gamma)
    denom = ln_mn_mass(gamma[1:]) + _ln_mvn_density(
        M0, sigma2, y,
        np.asfortranarray(exog[:, gamma_indicators]),
        cache_key
    )

    if use_cache:
        cache_key = tostring(gamma_star)
    numer = ln_mn_mass(gamma_star[1:]) + _ln_mvn_density(
        M0, sigma2, y,
        np.asfortranarray(exog[:, gamma_star_indicators]),
        cache_key
    )

    return np.exp(numer - denom)


def sample(exog, endog, M0, M0s, rho, gamma, y_rvs, gamma_rvs, rho_rvs,
           comparator, use_cholesky, use_cache):
    # 1. Gibbs step: draw y
    gamma_indicators = np.array(gamma, bool)
    y = draw_y(
        rho[gamma_indicators], endog,
        np.asfortranarray(exog[:, gamma_indicators]), y_rvs
    )

    # 2. Metropolis step: draw gamma and rho

    # Get the acceptance probability
    if gamma_rvs > 0:
        gamma_star = draw_gamma(gamma, gamma_rvs)
        prob_accept = calculate_accept(y, exog, M0, gamma, gamma_star,
                                       use_cholesky, use_cache)
    else:
        gamma_star = gamma
        prob_accept = 1

    # Update the arrays based on acceptance or not
    accept = prob_accept >= comparator
    if accept:
        rho = np.zeros(rho.shape)
        gamma = gamma_star.copy()
        # Draw rho
        gamma_indicators = np.array(gamma, bool)
        k_gamma = np.sum(gamma)

        rho[gamma_indicators] = draw_rho(
            np.asfortranarray(M0s[:k_gamma, :k_gamma]),
            y, np.asfortranarray(exog[:, gamma_indicators]),
            rho_rvs[:k_gamma]
        )
    else:
        rho = rho.copy()

    return y, gamma, rho, accept


# Metropolis Hastings Algorithm
def mh(exog, endog, G0, G, sigma=10, print_pct=0, use_cholesky=True,
       use_cache=True):
    # Parameters
    T, n = exog.shape
    iterations = G0 + G + 1
    I = 20   # controls shape of y_rvs
    N = 100  # controls number of periods y_rvs is drawn for
    progress_updates = np.ceil((iterations-1)*print_pct)

    # Cached arrays
    M0 = np.asfortranarray(np.eye(T))
    M0s = M0/sigma2

    # Data arrays
    gammas = np.zeros((n, iterations), np.int32, order="F")
    gammas[0, :] = 1
    rhos = np.zeros((n, iterations), order="F")
    ys = np.zeros((T, iterations), order="F")
    accepts = np.zeros((iterations,), order="F")

    # Random variates
    comparators = draw_rvs_comparators(iterations)
    gamma_rvs = draw_rvs_gamma(n, iterations)
    rho_rvs = draw_rvs_rho(n, iterations)

    # MH
    for t in range(1, iterations):
        # Conserve memory by drawing only y_rvs for N periods at a time
        l = t % N
        if l == 1:
            y_rvs = draw_rvs_y(T, (I, N))
        # Draw a Sample
        ys[:, t], gammas[:, t], rhos[:, t], accepts[t] = sample(
            exog, endog, M0, M0s,
            rhos[:, t-1], gammas[:, t-1], y_rvs[:, :, l],
            gamma_rvs[t-1], rho_rvs[:, t-1], comparators[t-1],
            use_cholesky, use_cache
        )

        # Report progress
        if progress_updates and t % progress_updates == 0:
            print('Iteration %d: %.2f%% complete'
                  % (t, (t/(iterations-1))*100))
    return ys, gammas, rhos, accepts

if __name__ == '__main__':
    np.random.seed(1234)

    # Estimation options
    use_cholesky = False
    use_cache = False

    # Model Parameters
    lag = 1
    sigma2 = 10

    # Data
    endog, exog = get_data()

    start_time = time.time()  # timing

    # Parameters
    G0 = 200
    G = 200

    # Iterate
    ys, gammas, rhos, accepts = mh(exog, endog, G0, G,
                                   use_cholesky=use_cholesky,
                                   use_cache=use_cache)

    end_time = time.time()
    elapsed = end_time - start_time  # timing

    print('Runtime of %d minutes and %.2f seconds'
          % (elapsed // 60, elapsed % 60))

    print('Number of draws to convergence: %d' % G0)
    print('Number of draws after convergence: %d' % G)
    print('Prior VC matrix for model parameters is: %.2f' % sigma2)
    print('Average Model Size: %.2f' % gammas[:, 1:].sum(0).mean())
