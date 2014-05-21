from __future__ import division

import numpy as np
import pandas as pd
from scipy import stats
import simulate
import matplotlib.pyplot as plt


class OPWModel(object):
    def __init__(self, data, lag, sigma2=10):
        # Set model properties
        self.data = data
        self.lag = lag
        self.sigma2 = sigma2

        # Get data arrays
        self.endog = self.data.get_endog_array(self.lag,
                                               dtype=np.int32).flatten()
        self.exog = self.data.get_exog_array(self.lag, order="F")
        self.T, self.n = self.exog.shape

        # Preload matrices
        self.M0 = np.asfortranarray(np.eye(self.T))
        self.M0s = self.M0/sigma2

        # Get work array for inversions
        self.NB_mvn = simulate.get_mvn_density_NB(
            self.M0, self.sigma2, np.ones(self.T), self.exog
        )
        self.NB_rho = simulate.get_draw_rho_NB(
            np.asfortranarray(self.M0s[:self.n, :self.n]),
            np.ones(self.T), self.exog, np.ones(self.n)
        )
        self.work_mvn = np.zeros((self.NB_mvn, self.NB_mvn), float, order="F")
        self.work_rho = np.zeros((self.NB_rho, self.NB_rho), float, order="F")

        # Setup the cache for inversions
        self.cache = {}
        self.cache_counts = {}

    def cache_expire(self, recent_threshold=3, alltime_threshold=10):
        for key in self.cache.keys():

            # Update the alltime count
            if key not in self.cache_counts:
                self.cache_counts[key] = 0
            self.cache_counts[key] += self.cache[key]['count']

            used_recently = self.cache[key]['count'] >= recent_threshold
            used_often = self.cache_counts[key] >= alltime_threshold

            # Delete infrequently used inversions
            if not used_recently and not used_often:
                del self.cache[key]
            # For those not deleted, reset the "recent" count
            else:
                self.cache[key]['count'] = 0

    def tostring(self, array):
        return ''.join(['1' if value else '0' for value in array])

    def accept(self, y, gamma, gamma_star):
        gamma_str = self.tostring(gamma)
        gamma_star_str = self.tostring(gamma_star)

        denom = simulate.ln_mn_mass(gamma[1:]) + simulate.ln_mvn_density_ch(
            self.M0, self.sigma2, y,
            np.asfortranarray(self.exog[:, np.array(gamma, bool)]),
            self.work_mvn,
            gamma_str, self.cache
        )

        numer = simulate.ln_mn_mass(gamma_star[1:])+simulate.ln_mvn_density_ch(
            self.M0, self.sigma2, y,
            np.asfortranarray(self.exog[:, np.array(gamma_star, bool)]),
            self.work_mvn,
            gamma_star_str, self.cache
        )

        return np.exp(numer - denom)

    def sample(self, rho, gamma, y_rvs, gamma_rvs, rho_rvs, comparator):
        # 1. Gibbs step: draw y
        y = simulate.draw_y(
            rho[gamma.astype(bool)],
            self.endog,
            np.asfortranarray(self.exog[:, gamma.astype(bool)]),
            y_rvs
        )

        # 2. Metropolis step: draw gamma and rho
        # (now using a new y)

        # Get the acceptance probability
        # (if gamma_rvs[t-1] == 0 then gamma does not change and we accept
        #  with certainty)
        if gamma_rvs > 0:
            gamma_star = np.array(simulate.draw_gamma(gamma, gamma_rvs))
            prob_accept = self.accept(y, gamma, gamma_star)
        else:
            gamma_star = gamma
            prob_accept = 1

        # Update the arrays based on acceptance or not
        # rho = rho.copy()
        accept = prob_accept >= comparator
        # print accept, prob_accept, comparator
        if accept:
            rho = np.zeros(rho.shape)
            gamma = gamma_star.copy()
            # Draw rho
            mask = gamma.astype(bool)
            k_gamma = np.sum(gamma)

            rho[mask] = simulate.draw_rho(
                np.asfortranarray(self.M0s[:k_gamma, :k_gamma]),
                y, np.asfortranarray(self.exog[:, mask]),
                rho_rvs, self.work_rho
            )
        else:
            rho = rho.copy()

        return y, gamma, rho, accept


class OPWEstimate(object):
    def __init__(self, model, burn_draws, converged_draws,
                 truncnorm_redraw=100, truncnorm_perdraw=20,
                 cache_expire=500, cache_recent_threshold=2,
                 cache_alltime_threshold=10):
        # Set estimator properties
        self.model = model
        self.burn_draws = burn_draws
        self.converged_draws = converged_draws
        self.iterations = self.burn_draws+self.converged_draws+1
        self.truncnorm_redraw = truncnorm_redraw
        self.truncnorm_perdraw = truncnorm_perdraw
        self.cache_expire = cache_expire
        self.cache_recent_threshold = cache_recent_threshold
        self.cache_alltime_threshold = cache_alltime_threshold

        # Setup the empty results object
        self.results = OPWResult(self, False)

        # Data arrays
        self.gammas = np.zeros((self.model.n, self.iterations),
                               np.int32, order="F")
        self.gammas[0, :] = 1
        self.rhos = np.zeros((self.model.n, self.iterations), order="F")
        self.ys = np.zeros((self.model.T, self.iterations), order="F")
        self.accepts = np.zeros((self.iterations,), order="F")

        # Random variates
        self.comparators = self.draw_rvs_comparators()
        self.gamma_rvs = self.draw_rvs_gamma()
        self.rho_rvs = self.draw_rvs_rho()

    def draw_rvs_comparators(self):
        return np.random.uniform(size=(self.iterations))

    def draw_rvs_rho(self):
        return np.random.multivariate_normal(
            mean=np.zeros(self.model.n),
            cov=np.eye(self.model.n),
            size=(self.iterations,)
        ).T

    def draw_rvs_gamma(self):
        return np.random.random_integers(0, self.model.n-1,
                                         size=(self.iterations,))

    def draw_rvs_y(self, shape):
        return np.array(
            np.random.normal(size=(self.model.T,)+shape),
            order="F"
        )

    def draw(self, print_progress=True, print_fraction=0.3):
        print_mod = np.floor(print_fraction*self.iterations)
        if self.truncnorm_redraw == 1:
            self.y_rvs = self.draw_rvs_y((self.truncnorm_perdraw,
                                          self.truncnorm_redraw))
        for t in range(1, self.iterations):
            # Conserve memory by drawing only y_rvs for N periods at a time
            if self.truncnorm_redraw == 1:
                l = t-1
            else:
                l = t % self.truncnorm_redraw
                if l == 1:
                    self.y_rvs = self.draw_rvs_y((self.truncnorm_perdraw,
                                                  self.truncnorm_redraw))

            # Cache operations
            if t % self.cache_expire == 1:
                self.model.cache_expire()

            # Draw a Sample
            (self.ys[:, t], self.gammas[:, t], self.rhos[:, t],
                self.accepts[t]) = self.model.sample(
                self.rhos[:, t-1], self.gammas[:, t-1], self.y_rvs[:, :, l],
                self.gamma_rvs[t-1], self.rho_rvs[:, t-1],
                self.comparators[t-1]
            )

            # print t, self.accepts[t], '-------------'

            # Report progress
            if print_progress and t % print_mod == 0:
                print('Iteration %d: %.2f%% complete'
                      % (t, (t/self.iterations)*100))

        self.results.update()
        return self.results


class OPWResult(object):
    def __init__(self, estimator, update=True):
        # Estimation Objects
        self.estimator = estimator
        self.model = self.estimator.model
        self.data = self.model.data

        # Initialize results objects
        self._inclusions = np.zeros((0))
        self._burn = None
        self._yhat = None
        self._probabilities = None
        self._parameters_summary = None
        self._periods_summary = None

        if update:
            self.update()

    def update(self):
        # Cache results objects
        self._inclusions = np.asarray(self.estimator.gammas, dtype=bool).T
        self._burn = self.estimator.burn_draws+1
        self._yhat = None
        self._probabilities = None
        self._parameters_summary = None
        self._periods_summary = None

    @property
    def accepts(self):
        # The zeroth period isn't associated with an accept
        return self.estimator.accepts[self._burn:]

    @property
    def burn(self):
        return self._burn-1

    @burn.setter
    def burn(self, value):
        self._burn = value+1

    @property
    def converged(self):
        return self.iterations - self.burn

    @property
    def iterations(self):
        return self.estimator.iterations-1

    @property
    def inclusions(self):
        # Only interested in masks for drawn models
        return self._inclusions.T[:, self._burn:]

    def inclusion(self, t):
        return self._inclusions[self._burn+t]

    @property
    def endog(self):
        return self.model.endog

    @property
    def exog(self):
        return self.model.exog

    @property
    def state_names(self):
        return self.data.state_names

    @property
    def latent(self):
        # Only interested in latent draws for drawn models
        return self.estimator.ys[:, self._burn:]

    @property
    def selectors(self):
        # Only interested in selectors for drawn models
        return self.estimator.gammas[:, self._burn:]

    @property
    def estimates(self):
        # Only interested in estimates for drawn models
        return np.ma.masked_array(self.estimator.rhos[:, self._burn:],
                                  mask=(~self.inclusions))

    @property
    def yhat(self):
        if self._yhat is None:
            self._yhat = np.zeros((self.model.T, self.iterations+1))
            for t in range(1, self.iterations+1):
                inclusion = self._inclusions[t]
                self._yhat[:, t] = np.dot(
                    self.exog[:, inclusion],
                    self.estimator.rhos[inclusion, t]
                )
        # Only interested in yhat for drawn models
        return self._yhat[:, self._burn:]

    @property
    def probabilities(self):
        if self._probabilities is None:
            self.yhat  # make sure yhat has been calculated
            self._probabilities = stats.norm.cdf(self._yhat)
        # Only interested in probabilities for drawn models
        return self._probabilities[:, self._burn:]

    @property
    def total_inclusions(self):
        # Note: does not include intercept
        # Only interested in inclusion for drawn models
        return self.estimator.gammas[1:, self._burn:].sum(1)

    @property
    def model_sizes(self):
        # Note: does not include intercept
        # Only interested in sizes for drawn models
        return self.estimator.gammas[1:, self._burn:].sum(0)

    @property
    def parameters_summary(self):
        if self._parameters_summary is None:
            self._parameters_summary = pd.DataFrame(
                {
                    'Post. Mean': self.estimates.mean(1),
                    'Post. Std.': self.estimates.std(1),
                    'N Inc.': self.inclusions.sum(1),
                    'Prob. Inc.': self.inclusions.sum(1) / self.converged
                },
                columns=['Post. Mean', 'Post. Std.', 'Prob. Inc.', 'N Inc.'],
                index=self.data.exog_names
            )
        return self._parameters_summary

    @property
    def periods_summary(self):
        if self._periods_summary is None:
            self._periods_summary = pd.DataFrame({
                'Rec.': self.endog,
                'Rec. Prob.': self.probabilities.mean(1)
            }, index=self.data.index[5:])
        return self._periods_summary

    def graph_probabilities(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()

        ax.plot(
            self.periods_summary.index,
            self.periods_summary.ix[:, 'Rec. Prob.'], 'k', alpha=0.8
        )
        ax.fill_between(
            self.periods_summary.index,
            self.periods_summary.ix[:, 'Rec.'], color='k', alpha=0.2
        )

        return ax

    def hist_posterior(self, axes=None):
        k = self.estimates.shape[0]
        if axes is None:
            fig, axes = plt.subplots(k // 4, 4)
        for i in range(k):
            ax = axes[i//4, i % 4]
            values = self.estimates[i, :].compressed()
            inclusions = values.shape[0]
            ax.hist(values, alpha=(inclusions/self.converged)**0.2)
            ax.set(title='%s (N=%d)' % (self.data.exog_names[i], inclusions))
        return axes

    def graph_inclusion(self, ax=None):
        # Start graphing at the zeroth model (i.e. before the first draw)
        if ax is None:
            fig, ax = plt.subplots()
        periods = np.arange(self.burn, self.iterations)
        for i in range(self.model.n):
            inclusion = self.selectors[i].astype(bool)
            ax.plot(
                self.estimator.gammas[i, self.burn:][inclusion]*i,
                periods[inclusion], 'k.'
            )

        ax.xaxis.set(ticks=range(self.model.n))
        ax.xaxis.set_ticklabels(self.data.exog_names, rotation=90)
        ax.set(xlim=(-1, self.model.n),
               xlabel='Predictor', ylabel='Iteration')

        return ax
