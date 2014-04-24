"""
Tests for OPW Estimation

Author: Chad Fulton
"""
from __future__ import division

import numpy as np
import pandas as pd
import data
import estimation
import results_piger
from numpy.testing import assert_allclose, assert_almost_equal
from nose.exc import SkipTest


class OPWReplication(estimation.OPWEstimate):
    def __init__(self, clip=0, lag=1, sigma2=10, burn_draws=0,
                 converged_draws=20, truncnorm_redraw=1, truncnorm_perdraw=1):
        # Data
        raw = pd.read_excel('data/recession probit data.xlsx',
                            'nber_indicator.csv')
        df = data.OPWData(raw.ix[:raw.shape[0]-clip, :])

        # Model
        model = estimation.OPWModel(df, lag=lag, sigma2=sigma2)

        # Initialize
        super(OPWReplication, self).__init__(model,
                                             burn_draws=0, converged_draws=20,
                                             truncnorm_redraw=1,
                                             truncnorm_perdraw=1)

        # Run the model
        self.draw(print_progress=False)


class TestSimple(OPWReplication):
    def __init__(self):
        self.true = results_piger.simple
        super(TestSimple, self).__init__()

    def draw_rvs_comparators(self):
        return np.array(self.true['comparators'], order="F")

    def draw_rvs_rho(self):
        return np.asfortranarray(np.array(self.true['rho_rvs']).T)

    def draw_rvs_gamma(self):
        # GAUSS uses 1-indexing
        return np.array(self.true['gamma_rvs'], order="F")-1

    def draw_rvs_y(self, shape):
        return np.asfortranarray(np.array(self.true['y_rvs']).T[:, None, :])

    def test_gamma(self):
        assert_almost_equal(
            self.results.selectors.T[:, 1:],
            self.true['gammas'],
            5
        )

    def test_rho(self):
        assert_almost_equal(
            self.results.estimates.T.compressed(),
            np.concatenate(self.true['rhos']),
            5
        )

    def test_y(self):
        assert_almost_equal(
            self.results.latent.T,
            self.true['ys'],
            5
        )

    def test_average_model_size(self):
        assert_almost_equal(
            self.results.model_sizes.mean(),
            self.true['average_model_size'],
            5
        )

    def test_inclusion_probability(self):
        assert_almost_equal(
            self.results.total_inclusions/self.results.iterations,
            self.true['inclusion_probabilities'],
            5
        )

    def test_rho_posterior_mean(self):
        assert_almost_equal(
            self.results.estimates.mean(1).compressed(),
            self.true['posterior_means'],
            5
        )