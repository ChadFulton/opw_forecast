from __future__ import division
import numpy as np
import pandas as pd
import time
import estimation
import data
np.set_printoptions(precision=5, suppress=True)

if __name__ == '__main__':
    np.random.seed(1234)

    # Model Parameters
    lag = 1
    sigma2 = 10

    # Get the dataset
    raw = pd.read_excel('data/recession probit data.xlsx',
                        'nber_indicator.csv')
    df = data.OPWData(raw.ix[:raw.shape[0], :])

    start_time = time.time()  # timing

    # Parameters
    G0 = 200
    G = 200

    mod = estimation.OPWModel(df, lag, sigma2=sigma2)
    est = estimation.OPWEstimate(mod, G0, G)
    res = est.draw(False)

    end_time = time.time()
    elapsed = end_time - start_time

    print('Runtime of %d minutes and %.2f seconds'
          % (elapsed // 60, elapsed % 60))

    print('Number of draws to convergence: %d' % res.burn)
    print('Number of draws after convergence: %d' % res.converged)
    print('Prior VC matrix for model parameters is: %.2f' % res.model.sigma2)
    print('Average Model Size: %.2f' % res.model_sizes.mean())
