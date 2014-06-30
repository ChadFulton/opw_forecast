from __future__ import division
import numpy as np
import pandas as pd
import time
import data
import cyestimation
np.set_printoptions(precision=5, suppress=True)

if __name__ == '__main__':
    np.random.seed(12345)

    # Model Parameters
    lag = 1
    sigma2 = 10

    # Get the dataset
    raw = pd.read_excel('data/recession probit data.xlsx',
                        'nber_indicator.csv')
    df = data.OPWData(raw.ix[:raw.shape[0], :])

    # Parameters
    G0 = 20000
    G = 20000

    start_time = time.time()  # timing

    est = cyestimation.Simulate(df.endog[:, 0].astype(np.int32), df.exog, sigma2, G0, G)
    est.advance()

    end_time = time.time()
    elapsed = end_time - start_time

    print('Runtime of %d minutes and %.2f seconds'
          % (elapsed // 60, elapsed % 60))

    print('Number of draws to convergence: %d' % G0)
    print('Number of draws after convergence: %d' % G)
    print('Prior VC matrix for model parameters is: %.2f' % sigma2)
    print('Average Model Size: %.2f' % np.array(est.indicators)[:, G0+1:].sum(0).mean())
