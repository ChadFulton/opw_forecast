from __future__ import division

import numpy as np
import pandas as pd
import re

class OPWData(pd.DataFrame):
    def __init__(self, *args, **kwargs):
        super(OPWData, self).__init__(*args, **kwargs)
    
        # Setup the index
        #self.index = pd.date_range(start='1960-01-01', end='2011-06-01', freq='MS')
        self.index = pd.date_range(start='1960-01-01', periods=self.shape[0], freq='MS')

        # Get the state column names
        self._state_emp_columns = self.columns[8:]
        self._state_growth_columns = [
            ('%s_emp_growth' % re.sub(' ?\(SA ?, ?[Tt]hous.?\) ?', '', state).lower()).replace(' ', '_')
            for state in self._state_emp_columns
        ]
        self._state_growth_columns[27] = '_'.join(self._state_growth_columns[27].split('_')[1:])
        self._state_growth_columns[49] = self._state_growth_columns[49].replace('.', '')
        self._state_names = [' '.join(col.split('_')[:-2]).title() for col in self._state_growth_columns]
        self._state_names[-1] = self._state_names[-1].upper()
        
        # Transform some data
        self['ff'] = self['Federal Funds Rate']
        self['sp500_return'] = self.growth_rate(self['S&P 500'])
        self['term_spread'] = (self['Ten Year Bond'] - self['Three Month Bill'])
        self['agg_emp_growth'] = self.growth_rate(self['U.S. Employment'])
        self['agg_ip_growth'] = self.growth_rate(self['U.S. Industrial Production'])

        # Setup the state employment growth columns
        self[self._state_growth_columns] = self.growth_rate(self[self._state_emp_columns])
        
        # Add an intercept
        self['intercept'] = np.ones(self.shape[0])
        
        # Get additional column names
        self._endog_columns = ['NBER Indicator']
        self._endog_names = ['NBER Indicator']
        self._financial_columns = ['ff', 'sp500_return', 'term_spread']
        self._financial_names = ['Federal Funds Rate', 'S&P 500 Return', 'Term Spread']
        self._growth_columns = ['agg_emp_growth', 'agg_ip_growth'] + self._state_growth_columns
        self._growth_names = ['Agg. Emp. Growth', 'Agg. IP Growth'] + self._state_names
        self._exog_columns = ['intercept'] + self._financial_columns + self._growth_columns
        self._exog_names = ['Constant'] + self._financial_names + self._growth_names

    def growth_rate(self, data, delta=3):
        return (data / data.shift(delta) - 1)*100
    
    def get_endog_array(self, lag, **kwargs):
        # 3   because the first three rows have NaN's
        # lag because when our exog is at time t, the endog
        #     needs to be at t+h
        # 1   because we use both time t data (financial variables)
        #     and time t-1 data (growth rates) to forecast s_{t+h}
        #     thus the start of our endog must be h+1 so that there
        #     are enough rows of the t-1 exog.
        start_idx = 3 + lag+1
        endog = pd.DataFrame(self.ix[start_idx:,self._endog_columns])
        
        return np.array(endog, **kwargs)

    def get_exog_array(self, lag, **kwargs):
        # Make a temporary dataframe and get rid of missing rows
        exog = pd.DataFrame(self.ix[:,self._exog_columns])
        # Since we want our growth rates to be time t-1 data, shift them
        exog[self._growth_columns] = exog[self._growth_columns].shift(1)
        # 3   because the first three rows have NaN's
        # 1   because we shifted the growth rate columns by 1
        start_idx = 3+1
        # -lag because we don't have endog rows (s_{t+h}) for the
        #     last h exog rows.
        end_idx = -lag
        
        return np.array(exog.ix[start_idx:end_idx,:], **kwargs)

    @property
    def endog(self):
        return self.get_endog_array(lag=1, dtype=np.int32)

    @property
    def exog(self):
        return self.get_exog_array(lag=1)

    @property
    def endog_names(self):
        return self._endog_names

    @property
    def exog_names(self):
        return self._exog_names
