import os
import numpy as np
from Lib.OpenFlux.scripts import common as tcom
import pathlib
import warnings
# current file path
cfp = pathlib.Path(__file__).parent.resolve()

def default_lag(length=711, diameter=5.3, pump=15, dt=20):
    # default time lag in number of data points
    return int(np.round((length * (np.pi * (diameter/2)**2) * (10**-6) / pump) * 60 * dt))

def time_lag(self, default=0, *args, **kwargs):
    default = default if default > 0  else default_lag()
    #"scripts/RFlux-scripts/tlag_detection.R"
    self.loc[:, 'co2'] = self['co2'].shift(-default)
    #self.loc[:, 'co2_conc'] = self.co2_conc.shift(-default)

    f = tuple([(np.isnan(self.co2)==False) * (np.isnan(self.w)==False)])
    x = np.array(self.loc[f, 'co2'])
    y = np.array(self.loc[f, 'w'])

    try:
        tlag_opt = tcom.LazyCallable(os.path.join(cfp, '..', 'RFlux-scripts/tlag_detection.R'), 'tlag_detection').__call__(
                x, y, mfreq=20)
        tlag_opt = int(tlag_opt[3][0])
        self.loc[:, 'co2'] = self.co2.shift(-tlag_opt)
        #self.loc[:, 'co2_conc'] = self.co2_conc.shift(-tlag_opt)
    except Exception as e:
        tlag_opt = default
        warnings.warn(f'{str(e)}.\nError when calculating time lag, default value ({tlag_opt}) used.')
    
    '''
    if self.ylag > 0:
        # ignore first timestamp since we jumped 5 observations in the beginning
        flux.wvlst = flux.wvlst[:, (period-self.ylag):]
        # last part depends on the time delay for max corr between co2 and w
        self.t = self.t[(period-self.ylag):]
    '''
    return self