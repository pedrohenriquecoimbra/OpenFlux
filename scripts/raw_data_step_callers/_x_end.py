import numpy as np
import pandas as pd
import warnings

def intepolate_small_gaps(self, cols=["w", "h2o", "co2", "t_sonic"], *args, smallgap=5*20, **kwargs):
    N = len(self)

    for c in cols:
        a_ = pd.Series(self[c]).isnull()
        b_ = a_.ne(a_.shift()).cumsum()
        c_ = b_.map(b_.value_counts()).where(a_)
        
        signan = np.isnan(self[c])
        if sum(signan) >= (len(signan) - 1):
            warnings.warn(f"Only nan for {c}.")
            continue
        gaps_ignore = signan * (c_ > smallgap)
        self.loc[:, c] = np.interp(np.linspace(0, 1, N),
                                   np.linspace(0, 1, N)[signan == False],
                                   self[c][signan == False])
        
        self[c].mask(gaps_ignore, np.nan, inplace=True)
    
    return self
