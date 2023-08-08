"""
ADD FLAGS
"""

import numpy as np
import pandas as pd


def fITC(flagdata, latitude, **kwargs):
    '''
    INTEGRAL TURBULENCE CHARACTERISTICS TEST

    flagdata: data to include ITC flag
    latitude: latitude to calculate Coriolis effect
    kwargs: used to define `n', a dictionary of variable names
    '''
    n = {k: k for k in ['zL', 'zm', 'z0', 'ol', 'ustar', 'sigmaw']}
    n.update(**{k: v for k, v in kwargs.items() if k in n.keys()})

    if n['zL'] not in flagdata.columns:
        #flagdata[n['zL']] = (flagdata[n['zm']]-flagdata[n['z0']]) / flagdata.ol
        flagdata[n['zL']] = flagdata[n['zm']] / flagdata.ol

    flagdata['sTp_Tpstar_mo'] = 2 * np.abs(flagdata[n['zL']])**(1/8)

    flagdata.loc[(np.abs(flagdata[n['zL']]) < 0.032), 'sTp_Tpstar_mo'] = 1.3
    # flagdata.loc[(flagdata[n['zL']] < 0.032) and (flagdata[n['zL']] < 0.032), 'sTp_Tpstar_mo'] = 1.3

    # f: Coriolis parameter
    f = 2. * 2.*np.pi/(24.*60.*60.) * np.sin(np.deg2rad(latitude))
    
    cond_ = (flagdata[n['zL']] > -0.2) * (flagdata[n['zL']] < 0.4)
    flagdata.loc[cond_, 'sTp_Tpstar_mo'] = (
        0.21 * np.log(1. * f / flagdata[n['ustar']]) + 3.1)[cond_]
    
    flagdata['sw_ustar'] = flagdata[n['sigmaw']] / flagdata[n['ustar']]
    flagdata['itc_co2'] = np.floor(np.abs((flagdata.sTp_Tpstar_mo -
                                           flagdata.sw_ustar) / flagdata.sTp_Tpstar_mo) * 100)

    flagdata.loc[flagdata.sTp_Tpstar_mo < 10**-6, 'itc_co2'] = 0

    flagdata['fITC'] = 2
    flagdata.loc[flagdata.itc_co2 <= 100, 'fITC'] = 1
    flagdata.loc[flagdata.itc_co2 <= 30, 'fITC'] = 0

    return flagdata


def fSTA(flagdata, data2, cov_name='cov_wco2', bydate=False, one_dta_per=None, datafreq='30min'):
    '''
    ADD STATIONARITY FLAG TO DATASET

    flagdata: data to include stationarity flag
    data2: complementary data at a faster pace for comparison
    cov_name: variable used for comparison
    one_dta_per: input how many data2 observations are accounted per flagdata observation
    '''

    if bydate:
        """
        breaks = np.unique(data2[bydate].dt.ceil(
            '30min'), return_index=True)[1]
        breaks.sort()
        #breaks = breaks[1:] if breaks[0] == 0 else breaks
        
        flagdata['sum_wco2'] = np.array([np.nanmean(p) for p in np.split(
            data2[cov_name], breaks, axis=0)]).ravel()
        """
        data2[bydate] = data2[bydate].dt.ceil(datafreq)
        data2 = data2.groupby(bydate)[cov_name].apply(np.nanmean)
        data2 = data2.reset_index()
        data2.columns = [bydate, 'sum_wco2']
        flagdata = pd.merge(flagdata, data2, 'left', bydate)
        #flagdata['sum_wco2'] = np.array(data2.groupby("_group_")[cov_name].apply(np.nanmean)).ravel()

    else:
        if one_dta_per:
            flagdata['sum_wco2'] = np.nanmean(np.array(
                data2[cov_name]).reshape(-1, int(one_dta_per)), axis=1).ravel()[:len(flagdata)]
        else:
            flagdata['sum_wco2'] = np.nanmean(np.array(
                data2[cov_name]).reshape(-1, int(round(len(data2)/len(flagdata)))), axis=1).ravel()

    flagdata['stat_wco2'] = np.floor(
        np.abs((flagdata.sum_wco2 - flagdata[cov_name]) / flagdata[cov_name]) * 100.)

    flagdata.loc[flagdata['sum_wco2'] <= -9000., ['stat_wco2']] = 99999
    flagdata.loc[flagdata[cov_name] <= -9000., ['stat_wco2']] = 99999
    flagdata.loc[flagdata[cov_name]
                 > 99999, ['stat_wco2']] = 99999

    flagdata['fSTA'] = 2
    flagdata.loc[flagdata.stat_wco2 <= 100, 'fSTA'] = 1
    flagdata.loc[flagdata.stat_wco2 <= 30, 'fSTA'] = 0
    return flagdata
