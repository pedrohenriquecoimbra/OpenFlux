import numpy as np
import scipy as sp
import pandas as pd

import os
import numpy as np
import pandas as pd
from Lib.OpenFlux.scripts import common as tcom
import pathlib
# current file path
cfp = pathlib.Path(__file__).parent.resolve()


def tilt_rotation(self, save_date=None, pre_calculus=False, *args, **kwargs):
    #assert 'tilt_kwargs' in kwargs.keys(), 'Assign options for tilt correction.'
    if 'tilt_kwargs' in kwargs.keys():
        kwargs.update(kwargs['tilt_kwargs'])

    tkw_ = kwargs.pop('method', None)
    if tkw_:
        tkw_ = {"method": tkw_}
    '''
    if 'avg_period' in kwargs.keys():
        __date_range = (min(flist(kwargs['avg_period'])), max(flist(kwargs['avg_period'])))
        assert (file_date >= __date_range[0]) and (file_date <= __date_range[1]), \
        'Date outside of averaging period. Date is {} and averaging period range is {}.'.format(file_date, __date_range)
        loopvar = [p for p in kwargs['avg_period'] if file_date in p][0]
    
    else:
        print('Warning! No averaging period given for level 5 (tilt correction) so using the opened dates ({})'.format(
            '{} to {}'.format(min(file_date).strftime('%Y-%m-%d %H:%M'), max(file_date).strftime('%Y-%m-%d %H:%M')) 
            if (isinstance(file_date, (list, np.ndarray))) and (len(file_date) > 1) 
            else file_date.strftime('%Y-%m-%d %H:%M')
        ))
        loopvar = file_date if isinstance(file_date, (list, np.ndarray)) else [file_date]
    '''
    if save_date:
        save_path = kwargs['setup_path'] + str(kwargs['setup_name']).format(
            '{}_{}'.format(min(save_date).strftime('%Y%m%d%H%M'),
                           max(save_date).strftime('%Y%m%d%H%M'))
            if (isinstance(save_date, (list, np.ndarray, pd.DatetimeIndex)))
            else save_date.strftime('%Y%m%d%H%M'))
        #save_date.strftime('%Y%m%d%H%M'))

        if (os.path.exists(save_path) == False) or (
                ('__iden__' in kwargs.keys()) and (not tcom.readable_file(save_path).check_id(kwargs['__iden__']))):
            if not pre_calculus:
                print('Warning! Using opened dates ({}) for calculating tilt as well as returning corrected data.'.format(
                    '{} to {}'.format(min(save_date).strftime(
                        '%Y-%m-%d %H:%M'), max(save_date).strftime('%Y-%m-%d %H:%M'))
                    if (isinstance(save_date, (list, np.ndarray))) and (len(save_date) > 1)
                    else save_date.strftime('%Y-%m-%d %H:%M')
                ))

            _, _, _, _theta, _phi = tiltcorrection(
                self['u'], self['v'], self['w'], **tkw_)

            '''
            if len(loopvar)==1:
                _, _, _, _theta, _phi = tiltcorrection(self['u'], self['v'], self['w'], method=kwargs['method'])
            else:
                _tmp = PPDataset.openinloop(1, loopvar, verbosity=0, **kwargs).data
                #self['u'], self['v'], self['w'] = tiltcorrection(self['u'], self['v'], self['w'], method='2r')
                _, _, _, _theta, _phi = tiltcorrection(_tmp['u'], _tmp['v'], _tmp['w'], method=kwargs['method'])
            '''
            if kwargs['setup_name']:
                tcom.readable_file(save_path, angles={'theta': _theta, 'phi': _phi},
                                   **{k: v for k, v in kwargs.items() if k in ['__text__', '__iden__']}).dump()

            if pre_calculus:
                return

        tilt_setup = tcom.readable_file(save_path).load().to_dict()['angles']
        _theta = tilt_setup['theta']
        _phi = tilt_setup['phi']
    else:
        _, _, _, _theta, _phi = tiltcorrection(
            self['u'], self['v'], self['w'], **tkw_)

    self.loc[:, 'u'], self.loc[:, 'v'], self.loc[:, 'w'], _, _ = tiltcorrection(
        self.loc[:, 'u'], self.loc[:, 'v'], self.loc[:, 'w'], **tkw_, _theta=_theta, _phi=_phi)

    return self

# Wilczak et al., 2001
# double (and triple) rotation should be done in loops of 30 minutes
def wilczak2001_2r(u, v, w, _theta=None, _phi=None, verbosity=0):
    #first rotation
    if _theta is None:
        _theta = np.arctan(np.nanmean(v)/np.nanmean(u))
    u1 = u * np.cos(_theta) + v * np.sin(_theta)
    v1 = -u * np.sin(_theta) + v * np.cos(_theta)
    w1 = w

    #second rotation
    if _phi is None:
        _phi = np.arctan(np.nanmean(w1)/np.nanmean(u1))
    u2 = u1 * np.cos(_phi) + w1 * np.sin(_phi)
    v2 = v1
    w2 = -u1 * np.sin(_phi) + w1 * np.cos(_phi)
    
    if verbosity > 0: print(np.nanmean(w), np.nanmean(w1), np.nanmean(w2))
    
    return u2, v2, w2, _theta, _phi


def wilczak2001_3r(u, v, w, _psi=None, verbosity=0):
    u2, v2, w2 = wilczak2001_2r(u, v, w)
    
    #third rotation
    if _psi is None:
        _psi = np.arctan((2 * np.nanmean(v2 * w2)) / (np.nanmean(v2**2) - np.nanmean(w2**2)))
    u3 = u2
    v3 = v2 * np.cos(_psi) + w2 * np.sin(_psi)
    w3 = -v2 * np.sin(_psi) + w2 * np.cos(_psi)
    
    if verbosity > 0: print(np.nanmean(w3))
    return u3, v3, w3, psi


def planarfit(u, v, w, verbosity=0):
    meanU = np.nanmean(u)
    meanV = np.nanmean(v)
    meanW = np.nanmean(w)

    def findB(meanU, meanV, meanW):
        su = np.nansum(meanU)
        sv = np.nansum(meanV)
        sw = np.nansum(meanW)

        suv = meanU * meanV
        suw = meanU * meanW
        svw = meanV * meanW
        su2 = meanU * meanU
        sv2 = meanV * meanV

        H = np.matrix([[1, su, sv], [su, su2, suv], [sv, suv, sv2]])
        g = np.matrix([sw, suw, svw]).T
        x = sp.linalg.solve(H, g)

        b0 = x[0][0]
        b1 = x[1][0]
        b2 = x[2][0]
        return b0, b1, b2
    
    b0, b1, b2 = findB(meanU,meanV,meanW)

    Deno = np.sqrt(1 + b1 **2 + b2 **2)
    p31 = -b1 / Deno
    p32 = -b2 / Deno
    p33 = 1 / Deno

    cosγ = p33 / np.sqrt(p32**2+p33**2)
    sinγ = -p32 / np.sqrt(p32**2 + p33**2)
    cosβ = np.sqrt(p32**2 + p33**2)
    sinβ = p31

    R2 = np.matrix([[1, 0, 0],
                    [0, cosγ, -sinγ],
                    [0, sinγ, cosγ]])
    R3 = np.matrix([[cosβ, 0, sinβ],
                    [0, 1, 0],
                    [-sinβ, 0, cosβ]])

    A0 = R3.T * R2.T * [[meanU], [meanV], [meanW]]

    α = np.arctan2(A0[1].tolist()[0][0],
                   A0[0].tolist()[0][0])

    R1 = np.matrix([[np.cos(α), -np.sin(α), 0],
                    [np.sin(α), np.cos(α), 0], 
                    [0, 0, 1]])

    A1 = R1.T * ((R3.T * R2.T) * np.matrix([u, v, w - b0]))

    U1 = np.array(A1[0])[0]
    V1 = np.array(A1[1])[0]
    W1 = np.array(A1[2])[0]
    
    if verbosity > 0: print(np.nanmean(w), np.nanmean(W1))
    
    if type(u) == pd.Series: U1 = pd.Series(U1)
    if type(v) == pd.Series: V1 = pd.Series(V1)
    if type(w) == pd.Series: W1 = pd.Series(W1)
    
    return U1, V1, W1

    
def tiltcorrection(*args, method='2r', **kwargs):
    if method == '2r': return wilczak2001_2r(*args, **{k: v for k, v in kwargs.items() if k in ['u', 'v', 'w', '_theta', '_phi', 'verbosity']})
    if method == '3r': return wilczak2001_3r(*args, **kwargs)
    if method == 'pf': return planarfit(*args, **kwargs)
    else:
        print("Choose between: double rotation (2r), triple rotation (3r), and planar fit (pf).")
        return

def printtiltcorrection():
    return
    """
    figprops = dict(figsize=(3*16/3, 2*3), dpi=72)
    fig = pyplot.figure(**figprops)
    window = (100,200)
        
    pyplot.subplot(211)
    pyplot.plot(r.df.TIMESTAMP, r.df.W)
    #pyplot.plot(df_site.TIMESTAMP[tlag:],  (w-df_site.W[tlag:]), 'r')
    pyplot.axhline(np.nanmean(r.df.W), color='k')
    pyplot.axhline(np.nanmean(r.y), color='r')
    pyplot.annotate(np.nanmean(r.df.W), (np.nanmin(r.df.TIMESTAMP), 0))

    pyplot.subplot(212)
    pyplot.plot((r.df.TIMESTAMP[0:])[window[0]:window[1]],  (r.y[0:])[window[0]:window[1]], ':k', label="planar fit")
    pyplot.plot((r.df.TIMESTAMP[0:])[window[0]:window[1]],  (r.df.W[0:])[window[0]:window[1]], 'k', label="measured")
    pyplot.plot((r.df.TIMESTAMP[0:])[window[0]:window[1]],  ((r.y[0:])[window[0]:window[1]]-(r.df.W[0:])[window[0]:window[1]]), ':r')
    pyplot.legend(loc='lower right')
    pyplot.show()
    
    if False:
        import os 
        os.environ['R_HOME'] = 'C:/Program Files/R/R-3.6.3'
        import rpy2.robjects as robjects
        from rpy2.robjects.packages import importr
        #robjects.r('getwd()')
        robjects.r.source('tlag_detection.R')
        importr("zoo")
        importr("data.table")

        tlag_detection = robjects.globalenv['tlag_detection']
        tlag_detection(x=robjects.vectors.FloatVector(df_site.CO2), 
                       y=robjects.vectors.FloatVector(df_site.W),
                       mfreq=20)
    """