"""
Check: https://github.com/regeirk/pycwt/blob/master/pycwt/sample/simple_sample.py

In this example we will load the NINO3 sea surface temperature anomaly dataset
between 1871 and 1996. This and other sample data files are kindly provided by
C. Torrence and G. Compo at
<http://paos.colorado.edu/research/wavelets/software.html>.
"""
import multiprocess as mp

from distutils.spawn import spawn
import warnings
from genericpath import isfile
import os
from random import random, randint
import numpy as np
import scipy as sp
import scipy.io.wavfile
import pandas as pd
import pycwt as wavelet
import matplotlib
from matplotlib import pyplot
from . import get_data, get_rawdata, common as tcom
#from scripts.tiltcorrection import tiltcorrection
import datetime
import copy
import pickle
import time
import gc

from itertools import chain  # for sound only
import IPython  # for sound only


import importlib
importlib.reload(tcom)
importlib.reload(get_data)


class FluxDataset(tcom.datahandler):
    def __init__(self, **kwargs):
        "fluxes"
        self.id_vars = ['TIMESTAMP']
        self.col_vars = ['wv']#, 'qc_wv', 'wv_std']
        self.timelst = None
        self.TIMESTAMP = None
        self.avg_period = None
        #self.ec = None
        #self.qc_ec = None
        self.wv = None
        #self.qc_wv = None
        self.wv_std = None
        self.meta = {}
        self.wvlst = []
        self.sdlst = []
        self.__dict__.update(**kwargs)

    def print(self):
        print('\n'.join("%s: %s" % item for item in vars(
            self).items() if item[0] not in ['timelst', 'TIMESTAMP', 'ec', 'qc_ec', 'wv', 'qc_wv', 'wv_std', 'wvlst']))
        return
            
    def save_flux(i, t_, xname, yname, prefix,
                      output_path="./wavelets_for_flux/data/tmp/flux/", dt=0.05,
                      period=30*60, pfreq=30*60, dj=1/2, max_period=1*60*60,
                      filefreq=30, overwrite=False, ignoreerr=False,
                      readkwargs={},  
                      savefig=False, **kwargs):
            #prefix = prefix + "_" + str(t_)[:10] + ".dat"
            prefix = prefix.format(min(t_).strftime('%Y%m%d%H%M'))
            print(prefix, end='\r')
            if overwrite == False and os.path.isfile(output_path+prefix):
                        return None
            
            tloop = time.time()

            # Get raw data
            #print(prefix, str(t_)[:10], end='\r')

            #t_1 = t_ + pd.Timedelta(24, 'hr') - \
            #    datetime.timedelta(seconds=0.05)
            try:
                if 'mother' in kwargs.keys():
                    readkwargs['mother'] = kwargs['mother']
                dfraw = prepare_wavelet_inside_coi(d0=min(t_), d1=max(t_),
                                                   dt=dt, max_period=max_period, filefreq=filefreq, **readkwargs)
                if dfraw.data.empty:
                    print('No dataframe was saved for {}'.format(prefix))
                    return None
                dfraw.data = dfraw.data.sort_values(by=['TIMESTAMP'])
                """
                for c in [xname, yname]:
                    p = np.polyfit(np.linspace(1, len(dfraw.data[c]), len(dfraw.data[c])), dfraw.data[c], 1)
                    dat_notrend = dfraw.data[c] - np.polyval(p, np.linspace(1, len(dfraw.data[c]), len(dfraw.data[c])))
                    dfraw.data[c] = dat_notrend
                    #std = dat_notrend.std()  # Standard deviation
                    #var = std ** 2  # Variance
                    #dfraw.data[c] = dat_notrend / std  # Normalized dataset
                    #del dat_notrend#, std, var
                """
                #dfraw.data[xname] = dfraw.data[xname] - np.mean(dfraw.data[xname])
                #dfraw.data[yname] = dfraw.data[yname] - np.mean(dfraw.data[yname])
                
                #dfraw = get_rawdata.FluxTowerRawData.__get__(rawfc,
                #    path=rawdata_path, file_pattern='{TIMESTAMP}', file_list=datarange)
            except AttributeError as e:
                warnings.warn(str(e) + ' ({})'.format(prefix))
                if ignoreerr:
                    return None
                else:
                    raise e
            tloop_rw = time.time()

            if dfraw.data.empty:
                print('No dataframe was saved for {}'.format(prefix))
                return None
            else:
                # Crosswavelet
                #dfraw = dfraw.interpolate(cols=[xname, yname])
                wv_x = dfraw.to_wv_class(
                    xname=xname, yname=yname, dj=dj, **kwargs).crosswavelet(max_period=max_period)
                tloop_wv = time.time()

            if savefig:
                wv_x.plot(significance_level=None)
                pyplot.axhline(np.log2(max_period),
                               ls=':', color='k')
                pyplot.axvline(min(t_), ls=':', color='k')
                pyplot.axvline(min(t_)+datetime.timedelta(days=1), ymax=np.log2(max_period),
                               ls=':', color='k')
                pyplot.savefig(output_path + '.'.join(prefix.split('.')[:-1]) + ".png")
                pyplot.close()
            
            tloop_sv = time.time()

            # Flux from wavelets
            #flux_inputs = {'period': int((1/dt)*period), 'pfreq': int((1/dt)*pfreq)}
            #wv_Fx = wv_x.to_flux(**flux_inputs, verbosity=1) if i == 0 else wv_x.to_flux(**flux_inputs)
            
            wv_Fx = wv_x.to_flux()
            
            wv_Fx = wv_Fx.screen(
                valarr = (pd.DatetimeIndex(wv_Fx.TIMESTAMP) >= min(t_)) * (pd.DatetimeIndex(wv_Fx.TIMESTAMP) < max(t_)),
                affected=['TIMESTAMP', 'wvlst'])
            
            if period:
                wv_Fx = wv_Fx.average_flux(set_avg_period=period)
            wv_Fx = wv_Fx.integrate_flux(max_freq=pfreq, coi=wv_x.coi_mask)
            
            tloop_fx = time.time()

            wv_Fx.wvlst = list(wv_Fx.wvlst)
            wv_Fx.sdlst = list(np.zeros(np.array(wv_Fx.wvlst).shape))
            wv_Fx.dump(path=output_path+prefix)

            tloop_dp = time.time()
            print('{} ({}, {}, {}, {}, {}, {} s)      '.format(
                prefix,
                round(tloop_rw-tloop, 2),
                round(tloop_wv-tloop, 2),  
                round(tloop_sv-tloop, 2),  
                round(tloop_fx-tloop, 2),
                round(tloop_dp-tloop, 2),
                round(time.time()-tloop, 2)), end='\r')
            return None
        
    def savefluxinloop(*args, **kwargs):        
        kwargs.update({'result': False})
        return tcom.multiprocess_framework(FluxDataset.save_flux, *args, **kwargs)
    
    def load_flux(i_, t_, prefix, read_path="./wavelets_for_flux/data/tmp/flux/"):
            filepath = read_path + \
                prefix+"_"+str(t_)[:10] + ".dat"

            if os.path.isfile(filepath):
                wv_Fi = FluxDataset().load(path=filepath)
            else:
                return None
            return {i_: wv_Fi}
    
    def loadfluxinloop(*args, **kwargs):
        return tcom.multiprocess_framework(FluxDataset.load_flux, *args, **kwargs)

    def to_DataFrame(self, all_freqs=False, **kwargs):
        fluxdata = {}

        if all_freqs:
            for i, f in enumerate(self.period):
                self.__dict__['wv' if 'wv' not in kwargs.keys() else kwargs['wv'] + '_freq_'+str(f)[:10]] = np.array(self.wvlst)[:, i]
                self.col_vars = self.col_vars + ['wv' if 'wv' not in kwargs.keys() else kwargs['wv'] + '_freq_'+str(f)[:10]]

        for k in self.__dict__.keys():
            if k in self.id_vars or k in self.col_vars or k in kwargs.keys():
                v = self.__dict__[k]
                if v is not None:
                    if kwargs and k in kwargs.keys():
                        fluxdata.update({kwargs[k]: v})
                    else:
                        fluxdata.update({k: v})

        fluxdf = pd.DataFrame(fluxdata)
        return fluxdf

    def collapse_flux(self, breaks, coi=None, verbosity=0):
        cself = FluxDataset(**self.__dict__)

        breaks = list(set([tcom.nearest(cself.period, b)[
            0] for b in breaks]))
        breaks.sort()

        if min(breaks) > 0:
            breaks = [0] + breaks
        if max(breaks) < max(cself.period):
            breaks = breaks + [max(cself.period)]
        relbreaks = []
        wvlst = []
        sdlst = []
        period = []
        coi_flag = []

        for i in range(len(breaks)-1):
            f0_ = tcom.nearest(cself.period, breaks[i])[
                0]
            f0 = list(cself.period).index(f0_)
            f1_ = tcom.nearest(cself.period, breaks[i+1])[
                0]
            f1 = list(cself.period).index(f1_)
            wvlst += [np.array(cself.wvlst)[:, f0:f1].sum(axis=1)]
            sdlst += [np.array(cself.sdlst)[:, f0:f1].sum(axis=1)]
            period += [f1_]
            relbreaks += [(f0_, f1_)]
            #coi_flag += (~np.array(coi)[:, f0:f1]).sum(
            #    axis=1).astype(bool) if coi != None else None

        if verbosity > 0:
            print(relbreaks)

        cself.wvlst = list(np.array(wvlst).T)
        cself.sdlst = list(np.array(sdlst).T)
        cself.coi_flag = coi_flag
        cself.period = np.array(period)
        return cself  # pwv, qc_pwv, pfreq

    def integrate_flux(self, max_freq=60*30, min_freq=None, coi=None):
        if max_freq != None:
            self.wv_flux_max_int_freq = tcom.nearest(self.period, max_freq)[0]
            f1 = list(self.period).index(self.wv_flux_max_int_freq)
            #[i for i, x in enumerate(self.period) if x == self.wv_flux_max_int_freq][0]
        else:
            self.wv_flux_max_int_freq = f1 = None

        if min_freq == None:
            self.wv_flux_min_int_freq = f0 = 0
        else:
            self.wv_flux_min_int_freq = tcom.nearest(self.period, min_freq)[0]
            f0 = list(self.period).index(self.wv_flux_min_int_freq)
        # use list.index(pfreq) ^

        self.wv = np.array(self.wvlst)[:, f0:f1].sum(axis=1)
        self.wv_std = np.array(self.sdlst)[:, f0:f1].sum(axis=1)

        self.coi_flag = (~np.array(coi)[:, f0:f1]).sum(axis=1).astype(bool) if coi is not None else None

        '''qc_pwv = np.array(self.wvlst).real[:, f1:].sum(axis=1)
        qc_pwv[qc_pwv != 0] = -1  # np.nan or 0?
        qc_pwv += 1
        self.qc_wv = qc_pwv'''
        return self #pwv, qc_pwv, pfreq
        
    def average_flux(self, wave=None, t0=None, set_avg_period=None,
                     verbosity=0, **kwargs):
        aself = FluxDataset(**self.__dict__)

        t = aself.t if 't' in aself.__dict__.keys() else aself.TIMESTAMP
        t0 = 0 if t0 == None else list(t).index(t0)
        wave = np.array(aself.wvlst) if wave is None else np.array(wave)
        
        '''loop where if 1dim -> 2dim, then mean, so back to 1dim'''
        block = np.array(pd.to_datetime(t).round(str(set_avg_period/60)+'Min').strftime('%Y%m%d%H%M%S.%f').astype(float))
        block = np.append(block.reshape(-1,1), wave, axis=1)

        # block should (already) be sorted, do it just in case
        # block

        aself.TIMESTAMP = pd.to_datetime(np.unique(block[:, 0]).astype(str), format='%Y%m%d%H%M%S.%f')
        aself.wvlst = [np.nanmean(p, axis=0) for p in np.split(block[:, 1:], np.unique(block[:, 0], return_index=True)[1][1:], axis=0)]
        aself.sdlst = [np.nanstd(p, axis=0) for p in np.split(block[:, 1:], np.unique(block[:, 0], return_index=True)[1][1:], axis=0)]
        aself.avg_period = set_avg_period
        return aself
    """
    def _average_flux(self, wave=None, t0=None, shrinking_rate=2, set_avg_period=None,
                     verbosity=0, **kwargs):
        aself = FluxDataset(**self.__dict__)

        if set_avg_period:
            shrinking_rate = int(set_avg_period / aself.dt) if aself.avg_period == None else int(set_avg_period / aself.avg_period)
        t = aself.t if 't' in aself.__dict__.keys() else aself.TIMESTAMP
        t0 = 0 if t0 == None else list(t).index(t0)
        wave = np.array(aself.wvlst) if wave is None else np.array(wave)
        #period = period / self.avg_period
        
        '''loop where if 1dim -> 2dim, then mean, so back to 1dim'''
        
        #print(t[t0], len(t[t0:]), shrinking_rate)
        max_shrink = int(shrinking_rate*(len(t)//shrinking_rate))
        t = np.array(t)[:max_shrink]
        wave = np.array(wave)[:max_shrink, :]
        #qcwv = np.array(aself.qc_wv)[:max_shrink]

        tf = np.min(np.array(t).reshape(-1, shrinking_rate), axis=1)
        fluxlst = np.nanmean(wave.reshape(
            (wave.shape[0]//shrinking_rate, shrinking_rate, wave.shape[1])), axis=1)
        sdlst = np.nanstd(wave.reshape(
            (wave.shape[0]//shrinking_rate, shrinking_rate, wave.shape[1])), axis=1)
        #qcf = np.nanmean(qcwv.reshape(-1, shrinking_rate), axis=1)

        aself.TIMESTAMP = tf
        #aself.qc_wv = qcf
        aself.wvlst = fluxlst
        aself.sdlst = sdlst
        aself.avg_period = shrinking_rate * aself.dt if aself.avg_period == None else aself.avg_period * shrinking_rate
        return aself
    """
    def ecfreqs(self):
        _, self.ecfreqs = wavelet_nearest_freq(self.meta['wvlst'], self.ec, self.meta['period'])
        return self
        
    def plot(self, wvclass=None, highlight=None, **kwargs):
        if wvclass is None:
            plot_flux(flux=self, highlight=highlight, **kwargs)            
        else:
            plot_waveletflux(flux=self, **kwargs)
        return

    def plot_Fwavelets(*args, position, **kwargs):
        assert len(args)==len(position), 'Position index and args do not match.'
        for i, a_ in enumerate(args):
            pyplot.subplot(position[i], sharex=pyplot.subplot(position[i-1])) if i>0 else pyplot.subplot(position[i])
            FluxDataset.plot_Fwavelet(a_, **kwargs)
        return
    
    def plot_Fwavelet(self, **kwargs):
        pyplot.contourf(self.TIMESTAMP, np.log2(
            self.period), np.array(self.wvlst).T, **kwargs)
        pyplot.yticks([np.log2(60), np.log2(60*5), np.log2(60*30), np.log2(60*60), np.log2(self.period.max())],
                labels=["1 mn", "5 mn", "30 mn", "1 h", str(round(self.period.max()/(60*60), 1)) + " h"])
        return
    
    def screen(self, valarr, affected):
        for va in list(affected):
            self.__dict__[va] = np.array(self.__dict__[va])[valarr]
        return self

'''
class Feddycov(FluxDataset):
    """Eddy Covariance flux"""
    def eddycov(self, edpro_df, xname, tname="TIMESTAMP", qcname=None, **kwargs):
        if self.TIMESTAMP != None:
            print("Overwriting previous TIMESTAMP.")
        #kwargs.update({'add_columns': [xname, qcname]})
        #edpro_df = get_data.fr_gri().filter({'TIMESTAMP': self.timelst}).data

        self.TIMESTAMP = np.array(edpro_df[tname])
        self.ec = np.array(edpro_df[xname])

        if qcname:
            self.qc_ec = np.array(edpro_df[qcname])
        
        if kwargs:
            for k, v in kwargs.items():
                self.__dict__[k] = np.array(edpro_df[v])
        
        return self
'''

class wv:
    def __init__(self, x=None, y=None, dt=0.05, **kwargs):
        "wavelet"
        self.x = x
        self.y = y
        self.dt = dt
        self.dj = 1/4
        self.t = np.arange(0, self.x.size) * self.dt if x is not None else None
        self.mother = wavelet.Morlet(6)
        self.ylag = 0
        self.meta = {}
        self.__dict__.update(**kwargs)
    ''' 
    def __help__(self):
        print("run: \n> raw() \n> tiltcorrection() \n> wavelet(), plot_wavelet() \n> flux()")
    
    def tiltcorrection(self, **kwargs):
        _, _, self.y = tiltcorrection(self.df.U, self.df.V, self.df.W, method='pf')
        self.y = self.y.interpolate().values.squeeze()
        return self
    
    def lagcorrection(self):
        self.x = self.x[self.ylag:]
        self.y = sp.ndimage.interpolation.shift(self.y, self.ylag, cval=np.NaN)[self.ylag:]
    '''
    def findout_coi(self, lenght=None):
        dt = self.dt
        lenght = len(self.x) if lenght is None else (int(1/dt)*60*60*24)
        n0 = (lenght)#//2  # (20*60*60*24) = len(data24h)
        _coi = (n0 / 2 - np.abs(np.arange(0, n0) - (n0 - 1) / 2))
        _coi = self.mother.flambda() * self.mother.coi() * dt * _coi

        print("max. coi:", max(_coi),
            "\nmax. frequency calculated (J=16/dj):", (2 * dt / self.mother.flambda()) * 2 ** 16,
              "\n", ((2 * dt / self.mother.flambda()) * 2 ** 16) / (dt*60*60), "h")
        return max(_coi)

    def crosswavelet(self, max_period=None, slices=1, unique_id=False, ** kwargs):
        check_xy_same = np.allclose(self.x, self.y)
        J = int(np.round(np.log2(max_period * (1/self.dt) / 2) / self.dj)) if max_period else None
        #if J:
        #    print('{} h ({})'.format(round((2 * self.dt * 2**(J * self.dj)) / (60*60), 2), np.round(2 * self.dt * 2**(J * self.dj))), end='\r')

        self.meta.update(kwargs)
        _kwargs = {k: v for k, v in self.__dict__.items() if k in ['dt', 'dj', 's0', 'J', 'wavelet', 'freqs']}
        _kwargs.update({'J': J, 'wavelet': self.mother}) # 's0': 2*self.dt, 
        #_kwargs.update(kwargs)
        #_kwargs = {key: _kwargs[key] for key in _kwargs.keys() & {'dt', 'dj', 's0', 'J', 'wavelet', 'freqs'}}  # not 'signal'
                
        for i in range(slices):
            unique_id = str(randint(0, 999)).zfill(4) if unique_id is True else unique_id

            file_path = "./wavelets_for_flux/data/tmp/crosswavelet/" +  str(unique_id) + '_' + \
                str(slices).zfill(2) + '_' + str(i).zfill(2) + ".crosswv"
            
            if unique_id and os.path.exists(file_path):
                W12, transform, freq, sj, coi = pickle.load(
                    open(file_path, 'rb'))

                self.W12 = W12 if i == 0 else np.append(self.W12, W12, axis=1)
                self.transform_wave = transform
                self.period = 1/freq
                self.scale = sj
                self.coi = coi if i == 0 else np.append(self.coi, coi)
            else:
                #y1_normal = (self.x - self.x.mean()) #/ self.x.std()
                #y2_normal = (self.y - self.y.mean()) #/ self.y.std()

                y1_normal = np.hsplit(self.x, slices)[i]
                if check_xy_same:
                    y2_normal = y1_normal
                else:
                    y2_normal = np.hsplit(self.y, slices)[i]
                
                W1, sj, freq, coi, _, _ = wavelet.cwt(y1_normal, **_kwargs)
                if check_xy_same:
                    W2 = W1
                else:
                    W2, sj, freq, coi, _, _ = wavelet.cwt(y2_normal, **_kwargs)
                #W12 = np.real(np.array(W1 / np.matrix(sj **0.5).T) * np.array(W2 / np.matrix(sj **0.5).T).conj()) * self.dt * (self.dj / self.Cdelta)
                
                W12 = W1 * W2.conj()
                self.W12 = W12 if i == 0 else np.append(
                    self.W12, W12, axis=1)
                #WCT = W12 / scales
                #WCT = WCT * self.dt * self.dj / self.Cdelta
                
                scales = np.ones([1, self.x.size]) * sj[:, None]
                self.transform_wave = (1 / sj[:, None]) * self.dt * self.dj / \
                    self.mother.cdelta
                
                self.period = 1/freq
                self.scale = sj
                self.coi = coi if i == 0 else np.append(self.coi, coi)

                if unique_id:
                    pickle.dump([W12, self.transform_wave,
                                freq, sj, coi], open(file_path, 'wb'))
            
        coi_mask = np.ones(self.W12.shape, dtype=bool)
        for i in range(len(self.t)):
            coi_period_threshold = list(self.period).index(
                tcom.nearest(self.period, self.coi[i])[0])
            for j in range(coi_period_threshold, len(coi_mask)):
                coi_mask[j][i] = False
                
        self.coi_mask = coi_mask
        
        #alpha, _, _ = wavelet.ar1(self.x)  # Lag-1 autocorrelation for red noise
        #signif, _ = wavelet.significance(1.0, self.dt, self.scale, 0, alpha=alpha, significance_level=0.95, wavelet=self.mother)

        #self.plot_preps = {'glbl_power': glbl_power, 'fft_theor': fft_theor, 'power': power, 'sig95': sig95}
        return self

    def significance_level(self, significance_level=0.95, **kwargs):
        a1, _, _ = wavelet.helpers.ar1(self.x)
        a2, _, _ = wavelet.helpers.ar1(self.y)
        Pk1 = wavelet.helpers.ar1_spectrum(1/self.period * self.dt, a1)
        Pk2 = wavelet.helpers.ar1_spectrum(1/self.period * self.dt, a2)
        dof = self.mother.dofmin
        PPF = sp.stats.chi2.ppf(significance_level, dof)
        signif = (self.x.std() * self.y.std() * (Pk1 * Pk2) ** 0.5 * PPF / dof)

        sig95 = np.ones([1, len(self.x)]) * signif[:, None]
        power = (np.abs(self.W12)) ** 2
        sig95 = power / sig95

        return a1, a2, sig95
        
    def to_flux(self, t0=None, period=None, pfreq=None, verbosity=0, **kwargs):
        #func_time = time.time()

        flux = FluxDataset(dt=self.dt)

        #t0 = 0 if t0 == None else list(self.t).index(t0)
        #period = int(1/self.dt * 60 * 30) if period is None else period
        #pfreq = int(period * self.dt) if pfreq is None else pfreq

        flux.wvlst = (np.apply_along_axis(lambda x: x * np.ravel(self.transform_wave), 0,
                                     self.W12) * self.coi_mask).T.real

        flux.period = self.period
        flux.TIMESTAMP = self.t
        #flux.qc_wv = self.qc
        return flux

    def to_wav(self, path="./wavelets_for_flux/data/tmp/wavelet.wav", sample=20*60, volume=32760, **kwargs):
        #def soundofecosystem(wave=np.empty(0), method="flux", show=0, volume=32760, freqrate=1121, freq_filter=(0, 100)):
        # int16: (-32768, +32767)

        soundwave = (self.W12 * self.transform_wave).T.real
        soundwave = soundwave[:sample]

        soundwave_ = []
        for per in range(int(len(soundwave)*self.dt)):
            soundwave_ = soundwave_ + \
                [soundwave[int(per/self.dt):int((per+1)/self.dt), :]]
        soundwave = np.array(soundwave)

        '''reduce noises'''
        def moving_average(x, w=3):
            return np.convolve(x, np.ones(w), 'valid') / w
        soundwave = np.apply_along_axis(
            moving_average, axis=0, arr=soundwave, w=60)
        soundwave = np.apply_along_axis(
            moving_average, axis=1, arr=soundwave, w=5)

        soundwave = np.kron(soundwave, np.ones((1, 48000//soundwave.shape[1])))
        sampletime, samplerate = soundwave.shape

        soundwave = np.ravel(soundwave)

        print("- %02i:%02d, freq.: %i (%i), range: (%.2f, %.2f)" % (
            sampletime//60, sampletime % 60, samplerate, len(soundwave), 
            np.nanmin(soundwave), np.nanmax(soundwave)))

        # deal with stereo .wav files
        #data = [d[0] if isinstance(d, np.ndarray) else d for d in data]
        #data = [d if (d > LOWrange) & (d < HIGHrange) else 0 for d in data]  # filter

        soundwave = soundwave * (volume / np.nanmax(soundwave))  # rescale

        #data = 0.5*data + 0.5*np.append(data[samplerate:], data[:samplerate]) # ghost, just for fun
        sp.io.wavfile.write(path, samplerate, soundwave.astype(np.int16))
        return
    
    def sound(self, sample=20*60, **kwargs):
        soundofecosystem(self.wave[:sample], **kwargs)
        return
        
    def dump(self, path="./wavelets_for_flux/data/tmp/wavelet.dat"):
        with open(path, 'wb') as file:
            pickle.dump(self, file)
        return self
        
    def load(self, path="./wavelets_for_flux/data/tmp/wavelet.dat"):
        with open(path, 'rb') as file:
            return(pickle.load(file))

    def plot(self, significance_level=None):
        Δt = (self.t[-1]-self.t[0]) / len(self.t)
        power = (np.abs(self.W12)) ** 2

        pyplot.contourf(self.t, np.log2(self.period), np.log2(power),
                        extend='both', cmap=pyplot.cm.viridis)
        extent = [self.t.min(), self.t.max(), 0, max(self.period)]
        if significance_level is not None:
            _, _, sig95 = self.significance_level(
                significance_level=significance_level)

            pyplot.contour(self.t, np.log2(self.period), sig95,
                           [-99, 1], colors='k', linewidths=1, extent=extent)
        pyplot.fill(np.concatenate([self.t, self.t[-1:] + Δt, self.t[-1:] + Δt,
                                    self.t[:1] - Δt, self.t[:1] - Δt]),
                    np.concatenate([np.log2(self.coi), [1e-9], np.log2(self.period[-1:]),
                                    np.log2(self.period[-1:]), [1e-9]]),
                    'k', alpha=0.3, hatch='x')
        pyplot.title('b) {} Wavelet Power Spectrum ({})'.format(
            'co2-ish', self.mother.name))
        pyplot.ylabel('Period')
        pyplot.ylim(np.log2([self.period.min(), self.period.max()]))
        pyplot.yticks([np.log2(60), np.log2(60*5), np.log2(60*30), np.log2(60*60), np.log2(self.period.max())],
                      labels=["1 m", "5 m", "30 m", "1 h", str(round(self.period.max()/(60*60), 1)) + " h"])

    def plot_complete(self, significance_level=0.95, ** kwargs):
        power = (np.abs(self.W12)) ** 2
        glbl_power = power.mean(axis=1)

        if significance_level:
            a1, a2, sig95 = self.significance_level(
                    significance_level=significance_level)

            _, fft_theor = wavelet.significance(
                1.0, self.dt, self.scale, 0, a1*a2, significance_level=significance_level, wavelet=self.mother)
        else:
            sig95 = None
            fft_theor = None

        plot_param = {'title': 'CO2 Analysis', 'signal1': '[CO2]', 's1units': "μmol/m3",
                      'signal2': 'W', 's2units': "m/s",
                      'label': 'CO2 Flux', 'units': 'µmol.m-2.s-1', 'mother_wavelet': self.mother.name}  # g.C.m-2
        plot_param.update(kwargs)
        plot_wavelet(t=self.t, dat=self.x, dat2=self.y, coi=self.coi,
                     period=self.period, plot_param=plot_param, sig95=sig95,
                     power=power, glbl_power=glbl_power, fft_theor=fft_theor)

    def plot_frequencies(*args, pp=5, period=30, labels=None, bypass=False):
        if bypass==False:
            assert sum([isinstance(a_, wv) == False for a_ in args]) == 0, \
                'Not all arguments passed follow class wavelet.'
            assert labels is None or len(labels) == len(args), \
                'Length of args and labels do not match.'
        
        maxfreq = 0
        wv_freq = ()
        covs = ()
        names = ()
        timefr = []
        for i, a_ in enumerate(args):
            pp_len = int(1/a_.dt) * 60 * period
            pp0 = pp * pp_len
            pp1 = pp0 + pp_len

            wv_freqi = [0] + list(np.nanmean((np.apply_along_axis(lambda x: x * np.ravel(a_.transform_wave), 0,
             a_.W12[:, pp0:pp1]) * a_.coi_mask[:, pp0:pp1]).real, axis=1)[:-1])
            wv_freq = wv_freq + (wv_freqi, )
            covs = covs + (np.cov(a_.x[pp0:pp1], a_.y[pp0:pp1])[0, 1], )
            names = names + \
                (a_.name if 'name' in a_.__dict__.keys() else None, )
            timefr = timefr + \
                [str(a_.t[pp0]) + str(a_.t[pp1])]
            maxfreq = max(maxfreq, a_.period.max())

        labels = names if labels is None else labels

        plot_periods = dict([[p*60, str(p)+" m"] if p < 60 else [p*60, str(int(p/60))+" h"]
                            for p in [1, 5, 30, 60, 12*60, 24*60, 48*60]])
        period_ticks = [[np.log2(k) for k in plot_periods if k < maxfreq/2] + [np.log2(maxfreq)]] + \
            [[plot_periods[k] for k in plot_periods.keys() if k < maxfreq/2] +
             [str(round(maxfreq/(60*60), 1)) + " h"]]
                
        pyplot.subplot(121)
        pyplot.axhline(y=0, color='black')
        for i, a_ in enumerate(args):
            pp_len = int(1/a_.dt) * 60 * period
            pp0 = pp * pp_len
            pp1 = pp0 + pp_len
            pyplot.plot(np.log2(a_.period), wv_freq[i] / covs[i], label=labels[i])

        pyplot.xticks(period_ticks[0], period_ticks[1])
        pyplot.ylim(-0.1,0.1)
        pyplot.legend(loc='lower right')

        pyplot.subplot(122)
        pyplot.axvline(np.log2(30*60), ls='--', color='k')
        for i, a_ in enumerate(args):
            pp_len = int(1/a_.dt) * 60 * period
            pp0 = pp * pp_len
            pp1 = pp0 + pp_len
            pyplot.plot(np.log2(a_.period),
                        np.cumsum(wv_freq[i]) / covs[i], label=labels[i])

        pyplot.xticks(period_ticks[0], period_ticks[1])
        pyplot.legend(loc='lower right')
        pyplot.title(timefr)
        return

    def show(self, figure):
        figure
        pyplot.show()


def prepare_wavelet_inside_coi(d0, d1, dt, max_period, filefreq, mother=wavelet.Morlet(6), **kwargs):
    buf = np.ceil((1 / (mother.flambda() * mother.coi()))
                  * max_period / (30*60))
    
    d0 = d0 - datetime.timedelta(minutes=(buf+1)*(30))
    d1 = d1 + datetime.timedelta(minutes=(buf+1)*(30)) - datetime.timedelta(seconds=dt)

    datarange = [pd.date_range(
                start=d0, end=d1, freq=str(filefreq) + " min")]
    '''check to see why I should add 30min'''
    # datarange += datetime.timedelta(minutes=filefreq)
    
    #datarange = datarange.strftime("%Y%m%d-%H%M").to_list() if filefreq < 24*60 \
    #    else list(np.unique(datarange.strftime("%y%m%d")))
    dfraw = get_rawdata.FluxTowerRawData(lookup=datarange, **kwargs)
    
    # Gapfilling to make sure it goes from d0 to d1
    '''
    tname = 'TIMESTAMP' if 'tname' not in kwargs.keys() else kwargs['tname']
    original_data = dfraw.data[tname]
    dfraw.data = pd.merge(
        dfraw.data.set_index(tname),
        pd.DataFrame({tname: pd.date_range(d0, d1, freq=str(dt) + ' S')}), 
        on=tname, how='outer').ffill()
    dfraw.data['qc'] = [0 if e in original_data else 0 for e in dfraw.data[tname]]
    '''
    return dfraw

def wavelet_nearest_freq(wvlst, obs, period):
    wvlst_cs = np.array([np.cumsum(f) for f in wvlst])
    _freqs = [np.where(wvlst_cs[f] == tcom.nearest(wvlst_cs[f], obs[f])[0])[
        0][0] for f in range(len(wvlst))]
    _periods = [period[f] for f in _freqs]
    return _freqs, _periods

# AD-ON

def soundofecosystem(wave=np.empty(0), method="flux", show=0, volume=32760, freqrate=1121, freq_filter=(0, 100)):    
    start_time = time.time()
    file = "./wavelets_for_flux/data/tmp/wavelet.wav"
     # int16: (-32768, +32767)
    
    print("1. Audio. (%.2f seconds)" % (time.time() - start_time), end=" ")        
    # 1/80 : 1121, 1/40 : 561
    #t, mother, period, data = calculate_wavelet(signal, dj=1/((freqrate-1)//14), plot=False)  # t, mother, period, power
    wave = [w for w in wave if sum(np.array(w)**2)>0]
    samplerate = 20*len(wave)
    print(samplerate)
    #data = np.asarray(data, float)
    data = np.array(list(map(list, zip(*wave))))  # reverse lines and columns
    #data = data.astype(np.int16)
        
    # if we want to filter
    #data = data[:120] # first 2 minutes
    #data = data[int(0.4*len(wave[0])):int(0.6*len(wave[0]))] # middle
        
    #samplerate, data = resample_rate(1024, data)
    #period = np.linspace(period.min(), period.max(), num=samplerate)  # necessary if resample
    data = np.array(list(chain.from_iterable(data)))  # flat list
    #data = np.array([item.real() for sublist in data for item in sublist])  # flat list & ignore imaginary part
        
    sampletime = len(data)/samplerate
    # int16: (-32768, +32767)
    print("- %02i:%02d, freq.: %i (%i), range: (%i, %i)" % (sampletime//60, sampletime%60, samplerate, len(data), np.nanmin(data), np.nanmax(data)))
    
    data = [d[0] if isinstance(d, np.ndarray) else d for d in data]  # deal with stereo .wav files
    #data = [d if (d > LOWrange) & (d < HIGHrange) else 0 for d in data]  # filter
    
    #if freq_filter!=(0, 100):
    #    data_ = [data[i:i + samplerate] for i in range(0, len(data), samplerate)]  # to nested list
    #    data_ = [d[int((freq_filter[0]/100)*(samplerate-1)):int((freq_filter[1]/100)*(samplerate-1))] for d in data_]
    #    samplerate = len(data_[0])
    #    data = np.array(list(chain.from_iterable(data_)))  # flat list
    
    data = np.array(data)
    data = data * (volume / np.nanmax(data)) # rescale
    #data = 0.5*data + 0.5*np.append(data[samplerate:], data[:samplerate]) # ghost, just for fun
    print("Writing .wav file. (%.2f seconds)" % (time.time() - start_time))
    sp.io.wavfile.write(file, samplerate, data.astype(np.int16))
    

    print("2. Video. (%.2f seconds)" % (time.time() - start_time))
    # https://stackoverflow.com/questions/61808191/is-there-an-easy-way-to-animate-a-scrolling-vertical-line-in-matplotlib
    # samplerate, data = read("./wavelets_for_flux/data/tmp/noise.wav")  # see exactly the sound
    duration = len(data)/samplerate # in sec
    
    #fig,ax = pyplot.subplots()
    figprops = dict(figsize=(15, 3), dpi=72)
    fig = pyplot.figure(**figprops)
    ax = pyplot.axes()
    vl = ax.axvline(0, ls='-', color='r', lw=1, zorder=10)
    #ax.set_xlim(0,duration)
    tx = np.linspace(0, duration, num=int(duration))
    data_ = [data[i:i + samplerate] for i in range(0, len(data), samplerate)]  # to nested list
    if len(data)%samplerate > 0:
        data_ = data_[:-1]  # drop last second if not full second
    power = np.array(list(map(list, zip(*data_))))  # reverse lines and columns
    #if method!="flux":
    period = np.linspace(0, samplerate, num=samplerate)
        
    pyplot.contourf(tx, period, np.log2(power), #np.log2(levels),
                    extend='both', cmap=pyplot.cm.viridis)
    pyplot.colorbar()
    ax.set_ylabel('Period')
    if 2**period.max()<10000:
        Yticks = 2 ** np.arange(np.ceil(np.log2(period.min())),
                                np.ceil(np.log2(period.max())))
        ax.set_yticks(np.log2(Yticks))
        ax.set_yticklabels(Yticks)
    
    print("4. Show. (%.2f seconds)" % (time.time() - start_time))
    #display(IPython.display.Audio(file, rate=samplerate, embed=True, autoplay=True))
    pyplot.show()
    return


# PLOT functions

def plot_wavelet(t, dat, dat2, coi, plot_param, period,
                 power=None, signif=None, sig95=None, scale_avg=None, scale_avg_signif=None,
                 glbl_signif=None, glbl_power=None, fft_theor=None, fft_power=None, fftfreqs=None):
    Δt = (t[-1]-t[0]) / len(t)
    flag2signals = True if ((dat2 is not None) and (len(dat2)>0)) else False
    covar = np.cov(dat, dat2)[0,1]
    
    #t = np.arange(0, len(t)) * dt + min(t).timestamp()
    # Prepare the figure
    pyplot.close('all')
    # pyplot.ioff()
    figprops = dict(figsize=(15, 8), dpi=72)
    fig = pyplot.figure(**figprops)

    # First sub-plot, the original time series anomaly and inverse wavelet
    # transform.
    ax = pyplot.axes([0.1, 0.75, 0.65, 0.2])
    if flag2signals:
        axb = ax.twinx()
        axb.plot(t, dat2, 'b', linewidth=1, alpha=0.6, label=plot_param['signal2'])
        ax.plot(t, dat, 'k', linewidth=1, label=plot_param['signal1'])
        ax.set_ylabel(r'{} [{}]'.format(plot_param['signal1'], plot_param['s1units']))
        axb.set_ylabel(r'{} [{}]'.format(plot_param['signal2'], plot_param['s2units']))
    else:
        #ax.plot(t, iwave, '-', linewidth=1, color=[0.5, 0.5, 0.5], label='Inverse Wavelet Transform')
        #ax.set_ylabel(r'{} [{}]'.format(plot_param['label'], plot_param['units']))
        ax.plot(t, dat, 'k', linewidth=1, label=plot_param['signal1'])
        ax.set_ylabel(r'{} [{}]'.format(
            plot_param['signal1'], plot_param['s1units']))
    ax.set_title('a) {}'.format(plot_param['title']))
    ax.legend(loc='lower right')
    if flag2signals: axb.legend(loc='upper right')

    # Second sub-plot, the normalized wavelet power spectrum and significance
    # level contour lines and cone of influece hatched area. Note that period
    # scale is logarithmic.
    bx = pyplot.axes([0.1, 0.37, 0.65, 0.28], sharex=ax)
    levels = [2**e for e in range(-6, 16)]
    bx.contourf(t, np.log2(period), np.log2(power),  # np.log2(levels),
                extend='both', cmap=pyplot.cm.viridis)
    extent = [t.min(), t.max(), 0, max(period)]
    if sig95 is not None: bx.contour(t, np.log2(period), sig95, [-99, 1], colors='k', linewidths=1, extent=extent)
    bx.fill(np.concatenate([t, t[-1:] + Δt, t[-1:] + Δt,
                                t[:1] - Δt, t[:1] - Δt]),
            np.concatenate([np.log2(coi), [1e-9], np.log2(period[-1:]),
                                np.log2(period[-1:]), [1e-9]]),
            'k', alpha=0.3, hatch='x')
    bx.set_title('b) {} Wavelet Power Spectrum ({})'.format(plot_param['label'], plot_param['mother_wavelet']))
    bx.set_ylabel('Period')
    #
    #Yticks = 2 ** np.arange(np.ceil(np.log2(period.min())),
    #                        np.ceil(np.log2(period.max())))
    bx.set_ylim(np.log2([period.min(), period.max()]))
    #bx.set_yticks(np.log2(Yticks))
    #bx.set_yticklabels(Yticks)
    Yticks = [[np.log2(60), np.log2(60*5), np.log2(60*30), np.log2(60*60), np.log2(period.max())], 
              ["1 m", "5 m", "30 m", "1 h", str(round(period.max()/(60*60), 1)) + " h"]]
    bx.set_yticks(Yticks[0])
    bx.set_yticklabels(Yticks[1])
    #pyplot.yticks(Yticks[0], Yticks[1])

    # Third sub-plot, the global wavelet and Fourier power spectra and theoretical
    # noise spectra. Note that period scale is logarithmic.
    cx = pyplot.axes([0.77, 0.37, 0.2, 0.28], sharey=bx)
    if flag2signals==False: 
        cx.plot(glbl_signif, np.log2(period), 'k--')
    #cx.plot(covar * fft_theor, np.log2(period), '--', color='#cccccc')
    #if flag2signals==False: 
    #    cx.plot(var * fft_power, np.log2(1. / fftfreqs), '-', color='#cccccc', linewidth=1.)
    #cx.plot(covar * glbl_power, np.log2(period), 'k-', linewidth=1.5)
    cx.plot(np.log2(power.mean(1)), np.log2(period), 'k-', linewidth=1.5)
    cx.set_title('c) Global Wavelet Spectrum')
    cx.set_xlabel(r'Power [({})^2]'.format(plot_param['units']))
    #cx.set_xlim([0, glbl_power.max() * var])
    cx.set_ylim(np.log2([period.min(), period.max()]))
    cx.set_yticks(Yticks[0])
    cx.set_yticklabels(Yticks[1])
    pyplot.setp(cx.get_yticklabels(), visible=False)

    if flag2signals==False:
        # Fourth sub-plot, the scale averaged wavelet spectrum.
        dx = pyplot.axes([0.1, 0.07, 0.65, 0.2], sharex=ax)
        dx.axhline(scale_avg_signif, color='k', linestyle='--', linewidth=1.)
        dx.plot(t, scale_avg, 'k-', linewidth=1.5)
        dx.set_title('d) {}--{} year scale-averaged power'.format(2, 8))
        dx.set_xlabel('Time (year)')
        dx.set_ylabel(r'Average variance [{}]'.format(plot_param['units']))
        dx.set_xlim([t.min(), t.max()])

        # ex = pyplot.axes([0.77, 0.07, 0.2, 0.2], sharey=dx)
        # ex.plot(df_site["t"], df_site["NEE_VUT_REF"], 'k--')
        # ex.set_title('e) Interactive')
        # pyplot.setp(ex.get_yticklabels(), visible=False)

    pyplot.show()


def plot_flux(flux, highlight=None, figprops=dict(figsize=(4*16/3, 2*3), dpi=72),
              ylabel="NEE", yunit=r'$ μmol.CO_2.m^{-2}.s^{-1}$', **kwargs):
    condition = (flux.ec == np.nan) if highlight is None else highlight

    if highlight is None: fig, (a0, a2) = pyplot.subplots(1, 2, gridspec_kw={'width_ratios': [4, 1]}, figsize=(4*16/3, 2*2), dpi=72)
    else: fig, (a0, a1, a2) = pyplot.subplots(1, 3, gridspec_kw={'width_ratios': [3, 1, 1]}, figsize=(4*16/3, 2*2), dpi=72)
    
    for (k,v) in kwargs.items():
        if (k.startswith('y') & (len(k)<3)): a0.plot(flux.TIMESTAMP, v, color='C0', label=k)
    #a0.plot(flux.TIMESTAMP, flux.ec, color='red', label="eddy cov")
    #a0.fill_between(flux.TIMESTAMP, y1=max(max(flux.ec), max(flux.wv)), y2=min(min(flux.ec), min(flux.wv)), where=condition, alpha=0.3)
    #a0.scatter(flux.TIMESTAMP[np.where((flux.qc_ec >0))], flux.ec[np.where((flux.qc_ec >0))], color='red', marker="x")
    a0.plot(flux.TIMESTAMP, flux.wv, color='C0', label="wavelet")
    a0.scatter(flux.TIMESTAMP[np.where((flux.qc_wv >0))], flux.wv[np.where((flux.qc_wv >0))], color='C0', marker="x")
    #a0.scatter(flux.TIMESTAMP[sp.signal.find_peaks(flux.wv, height=0)[0]], flux.wv[sp.signal.find_peaks(flux.wv, height=0)[0]], color='C0', marker="^")
    a0.legend(loc='lower right')
    a0.set_ylabel(ylabel + ' (' + yunit + ')')
    
    if highlight is not None: 
        a1.set_title("highlight")
        #a1.scatter(flux.ec[np.where((flux.qc_ec==0) & (flux.qc_wv==0) & condition)], flux.wv[np.where((flux.qc_ec==0) & (flux.qc_wv==0) & condition)], color='grey')
        #a1.scatter(flux.ec[np.where((flux.qc_ec >0) & condition)], flux.wv[np.where((flux.qc_ec >0) & condition)], color='red', marker="x")
        #a1.scatter(flux.ec[np.where((flux.qc_wv >0) & condition)], flux.wv[np.where((flux.qc_wv >0) & condition)], color='C0', marker="x")
        #a1.plot(flux.ec[np.where((flux.qc_ec==0) & condition)], flux.ec[np.where((flux.qc_ec==0) & condition)], '--')
        a1.axline([0, 0], [1, 1], linestyle='--')
        a1.set_xlabel("eddy cov")
        a1.set_ylabel("wavelet")

    if highlight is not None: a2.set_title("others")
    #a2.scatter(flux.ec[np.where((flux.qc_ec==0) & (flux.qc_wv==0) & (condition==False))], flux.wv[np.where((flux.qc_ec==0) & (flux.qc_wv==0) & (condition==False))], color='grey')
    #a2.scatter(flux.ec[np.where((flux.qc_ec >0) & (condition==False))], flux.wv[np.where((flux.qc_ec >0) & (condition==False))], color='red', marker="x")
    #a2.scatter(flux.ec[np.where((flux.qc_wv >0) & (condition==False))], flux.wv[np.where((flux.qc_wv >0) & (condition==False))], color='C0', marker="x")
    #a1.plot(flux.ec[np.where((flux.qc_ec==0) & condition)], flux.ec[np.where((flux.qc_ec==0) & condition)], '--')
    a2.axline([0, 0], [1, 1], linestyle='--')
    a2.set_xlabel("eddy cov")
    a2.set_ylabel("wavelet")

    pyplot.show()
    return


def plot_waveletflux(flux, figprops=dict(figsize=(4*16/3, 2*3), dpi=72)):
    #parts = [i for i, x in enumerate(period) if x==tcom.nearest(period, period_chosen)[0]][0]
    
    #pflux = np.array([sum(f[:parts]) for f in flux_lst])
    
    #valid = np.array([1 if next((i for i, x in enumerate(reversed(flux_lst[ix][:int(len(flux_lst[ix])*parts[0])])) if x), None)==0 else 0 for ix in range(len(flux_lst))])
    #valid = np.array([1 if next((i for i, x in enumerate(reversed(flux_lst[ix][:parts])) if x), None)==0 else 0 for ix in range(len(flux_lst))])
    #valid = np.where(valid == 1)
    #pflux = np.array([parts[1] * sum(f[:int(len(f)*parts[0])]) for f in flux_lst])

    #
    def compute_distance(x, ylst, qc, k=1):
        k = int(k+1)
        valid = list(set(np.where((qc == 0))[0]))
        if len(valid)==0: return(np.nan)
        distance = abs(((np.array([sum(f[:k]) for f in np.array(ylst)[valid]]) - np.array(x)[valid])**2).mean())
        return(distance)
        
    dt = flux.meta['dt']
    TIMESTAMP = flux.TIMESTAMP
    wvlst = flux.meta['wvlst']
    period = flux.meta['period']
    pfreq_ = flux.meta['wv_period']
    pfreq = list(period).index(pfreq_)
    qc_shared = np.array(max(list(flux.qc_ec), list(flux.qc_wv)))
    
    #t_opt_range = [min(flux.TIMESTAMP), min(max(flux.TIMESTAMP), max(t))]
    #coi_min = period[pfreq] #min(np.nanmin(coi_partial), period[int(len(period)*parts)])
    #pcoi = [coi_min if ((flux.TIMESTAMP[ix] > t_opt_range[0]) and (flux.TIMESTAMP[ix] < t_opt_range[1])) else 1e-9 for ix in range(len(flux.TIMESTAMP))]
    
    fig = pyplot.figure(**figprops)
    ql_dic = {0: 'o', 1: 'x', 2: 'x', "NA": '_'}
    plot_periods = dict([[p*60, str(p)+" m"] if p<60 else [p*60, str(int(p/60))+" h"] for p in [1, 5, 30, 60, 12*60, 24*60, 48*60]])
    period_ticks = [[np.log2(k) for k in plot_periods if k < period.max()/2] + [np.log2(period.max())]] + \
    [[plot_periods[k] for k in plot_periods.keys() if k < period.max()/2] + [str(round(period.max()/(60*60), 1)) + " h"]]

    # wavelets considered for flux (wavelet inside coi)
    pyplot.subplot(241)
    #ax = pyplot.axes([0.6, 0.5, 0.8, 0.2])
    pyplot.contourf(flux.TIMESTAMP, np.log2(period), np.log2(list(map(list, zip(*wvlst)))), extend='both', cmap=pyplot.cm.viridis)
    pyplot.fill(np.concatenate([flux.TIMESTAMP, 
                                [el + np.timedelta64(30, 'm') for el in flux.TIMESTAMP[-1:]],
                                [el + np.timedelta64(30, 'm') for el in flux.TIMESTAMP[-1:]], 
                                [el - np.timedelta64(30, 'm') for el in flux.TIMESTAMP[:1]],
                                [el - np.timedelta64(30, 'm') for el in flux.TIMESTAMP[:1]]]),
                np.concatenate([np.log2([period[pfreq] for ix in range(len(flux.TIMESTAMP))]), 
                                [1e-9], np.log2(period[-1:]), np.log2(period[-1:]), [1e-9]]),
                'r', alpha=0.3)
    pyplot.ylim(np.log2([period.min(), period.max()]))
    pyplot.yticks(period_ticks[0], period_ticks[1])
    pyplot.title("Region taken into account")
    pyplot.gca().get_xaxis().set_visible(False)
    
    pyplot.subplot(242)
    #ax = pyplot.axes([0.6, 0.5, 0.8, 0.2])
    pyplot.contourf(flux.TIMESTAMP, np.log2(period), np.log2(list(map(list, zip(*wvlst)))), extend='both', cmap=pyplot.cm.viridis)
    pyplot.fill(np.concatenate([flux.TIMESTAMP, 
                                [el + np.timedelta64(30, 'm') for el in flux.TIMESTAMP[-1:]],
                                [el + np.timedelta64(30, 'm') for el in flux.TIMESTAMP[-1:]], 
                                [el - np.timedelta64(30, 'm') for el in flux.TIMESTAMP[:1]],
                                [el - np.timedelta64(30, 'm') for el in flux.TIMESTAMP[:1]]]),
                np.concatenate([np.log2(flux.ecfreqs), [1e-9], np.log2(period[-1:]), np.log2(period[-1:]), [1e-9]]),
                'white', alpha=1)
    pyplot.scatter(flux.TIMESTAMP[flux.qc_ec>0], np.log2(np.array(flux.ecfreqs)[flux.qc_ec>0]), color='red', marker="x")
    pyplot.ylim(np.log2([period.min(), period.max()]))
    pyplot.yticks(period_ticks[0], period_ticks[1])
    
    pyplot.subplot(243)
    #pyplot.bar(flux.TIMESTAMP, flux.ec-flux.wv, 0.02, label='Δ EC-WV')
    pyplot.legend(loc='lower right')
    pyplot.gca().get_xaxis().set_visible(False)
    
    pyplot.subplot(244)
    #pyplot.plot(np.log2(period), np.log2([compute_distance(flux.ec, wvlst, qc_shared, i) for i in np.linspace(1, len(period), len(period))-1]))
    #pyplot.scatter(np.log2(period[pfreq]), np.log2(compute_distance(flux.ec, wvlst, qc_shared, pfreq)), c='red')
    pyplot.ylim(bottom=0)
    pyplot.ylabel("RMSE (log2)")
    pyplot.xticks(period_ticks[0], period_ticks[1])
    
    # time plots with eddy covariance and reconstructed flux
    pyplot.subplot(245)
    #pyplot.plot(flux.TIMESTAMP, flux.ec, color='red', label="eddy cov")
    #pyplot.scatter(flux.TIMESTAMP[flux.qc_ec>0], flux.ec[flux.qc_ec>0], color='red', marker="x")
    pyplot.plot(flux.TIMESTAMP, flux.wv, '-C0', label="wavelet ("+str(round(pfreq_//(60)))+"min)")
    pyplot.scatter(flux.TIMESTAMP[flux.qc_wv>0], flux.wv[flux.qc_wv>0], color='C0', marker="x")
    pyplot.legend(loc='lower right')
    
    pyplot.subplot(246)  
    nflux_freqs = [[i for i, x in enumerate(period) if x == tcom.nearest(
        period, f*dt*60)[0]][0] for f in [5, 10, 30, 60]]
    nflux_freqs = [np.array([sum(f[:nflux_freqs[0]]) for f in wvlst]), 
                   np.array([sum(f[nflux_freqs[0]:nflux_freqs[1]]) for f in wvlst]), 
                   np.array([sum(f[nflux_freqs[1]:nflux_freqs[2]]) for f in wvlst]),
                   np.array([sum(f[nflux_freqs[2]:nflux_freqs[3]]) for f in wvlst]),
                   np.array([sum(f[nflux_freqs[3]:]) for f in wvlst])]
    pyplot.bar(flux.TIMESTAMP, nflux_freqs[0], 0.02, label='5 min')
    pyplot.bar(flux.TIMESTAMP, nflux_freqs[1], 0.02, bottom=nflux_freqs[0], label='10 min')
    pyplot.bar(flux.TIMESTAMP, nflux_freqs[2], 0.02, bottom=nflux_freqs[0]+nflux_freqs[1], label='30 min')
    pyplot.bar(flux.TIMESTAMP, nflux_freqs[3], 0.02, bottom=nflux_freqs[0]+nflux_freqs[1]+nflux_freqs[2], label='1 h')
    #pyplot.bar(flux.TIMESTAMP, nflux_freqs[4], 0.02, bottom=nflux_freqs[0]+nflux_freqs[1]+nflux_freqs[2]+nflux_freqs[3], label='> 1hr')
    pyplot.legend(loc='lower right')  

    pyplot.subplot(247)  
    nflux_freqs = [[i for i, x in enumerate(period) if x == tcom.nearest(
        period, f*dt*60)[0]][0] for f in [30, 60]]
    nflux_freqs = [np.array([sum(f[:nflux_freqs[0]]) for f in wvlst]), 
                   np.array([sum(f[nflux_freqs[0]:nflux_freqs[1]]) for f in wvlst]), 
                   np.array([sum(f[nflux_freqs[1]:]) for f in wvlst])]
    pyplot.bar(flux.TIMESTAMP, nflux_freqs[1], 0.02, label='30 min - 1 h')
    pyplot.bar(flux.TIMESTAMP, nflux_freqs[2], 0.02, bottom=nflux_freqs[1], label='> 1hr')
    pyplot.legend(loc='lower right')  
    #pyplot.plot(flux.TIMESTAMP, np.log2(flux.ecfreqs), color='C0', label="coi")
    #pyplot.scatter(flux.TIMESTAMP[flux.qc_ec==0], np.log2(np.array(flux.ecfreqs)[flux.qc_ec==0]), color='C0')
    #pyplot.scatter(flux.TIMESTAMP[flux.qc_ec>0], np.log2(np.array(flux.ecfreqs)[flux.qc_ec>0]), color='red', marker="x")
    #pyplot.yticks(period_ticks[0], period_ticks[1])
    
    # eddy cvovariance against reconstructed flux
    pyplot.subplot(248)
    '''for g in np.unique(qc_shared):
        if np.isnan(g): g = "NA"
        pyplot.scatter(flux.ec[qc_shared==g], 
                       flux.wv[qc_shared==g], 
                       marker=ql_dic[g], c='gray')
    pyplot.plot(flux.ec[flux.qc_ec==0], flux.ec[flux.qc_ec==0], 'k')
    pyplot.plot(flux.ec, flux.ec, '--k')'''
    pyplot.xlabel("eddy covariance")
    pyplot.ylabel("wavelet")
    
    pyplot.show()
    return
