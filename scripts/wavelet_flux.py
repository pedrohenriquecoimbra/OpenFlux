import pywt
import sys
import pycwt

"""
Check: https://github.com/regeirk/pycwt/blob/master/pycwt/sample/simple_sample.py

In this example we will load the NINO3 sea surface temperature anomaly dataset
between 1871 and 1996. This and other sample data files are kindly provided by
C. Torrence and G. Compo at
<http://paos.colorado.edu/research/wavelets/software.html>.
"""
import multiprocess as mp
import itertools
from distutils.spawn import spawn
import warnings
from genericpath import isfile
import os
import re
import math
from random import random, randint
import numpy as np
import scipy as sp
import scipy.io.wavfile
import pandas as pd
import pycwt as wavelet
import matplotlib
from matplotlib import pyplot
from Lib.OpenFlux.scripts import get_data, get_rawdata, common as tt
from Lib.OpenFlux.scripts.wavelets_TorrenceCompo.wave_python import waveletFunctions as wv_tc
#import scripts.get_data as get_data
#import scripts.get_rawdata as get_rawdata
#import scripts.common as tt
#from scripts.tiltcorrection import tiltcorrection
import datetime
import copy
import pickle
import time
import gc

from itertools import chain  # for sound only
import IPython  # for sound only

import fcwt

import importlib
importlib.reload(tt)
importlib.reload(get_data)


def calculate_coi(N, dt=0.05, param=6, mother="MORLET"):
    # from Torrence and Compo

    # fourier factor (c)
    # _, _, c, _ = wv_tc.wave_bases(
    #    mother=mother, k=np.array([-1, -1]), scale=-1, param=param)
    # coi = c * dt * np.concatenate((
    #        np.insert(np.arange(int((N + 1) / 2) - 1), [0], [1E-5]),
    #        np.insert(np.flipud(np.arange(0, int(N / 2) - 1)), [-1], [1E-5])))

    # from pycwt
    # fourier factor (c)
    c = wavelet.flambda() * wavelet.coi()
    coi = (N / 2 - np.abs(np.arange(0, N) - (N - 1) / 2))
    coi = c * dt * coi
    return coi

def coitomask(coi, shape, scales, false=False):
    #  freq |_____ 
    #         time
    coi_mask = np.ones(shape, dtype = bool)
    tx = shape[0] if shape[0] == len(coi) else shape[1]
    fx = shape[1] if shape[0] == len(coi) else shape[0]
    
    for i in range(tx):
        print("i", i, tx)
        coi_period_threshold = list(scales).index(
            tt.nearest(scales, coi[i])[0])

        for j in range(coi_period_threshold, fx):
            print("i", i, "j", j, fx, "coi_period_threshold", coi_period_threshold)
            coi_mask[i][j] = false
    return coi_mask


def bufferforfrequency_dwt(N=0, n_=30*60*20, fs=20, level=None, f0=None, max_iteration=10**4, wave='db6', **kw):
    if level is None and f0 is None: level = 18
    lvl = level if level is not None else int(np.ceil(np.log2(fs/f0)))
    n0 = N
    cur_iteration = 0
    while True:
        n0 += pd.to_timedelta(n_)/pd.to_timedelta("1S") * fs if isinstance(n_, str) else n_
        if lvl <= pywt.dwt_max_level(n0, wave):
            break
        cur_iteration += 1
        if cur_iteration > max_iteration:
            warnings.warn('Limit of iterations attained before buffer found. Current buffer allows up to {} levels.'.format(
                pywt.dwt_max_level(n0, wave)))
            break
    return (n0-N) * fs**-1


def bufferforfrequency(f0, dt=0.05, param=6, mother="MORLET", wavelet=pycwt.Morlet(6)):
    #check if f0 in right units
    # f0 ↴
    #    /\
    #   /  \
    #  /____\
    # 2 x buffer
    
    # _, _, c, _ = wv_tc.wave_bases(mother=mother, k=np.array([-1, -1]), scale=-1, param=param)
    c = wavelet.flambda() * wavelet.coi()
    n0 = 1 + (2 * (1/f0) * (c * dt)**-1)
    N = int(np.ceil(n0 * dt))

    # xmax = f0 / (c * dt)
    # N = ((xmax + 2) * 2) - 1
    # N int(np.ceil(N * dt)) #(1 / (λ * c)) * f0
    return N

def djfromscales(freqs, fn, scaling='log'):
    # dj to be confirmed
    if scaling == "lin":
        dj = np.mean([(1/freqs[i])-(1/freqs[i-1]) for i in range(1, fn)])
    if scaling == "log":
        dj = np.mean([(1/freqs[i])/(1/freqs[i-1]) for i in range(1, fn)])
    else:
        dj = np.mean([(1/freqs[i])/(1/freqs[i-1]) for i in range(1, fn)])
    dj = math.log(2, 2) / math.log(2, dj)
    return dj

def cwt(input, fs, f0, f1, fn, nthreads=1, scaling="log", fast=False, norm=True, Morlet=6.0):

    #check if input is array and not matrix
    if input.ndim > 1:
        raise ValueError("Input must be a vector")

    #check if input is single precision and change to single precision if not
    if input.dtype != 'single':
        input = input.astype('single')

    morl = fcwt.Morlet(Morlet) #use Morlet wavelet with a wavelet-parameter

    #Generate scales

    if scaling == "lin":
        scales = fcwt.Scales(morl,fcwt.FCWT_LINFREQS,fs,f0,f1,fn)
    elif scaling == "log":
        scales = fcwt.Scales(morl,fcwt.FCWT_LOGSCALES,fs,f0,f1,fn)
    else:
        scales = fcwt.Scales(morl,fcwt.FCWT_LOGSCALES,fs,f0,f1,fn)

    _fcwt = fcwt.FCWT(morl, int(nthreads), fast, norm)

    output = np.zeros((fn,input.size), dtype='csingle')
    freqs = np.zeros((fn), dtype='single')
    
    _fcwt.cwt(input,scales,output)
    scales.getFrequencies(freqs)

    return freqs, output


def icwt(W, sj, dt, dj, Cd=None, psi=None, wavelet=pycwt.wavelet.Morlet(6)):
    if Cd is None: Cd = wavelet.cdelta
    if psi is None: psi = wavelet.psi(0)
        
    a, b = W.shape
    c = sj.size
    if a == c:
        sj_ = (np.ones([b, 1]) * sj).transpose()
    elif b == c:
        sj_ = np.ones([a, 1]) * sj
    
    x = (W.real / (sj_ ** .5)) * ((dj * dt ** .5) / (Cd * psi))
    return x

   
def dwt(*args, level=None, wave="db6"):
    Ws = []
    for X in args:
        Ws += [pywt.wavedec(X, wave, level=level)]
    level = len(Ws[-1])-1
    return Ws


def idwt(*args, N, level=None, wave="db6"):
    #assert sum([s==level for s in W.shape]), "Coefficients don't have the same size as level."
    def wrcoef(N, coef_type, coeffs, wavename, level):
        a, ds = coeffs[0], list(reversed(coeffs[1:]))

        if coef_type == 'a':
            return pywt.upcoef('a', a, wavename, level=level, take=N)  # [:N]
        elif coef_type == 'd':
            return pywt.upcoef('d', ds[level-1], wavename, level=level, take=N)  # [:N]
        else:
            raise ValueError("Invalid coefficient type: {}".format(coef_type))
    
    Ys = []
    for W in args:
        A1 = wrcoef(N, 'a', W, wave, level)
        D1 = [wrcoef(N, 'd', W, wave, i) for i in range(1, level+1)]
        Ys += [np.array(D1 + [A1])]
    return Ys, level

def loaddatawithbuffer(d0, d1=None, freq=None, buffer=None, 
                       tname="TIMESTAMP", **kwargs):
    if isinstance(d0, (pd.DatetimeIndex)):
        d0, d1 = [np.nanmin(d0), np.nanmax(d0)]
    
    if buffer == None:
        datarange = [pd.date_range(start=d0, end=d1, freq=freq)[:-1] + pd.Timedelta(freq)]
    else:
        freqno = int(re.match("\d*", "30min")[0])
        
        bufi = np.ceil(buffer / (freqno*60)) * freqno
        datarange = [
            pd.date_range(
                start=pd.to_datetime(d0) - pd.Timedelta(bufi, unit='min'),
                end=pd.to_datetime(d1) + pd.Timedelta(bufi, unit='min'),
                freq=freq)[:-1] + pd.Timedelta(freq)]
                
    if not datarange:
        return pd.DataFrame()
    
    data = get_rawdata.FluxTowerRawData(lookup=datarange, **kwargs)
    if data == None or data.data.empty:
        return data.data
    data.data[tname] = pd.to_datetime(data.data[tname])
    
    if buffer:
        d0 = pd.to_datetime(d0) - pd.Timedelta(buffer*1.1, unit='s')
        d1 = pd.to_datetime(d1) + pd.Timedelta(buffer*1.1, unit='s')
        data.filter({tname: (d0, d1)})

    # garantee all data points, if any valid time, else empty dataframe
    if np.sum(np.isnat(data.data.TIMESTAMP)==False):
        data.data = pd.merge(pd.DataFrame({tname: pd.date_range(*tt.nanminmax(data.data.TIMESTAMP), freq="0.05S")}),
                            data.data,
                            on=tname, how='outer').reset_index(drop=True)
        return data.data
    else:
        pd.DataFrame()


def matrixtotimetable(time, mat, c0name="TIMESTAMP", **kwargs):
    assert len(time) in mat.shape, f"Time ({time.shape}) and matrix ({mat.shape}) do not match."
    mat = np.array(mat)

    if len(time) != mat.shape[0] and len(time) == mat.shape[1]:
        mat = mat.T

    __temp__ = pd.DataFrame(mat, **kwargs)
    __temp__.insert(0, c0name, time)

    return __temp__


def conditional_sampling(Y12, *args, level=None, wave="db6", false=0):
    nargs = len(args) + 1
    YS = [Y12] + list(args)
    #YS, _ = decompose(*args, level=level, wave=wave)
    #Yi = {}
    Ys = {}
    label = {1: "+", -1: "-"}

    for co in set(itertools.combinations([1, -1]*nargs, nargs)):
        name = 'xy{}a'.format(''.join([label[c] for c in co]))
        Ys[name] = Y12
        for i, c in enumerate(co):
            xy = 1 * (c*YS[i] >= 0)
            #xy[xy==0] = false
            xy = np.where(xy == 0, false, xy)
            Ys[name] = Ys[name] * xy
    return Ys


def universal_wt(signal, method, fs=20, f0=1/(3*60*60), f1=10, fn=100, 
                 dj=1/12, inv=True, **kwargs):
    assert method in [
        'dwt', 'cwt', 'fcwt'], "Method not found. Available methods are: dwt, cwt, fcwt"
    if method== "dwt":
        lvl = kwargs.get('level', int(np.ceil(np.log2(fs/f0))))
        # _l if s0*2^j; fs*2**(-_l) if Hz; (1/fs)*2**_l if sec.
        sj = [_l for _l in np.arange(1, lvl+2, 1)]
        waves = dwt(signal, level=lvl, **kwargs)
        if inv:
            N = np.array(signal).shape[-1]
            waves = idwt(*waves, N=N, level=lvl, **kwargs)
        wave = waves[0][0]
    elif method == 'fcwt':
        _l, wave = cwt(signal, fs, f0, f1, fn, **kwargs)
        sj = np.log2(fs/_l)
        if inv:
            wave = icwt(wave, sj=sj, dt=fs, dj=dj, **kwargs, 
                        mother=pycwt.wavelet.Morlet(6))
    elif method == 'cwt':
        wave, sj, _, _, _, _ = pycwt.cwt(
            signal, dt=1/fs, s0=2/fs, dj=dj, J=fn-1)
        sj = np.log2(sj*fs)
        if inv:
            wave = icwt(wave, sj=sj, dt=fs**-1, dj=dj, **kwargs)
    return wave, sj

class waveletflux(tt.datahandler):
    def __init__(self, **kwargs):
        "fluxes"
        self.id_vars = ['TIMESTAMP']
        self.t_name = 'TIMESTAMP'
        self.dt = 0.05
        self.Wxy = {}
        self.FWxy = {}
        self.FWxy_std = {}
        self.FWxy_spectra = {}
        self.Fxy = {}
        self.coi_mask = np.array([])

        self.col_vars = ['Fxy']  # , 'qc_wv', 'wv_std']
        self.TIMESTAMP = None
        self.avg_period = None
        #self.wv_std = None
        self.meta = {}

        self.data = pd.DataFrame()
        self.ignore = ['Wxy', 'FWxy_std', 'FWxy_spectra']
        self.__dict__.update(**kwargs)
    
    def select_max_period(self, max_period):
        self.J = int(np.round(np.log2(max_period * (1/self.dt) / 2) /
                self.dj)) if max_period else None
        return self
    
    def copy(self):
        aself = waveletflux(**{k: copy.deepcopy(v) for k, v in self.__dict__.items()})
        return aself
    
    def load_data(self, guaranteeCOI=True, **kwargs):
        if guaranteeCOI:
            return self.prepare_data(**kwargs)
        
    
    def prepare_data(self, d0, d1=None, t_out=None, filefreq=30, max_period=None, **kwargs):
        self.meta.update(**{k: v for k, v in locals().items()
                         if k in ['d0', 'd1', 'dt', 'period', 'dj']})
        self.meta.update(**{k: v for k, v in kwargs.items()
                         if k in ['d0', 'd1', 'dt', 'period', 'dj']})
        self.meta.update(**{'max_period': max_period})

        self.__dict__.update({k: v for k, v in {'filefreq': filefreq}.items()})
                
        try:
            if 'mother' in kwargs.keys():
                kwargs['mother'] = kwargs['mother']
            #print('d0d1', d0, d1)
            if isinstance(d0, (list, np.ndarray)):
                d0, d1 = tt.minmax(d0)
                         
            if max_period:
                d0 = pd.to_datetime(d0)
                d1 = pd.to_datetime(d1)
                #Δ = d1-d0

                # seconds to have a peak of max_period
                buf = (1 / (self.wavelet.flambda() * self.wavelet.coi())) * max_period
                d0_ = d0 - pd.Timedelta(buf*1.1, unit='s')
                d1_ = d1 + pd.Timedelta(buf*1.1, unit='s')

                # files to get
                bufi = np.ceil(buf / (filefreq*60)) * filefreq
                d0 = d0 - pd.Timedelta(bufi, unit='min')#).strftime('%Y%m%d%H%M')
                d1 = d1 + pd.Timedelta(bufi, unit='min')#).strftime('%Y%m%d%H%M')
                
                self.J = int(np.round(np.log2(max_period * (1/self.dt) / 2) /
                self.dj)) if max_period else None

                #if t_out == None:
                #    t_out = (min(t_) - datetime.timedelta(seconds=buf),
                #             max(t_) + datetime.timedelta(seconds=buf-self.dt))

            datarange = [pd.date_range(
                    start=d0, end=d1, freq=str(filefreq) + " min")[:-1] + pd.Timedelta(filefreq, unit='min')]
            #print('datarange', d0, d1, bufi, pd.Timedelta(bufi, unit='min'), datarange)
            if max_period:
                self.data = get_rawdata.FluxTowerRawData(lookup=datarange, **kwargs)
                if self.data.data.empty:
                    self.data = self.data.data
                    warnings.warn("Data given is empty.")
                    return self
                #print('max_period', max_period, self.data.data.columns)
                self.data.data[self.t_name] = pd.to_datetime(
                    self.data.data[self.t_name])
                self.data = self.data.filter({self.t_name: (d0_, d1_)}).data
                
            else:
                self.data = get_rawdata.FluxTowerRawData(
                    lookup=datarange, **kwargs).data
                #print('max_period', max_period, self.data.columns)
            
            if self.data.empty:
                warnings.warn("Data given is empty.")
                return self
            
            self.data = self.data.reset_index(drop=True)
            self.data[self.t_name] = pd.to_datetime(self.data[self.t_name])
            self.data = self.data.sort_values(self.t_name) 
            self.TIMESTAMP = self.data[self.t_name]

            #if t_out is not None:
            #    self.data = self.data[self.data[self.t_name]
            #                        > d0 and (self.data[self.t_name] < d1)]

            self.data = self.data.sort_values(by=[self.t_name])

        except AttributeError as e:
            warnings.warn(str(e) + ' ()')
            return self
        return self
    
    def fcwt(self, x, y=None, data=None, lowmemory=False, tol=36, **kw):
        data = data if data is not None else self.data
        data = pd.read_csv(data) if isinstance(data, str) else data
        
        if isinstance(x, str) and isinstance(y, str):
            x = [f'{x}*{y}']

        if data is None or data.empty:
            if y is not None:
                data = {'x': x, 'y': y}
                del x, y
                x = ['x*y']
            else:
                data = {'x': x}
                del x, y
                x = ['x*x']

        W = {}

        xy = [v.split('*')[:2] for v in x]

        for i, xy_ in enumerate(xy):
            if len(xy_) == 2:
                x_, y_ = xy_
            elif len(xy_) == 1:
                x_ = y_ = xy_[0]
            n_ = str(x_)+'*'+str(y_)

            for v in xy_:
                if v not in W.keys():
                    signal = data[v]
                    signan = np.isnan(signal)
                    if np.sum(signan) <= tol if tol > 1 else tol * len(signal):
                        signal = np.array(signal[signan == False]) if np.sum(
                            signan) else np.array(signal)
                        W[v], sj, freq, coi, _, _ = pycwt.cwt(
                            signal, **kwargs)
                        W[v] = tt.insert_in_array(
                            W[v], [i for i, s in enumerate(signan) if s])
                    else:
                        warnings.warn('During wavelet calculation, {} NANs were found in variable {}. Exceeding threshold set as {}.'.format(
                            np.sum(signan), v, tol))
                        continue
                    #print(">", v, np.nanmean(W[v]), np.mean(W[v]), np.sum(np.isnan(data[v])), len(np.array(data[v])), np.mean(data[v]), np.nanmean(data[v]), kwargs)

            if x_ not in W.keys() or y_ not in W.keys():
                continue

            # cross calculate
            Wxy = W[x_] * W[y_].conjugate()
            #print(x_, np.mean(W[x_]), np.mean(W[x_].conjugate()))
            #print(y_, np.mean(W[y_]), np.mean(W[y_].conjugate()))
            #print('Wxy', np.mean(Wxy))
            if 'Wxy' not in self.ignore:
                self.Wxy[n_] = Wxy

            if 'transform_wave' not in self.__dict__.keys():
                # wavelet coeffifient to flux
                self.transform_wave = (1 / sj[:, None]) * kwargs['dt'] * kwargs['dj'] / \
                    kwargs['wavelet'].cdelta
            if 'period' not in self.__dict__.keys():
                self.period = 1/freq
            if 'scale' not in self.__dict__.keys():
                self.scale = sj
            if 'coi' not in self.__dict__.keys():
                self.coi = coi

            self.FWxy[n_] = (np.apply_along_axis(lambda x: x * np.ravel(self.transform_wave), 0,
                                                 Wxy)).T.real
            #print('self.transform_wave', np.mean(self.transform_wave))
            #print('Wxy', np.mean(Wxy))

            if 'FWxy_spectra' not in self.ignore:
                self.FWxy_spectra[n_] = np.nanmean(self.FWxy[n_], axis=0)

            if 'FWxy_std' not in self.ignore:
                self.FWxy_std[n_] = list(
                    np.zeros(np.array(self.FWxy_std).shape))

            # clear
            for v in xy_:
                if v in W.keys() and v not in tt.flist(xy[i:]) or lowmemory == True:
                    del W[v]
        return self
    
    def get_flux(self, x, y=None, data=None, lowmemory=False, verbosity=0, tol=36, **kw):
        t0 = time.time()
        data = data if data is not None else self.data
        data = pd.read_csv(data) if isinstance(data, str) else data
        
        #if data is not None and data.empty:
        #    warnings.warn("Data given is empty.")
        #    return None
            
        kwargs = {k: v for k, v in self.__dict__.items(
        ) if k in ['dt', 'dj', 's0', 'J', 'wavelet', 'freqs']}
        kwargs.update({k: v for k, v in kw.items(
        ) if k in ['dt', 'dj', 's0', 'J', 'wavelet', 'freqs']})

        if isinstance(x, str) and isinstance(y, str):
            x = [f'{x}*{y}']

        if data is None or data.empty:
            if y is not None:
                data = {'x': x, 'y': y}
                del x, y
                x = ['x*y']
            else:
                data = {'x': x}
                del x, y
                x = ['x*x']

        W = {}

        xy = [v.split('*')[:2] for v in x]

        for i, xy_ in enumerate(xy):
            if len(xy_) == 2:
                x_, y_ = xy_
            elif len(xy_) == 1:
                x_ = y_ = xy_[0]
            n_ = str(x_)+'*'+str(y_)
            
            for v in xy_:
                if v not in W.keys():
                    signal = data[v]
                    signan = np.isnan(signal)
                    if np.sum(signan) <= tol if tol > 1 else tol * len(signal):
                        signal = np.array(signal[signan==False]) if np.sum(signan) else np.array(signal)
                        W[v], sj, freq, coi, _, _ = pycwt.cwt(
                            signal, **kwargs)
                        W[v] = tt.insert_in_array(W[v], [i for i, s in enumerate(signan) if s])
                    else:
                        warnings.warn('During wavelet calculation, {} NANs were found in variable {}. Exceeding threshold set as {}.'.format(np.sum(signan), v, tol))
                        continue
                    #print(">", v, np.nanmean(W[v]), np.mean(W[v]), np.sum(np.isnan(data[v])), len(np.array(data[v])), np.mean(data[v]), np.nanmean(data[v]), kwargs)

            if x_ not in W.keys() or y_ not in W.keys():
                continue

            # cross calculate
            Wxy = W[x_] * W[y_].conjugate()
            #print(x_, np.mean(W[x_]), np.mean(W[x_].conjugate()))
            #print(y_, np.mean(W[y_]), np.mean(W[y_].conjugate()))
            #print('Wxy', np.mean(Wxy))
            if 'Wxy' not in self.ignore:
                self.Wxy[n_] = Wxy

            if 'transform_wave' not in self.__dict__.keys():
                # wavelet coeffifient to flux
                self.transform_wave = (1 / sj[:, None]) * kwargs['dt'] * kwargs['dj'] / \
                    kwargs['wavelet'].cdelta
            if 'period' not in self.__dict__.keys():
                self.period = 1/freq
            if 'scale' not in self.__dict__.keys():
                self.scale = sj
            if 'coi' not in self.__dict__.keys():
                self.coi = coi
            
            self.FWxy[n_] = (np.apply_along_axis(lambda x: x * np.ravel(self.transform_wave), 0,
                                            Wxy)).T.real
            #print('self.transform_wave', np.mean(self.transform_wave))
            #print('Wxy', np.mean(Wxy))

            if 'FWxy_spectra' not in self.ignore:
                self.FWxy_spectra[n_] = np.nanmean(self.FWxy[n_], axis=0)

            if 'FWxy_std' not in self.ignore:
                self.FWxy_std[n_] = list(
                    np.zeros(np.array(self.FWxy_std).shape))
            
            # clear
            for v in xy_:
                if v in W.keys() and v not in tt.flist(xy[i:]) or lowmemory == True:
                    del W[v]
        
        if verbosity: print(time.time() - t0, 's')
        return self

    '''def save_flux(self, output_path):
        prefix = prefix.format(min(self.data[self.t_name]).strftime('%Y%m%d%H%M'))
        self.dump(path=output_path+prefix)
        return'''
    
    def calculate_coi(self, lenght=None):
        dt = self.dt
        n0 = len(self.TIMESTAMP) if lenght is None else (
                int(1/dt)*lenght)
        coi = (n0 / 2 - np.abs(np.arange(0, n0) - (n0 - 1) / 2))
        coi = self.wavelet.flambda() * self.wavelet.coi() * dt * coi
        
        self.coi = coi

    def calculate_coimask(self, coi):
        coi_mask = np.ones((len(self.period), len(self.TIMESTAMP)), dtype=bool)
        for i in range(coi_mask.shape[1]):
            coi_period_threshold = list(self.period).index(
                tt.nearest(self.period, coi[i])[0])
                
            for j in range(coi_period_threshold, coi_mask.shape[0]):
                coi_mask[j][i] = False

        self.coi_mask = coi_mask.T
        return self
    
    def print(self):
        print('\n'.join("%s: %s" % item for item in vars(
            self).items() if item[0] not in ['timelst', 'TIMESTAMP', 'ec', 'qc_ec', 'wv', 'qc_wv', 'wv_std', 'wvlst']))
        return    
    
    def load_flux(i_, t_, prefix, read_path="./wavelets_for_flux/data/tmp/flux/"):
            filepath = read_path + \
                prefix+"_"+str(t_)[:10] + ".dat"

            if os.path.isfile(filepath):
                wv_Fi = waveletflux().load(path=filepath)
            else:
                return None
            return {i_: wv_Fi}
    
    def to_DataFrame(self, all_freqs=False, **kwargs):
        fluxdata = {}

        if all_freqs:
            for xy, w in self.FWxy.items():
                for i, f in enumerate(self.period):
                    _name = xy.replace('*', '_') + '_freq_'+str(f)[:10]

                    self.__dict__[_name] = np.array(w)[:, i]
                    self.col_vars = self.col_vars + [_name]

        for k in self.__dict__.keys():
            if k in self.id_vars or k in self.col_vars or k in kwargs.keys():
                v = self.__dict__[k]
                if isinstance(v, dict):
                    for k_, v_ in v.items():
                        _name = f'{k}_{k_}'.replace('*', '_')[:10] if k != 'Fxy' else f'{k_}'.replace('*', '_')[:10]
                        i = 0
                        while _name in fluxdata.keys():
                            _name += f'_{i}'
                            i += 1
                            if i > 10: break
                        fluxdata.update({_name: v_})
                elif v is not None:
                    if kwargs and k in kwargs.keys():
                        fluxdata.update({kwargs[k]: v})
                    else:
                        fluxdata.update({k: v})

        fluxdf = pd.DataFrame(fluxdata)

        # move timestamp to first column
        c = fluxdf.pop(self.t_name)
        fluxdf.insert(0, self.t_name, c)

        return fluxdf

    def collapse_flux(self, breaks, coi=None, verbosity=0):
        cself = waveletflux(**self.__dict__)

        breaks = list(set([tt.nearest(cself.period, b)[
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
            f0_ = tt.nearest(cself.period, breaks[i])[
                0]
            f0 = list(cself.period).index(f0_)
            f1_ = tt.nearest(cself.period, breaks[i+1])[
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
        #iself = waveletflux(**self.__dict__)

        if max_freq != None:
            self.wv_flux_max_int_freq = tt.nearest(self.period, max_freq)[0]
            f1 = list(self.period).index(self.wv_flux_max_int_freq)
            #[i for i, x in enumerate(self.period) if x == self.wv_flux_max_int_freq][0]
        else:
            self.wv_flux_max_int_freq = f1 = None

        if min_freq == None:
            self.wv_flux_min_int_freq = f0 = 0
        else:
            self.wv_flux_min_int_freq = tt.nearest(self.period, min_freq)[0]
            f0 = list(self.period).index(self.wv_flux_min_int_freq)
        # use list.index(pfreq) ^

        for xy, w in self.FWxy.items():
            self.Fxy[xy] = np.nansum(np.array(w)[:, f0:f1], axis = 1)
            #self.FWxy_std[xy] = np.array(self.FWxy_std[xy])[
            #    :, f0:f1].sum(axis=1)

        self.coi_flag = (~np.array(coi)[:, f0:f1]).sum(axis=1).astype(bool) if coi is not None else None

        return self
        
    def average_flux(self, wave=None, t0=None, set_avg_period=None,
                     verbosity=0, **kwargs):
        #aself = waveletflux(**self.__dict__)

        t = self.t if 't' in self.__dict__.keys() else self.TIMESTAMP
        t0 = 0 if t0 == None else list(t).index(t0)

        for xy, w in self.FWxy.items():
            wave = np.array(w)# if wave is None else np.array(wave)

            '''loop where if 1dim -> 2dim, then mean, so back to 1dim'''
            TMSTMPround = np.array(pd.to_datetime(t).ceil(
                str(set_avg_period/60)+'Min').strftime('%Y%m%d%H%M%S.%f').astype(float))
            #block = np.append(block.reshape(-1,1), wave, axis=1)

            self.TIMESTAMP = pd.to_datetime(
                np.unique(TMSTMPround).astype(str), format='%Y%m%d%H%M%S.%f')

            breaks = np.unique(TMSTMPround, return_index=True)[1]
            breaks.sort()
            breaks = breaks[1:] if breaks[0] == 0 else breaks
            
            self.FWxy[xy] = np.array([np.nanmean(p, axis=0) for p in np.split(
                wave, breaks, axis=0)])
            self.FWxy_std[xy] = np.array([np.nanstd(p, axis=0) for p in np.split(
                wave, breaks, axis=0)])

        self.coi_mask = np.array([np.nanmin(p, axis=0) for p in np.split(
            self.coi_mask, breaks, axis=0)])
        self.avg_period = set_avg_period
        return self
    
    def ecfreqs(self):
        _, self.ecfreqs = wavelet_nearest_freq(self.meta['wvlst'], self.ec, self.meta['period'])
        return self
    
    def screen(self, valarr, affected):
        for va in list(affected):
            if isinstance(self.__dict__[va], dict):
                self.__dict__[va] = {k: np.array(v)[valarr] for k, v in self.__dict__[va].items()}
            else:
                self.__dict__[va] = np.array(self.__dict__[va])[valarr]
        return self
            

if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if '=' not in a]
    kwargs = dict([a.split('=') for a in sys.argv[1:] if '=' in a])
    waveletflux(**kwargs).get_flux(*args, **kwargs)


def wavelet_nearest_freq(wvlst, obs, period):
    wvlst_cs = np.array([np.cumsum(f) for f in wvlst])
    _freqs = [np.where(wvlst_cs[f] == tt.nearest(wvlst_cs[f], obs[f])[0])[
        0][0] for f in range(len(wvlst))]
    _periods = [period[f] for f in _freqs]
    return _freqs, _periods
