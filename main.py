"""Required libraries"""

import threading, multiprocessing, queue
import warnings
import pycwt
import time
import pathlib
import copy
import scipy as sp
from functools import reduce
import importlib.util
import sys
import numpy as np
import pandas as pd
import os
import re
import glob
import __gargantua__ as mm
from __scripts__ import gargantua as gg
from __scripts__ import common as gt
from Lib.open_flux.scripts import get_data, get_rawdata, wavelet_flux as wc, pre_processing as prep, post_processing as posp, QA_QC, multilevel_decomposition as mld
from Lib.open_flux.scripts.corrections.despike import mauder2013
posp.QAQC = QA_QC

pcd = pathlib.Path(__file__).parent.resolve()


def savefrequency(ymd, freq="1D"):
    if not isinstance(ymd, pd.DatetimeIndex):
        s, e, f = ymd
        ymd = pd.date_range(s, e, freq=f)
    #return time list and label, check what we actually need 
    slowtime = ymd.floor(freq)
    strtime = [re.sub('[-: ]', '', t) for t in slowtime.strftime('%')]
    if freq.endswith("D"): strtime = [t[:8] for t in strtime]
    if freq.endswith("H") or freq.endswith("Min"): strtime = [t[:12] for t in strtime]
    return slowtime, strtime


def list_of_dates(s, e=None, f=None):
    if isinstance(s, pd.DatetimeIndex):
        return list(s)
    return list(pd.date_range(s, e, freq=f))


dailyb = {
    "block": [[["RawCorrections"], ["run_eddycovariance"], ["run_uncertainty"]], 
              [['run_cwt'], ['run_dwt']],
              [['yearlyEC'], ['yearlyDWT'], ['yearlyCWT']],
              [['unify_dataset']],
              [['flag_dataset']]]
}


def daily(path, addons=[], d0=None, d1=None, δ=2, *args, **kwargs):
    from datetime import datetime, timedelta

    byesterday = (datetime.today() - timedelta(δ)) if d0==None else pd.to_datetime(d0)
    yesterday = (datetime.today() - timedelta(1)) if d1==None else pd.to_datetime(d1)
    rangedays = (byesterday.strftime('%Y%m%d0000'), yesterday.strftime('%Y%m%d2359'))

    
    ymd = [rangedays[0], rangedays[-1], '30Min']
    yyy = [y for y in range(byesterday.year, yesterday.year+1)]
    kwargs_ = {'__init__': {'<TIME_BEGIN>': rangedays[0], '<TIME_END>': rangedays[1]},
               'unify_dataset': {'ymd': yyy},
               'flag_dataset': {'ymd': yyy}}
    kwargs_.update(kwargs)
    mm.api(setup=mm.menufromfile(path, *addons),
        block=[[['RawCorrections', 'run_eddycovariance', 'run_uncertainty']],
               [['run_cwt', 'run_dwt']],
               [['yearlyEC', 'yearlyDWT', 'yearlyCWT']],
               [['unify_dataset']],
               [['flag_dataset']]],
        **kwargs_)
    return

"""
def __routine__(path, addons=[], *args, selected=[], **kwargs):
	# GET SETUP
    #setup = gg.readable_file(path).safe_load().to_refdict()
    setup = menufromfile(path, *addons, **kwargs)
    '''
    setup = gg.readable_file(path).safe_load().to_dict()
    for add in addons:
        setup = gt.update_nested_dict(
            setup, gg.readable_file(add).safe_load().to_dict())
    setup = gt.update_nested_dict(setup, kwargs)
    setup = gt.referencedictionary(setup, kinit=True)
    '''

    # UPDATE SETUP 
    # e.g.: 
    # setup['__init__']['<START_DATE>'] = '20230323' 
    # kwargs = {(['__init__', '<START_DATE>'], '20230323)'} 
    if selected == []:
        selected = [k for k, v in setup['__init__']['OPEN_DEFAULTS']['selected_functions'].items() if v['state']]
    #selected = blocks if blocks else [[[k] for k, v in setup['__init__']['OPEN_DEFAULTS']['selected_functions'].items() if v['state']]]   
    
    # RUN
    kw = {f: {'fc': gg.LazyCallable(gt.trygetfromdict(setup, [f, '__init__', 'path'], setup['__init__']['path']), 
                                    gt.trygetfromdict(setup, [f, '__init__', 'function'], f)),
           'kw': {k: v for k, v in setup[f].items() if k.startswith('__')*k.endswith('__')==False}} \
              for f, _ in setup.items() if f.startswith('__')*f.endswith('__')==False and f in selected}
    fc = gg.LazyCallable(gt.trygetfromdict(setup, ['__routine__', '__init__', 'path'], setup['__init__']['path']), 
                         setup['__routine__']['__init__']['function']).__get__().fc
    kw.update({k: v for k, v in setup['__routine__'].items() if k.startswith('__')*k.endswith('__')==False})
    
    fc(**kw)
    return
"""



def run_preparation(ymd, multiprocess=1, popup=True, verbosity=True, **kwargs):
    if verbosity: print('\nRUNNING RAW DATA PROCESSING\n')
    
    kwargs.update({'result': False}) # do not return dataset

    # add .data to return dataframe
    _ = gt.multiprocess_framework(
        prep.universalcall, multiprocess=multiprocess, loopvar=list_of_dates(*ymd), verbosity=verbosity, **kwargs)
    
    if popup:
        gt.popup()
    return


def run_metadata(ymd, averaging=[30], fixed_values={}, output_path=None, verbosity=1):
    
    return

def run_bioclimatology(ymd, raw_kwargs, averaging=[30], overwrite=False, output_avg=True, output_path=None, verbosity=1):
    if verbosity: print('\nRUNNING BIOCLIMATOLOGY\n')

    bm_avg = gg.LazyCallable("Lib/open_flux/scripts/avg_fast.jl",
                             "biomet_all").__get__().fc
    df_avg = gg.LazyCallable("Lib/open_flux/scripts/avg_fast.jl",
                             "average_similar_columns").__get__().fc

    for a in averaging:
        for y in np.unique(ymd):
            print(y, ' '*5, end='\r')

            if not overwrite and os.path.exists(output_path.format(y, str(a).zfill(2))):
                if verbosity>1: 
                    warnings.warn(
                    "exists: File already exists ({}).".format(y, str(a).zfill(2)))
                    
                if not os.path.exists(output_path.format(y, str(a).zfill(2)).rsplit('.', 1)[0] + '.csv'):
                    df_avg(output_path.format(y, str(a).zfill(2)).rsplit('.', 1)[0] + '.raw.csv',
                           output_path.format(y, str(a).zfill(2)))
                continue

            dates_list = {}
            files_list = {}
            heads_list = {}

            for root, directories, files in os.walk(raw_kwargs['mother_path']):
                for name in files:
                    fullpath = os.path.join(root, name)
                    dateparts = re.findall(
                        '_BM_(' + str(y) + '[0-9]*)_(L[0-9]{2}_F[0-9]{2}).csv', name, flags=re.IGNORECASE)
                    headparts = re.findall(
                        '_BMHEADER_([0-9]*)_(L[0-9]{2}_F[0-9]{2}).csv', name, flags=re.IGNORECASE)

                    if len(dateparts) == 1:
                        id_ = dateparts[0][1]
                        dates_list[id_] = [dateparts[0][0]] if id_ not in dates_list.keys(
                        ) else dates_list[id_] + [dateparts[0][0]]
                        files_list[id_] = [fullpath] if id_ not in files_list.keys(
                        ) else files_list[id_] + [fullpath]

                    if len(headparts) == 1:
                        id_ = headparts[0][1]
                        heads_list[id_] = [fullpath] if id_ not in heads_list.keys(
                        ) else heads_list[id_] + [fullpath]
            
            if files_list:
                """
                for k in files_list.keys():
                    jlcsv = raw_kwargs['jlcsv'] if 'jlcsv' in raw_kwargs.keys() else {}

                    if 'header' not in jlcsv and heads_list and heads_list[k]:
                        jlcsv['header'] = pd.read_csv(
                            heads_list[k][0]).columns.tolist()
                    if "log" in jlcsv and "{}" in jlcsv["log"]:
                        jlcsv['log'] = copy.deepcopy(jlcsv['log']).format(f"{y}_{k}")
                    
                    _out_ = output_path.format(
                        y, str(a).zfill(2)).rsplit('.', 1)
                    
                    bm_avg(files_list[k], dates_list[k],
                        jlcsv, a, f"{_out_[0]}_{k}.{_out_[1]}")
                """                
                jlcsv = [copy.deepcopy(raw_kwargs['jlcsv']) if 'jlcsv' in raw_kwargs.keys() else {
                    } for _ in files_list.keys()]

                for j, k in enumerate(files_list.keys()):
                    if "log" in raw_kwargs['jlcsv'] and "{}" in raw_kwargs['jlcsv']["log"]:
                        jlcsv[j]['log'] = jlcsv[j]['log'].format(y)

                    if 'header' not in raw_kwargs['jlcsv'] and heads_list and heads_list[k]:
                        jlcsv[j]['header'] = pd.read_csv(
                            heads_list[k][0]).columns.tolist()
                del j, k
                
                bm_avg(list(files_list.values()), list(dates_list.values()),
                       jlcsv, a, output_path.format(y, str(a).zfill(2)).rsplit('.', 1)[0] + '.raw.csv')

                df_avg(output_path.format(y, str(a).zfill(2)).rsplit('.', 1)[0] + '.raw.csv',
                       output_path.format(y, str(a).zfill(2)))
                
    return


def run_eddycovariance(ymd, raw_kwargs, unc_kwargs, *a, script=["Lib/open_flux/scripts/avg_fast.jl", "eddycov_average"], **kw,):
    print('\nRUNNING EDDY COVARIANCE\n')
    return run_covariancestats(ymd, raw_kwargs, unc_kwargs, *a, _script_=script, **kw)


def run_uncertainty(ymd, raw_kwargs, *a, script=["Lib/open_flux/scripts/avg_fast.jl", "randomuncertainty"], **kw):
    print('\nRUNNING UNCERTAINTY\n')
    return run_covariancestats(ymd, raw_kwargs, *a, _script_=script, **kw)

# Transform it to julia
# (implement check if already exists)
# implement .inprogress
# implement .yaml read to kwargs


def run_covariancestats(ymd, *args, output_path, _script_, averaging=[30], overwrite=False, file_duration="1D", verbosity=1):
    if isinstance(_script_, (list, tuple)):
        script = gg.LazyCallable(*_script_).__get__().fc
    elif isinstance(_script_, str):
        script = gg.LazyCallable(
            "Lib/open_flux/scripts/avg_fast.jl", _script_).__get__().fc
    elif isinstance(_script_, queue.Queue):
        script = _script_.get()
        #_script_.put(script)

    #ymd = np.unique(['{}{}{}'.format(d.year, str(d.month).zfill(2), str(d.day).zfill(2)) for d in list_of_dates(*ymd)])
    ymd = np.unique(savefrequency(ymd, file_duration)[1])
    
    todo = []

    # if already done ALL averaging, continue
    for a in averaging:        
        for i, y in enumerate(ymd):
            s = args[0]['suffix'] if 'suffix' in args[0].keys() else ''
            if not overwrite and os.path.exists(output_path.format(s, y, str(a).zfill(2))):
                if verbosity>1: warnings.warn("exists: File already exists ({}).".format(y))
                continue
            else:
                todo += [y]
    if not todo:
        if verbosity>=1: warnings.warn("exists: All files already exists.")
        return
    
    for a in averaging:        
        for i, y in enumerate(ymd):
            jlargs = []
            
            curoutpath = output_path.format(s, y, str(a).zfill(2))

            # if already done averaging, continue
            s = args[0]['suffix'] if 'suffix' in args[0].keys() else ''
            if not overwrite and os.path.exists(curoutpath):
                if verbosity>1: warnings.warn("exists: File already exists ({}).".format(y))
                #if os.path.exists(curoutpath_inprog): os.remove(curoutpath_inprog)
                continue

            curoutpath_inprog = curoutpath.rsplit(".", 1)[0] + ".inprogress"
            if gt.checkifinprogress(curoutpath_inprog):
                continue
            
            for _arg in args:
                _arg_kw = gt.update_nested_dicts(os.path.join(_arg['path'], 'readme.txt'), 
                                                 #*[os.path.join(_arg['path'], p, 'readme.txt') for p in os.listdir(_arg['path']) if os.path.isdir(os.path.join(_arg['path'], p))],
                                                 {'jlcsv': _arg['jlcsv'], 'FILE_RAW': {k: v for k, v in _arg.items() if not isinstance(v, dict)}},
                                                 fstr=lambda d: gg.readable_file(d).safe_load().to_dict())
                _arg['jlcsv'] = _arg_kw['jlcsv']
                found_files = gt.get_files_paths_using_regex(_arg['path'], startswith=str(y), pattern=_arg_kw['FILE_RAW']['file_pattern'])
                if found_files:
                    cor_dates, cor_files = list(zip(*found_files.items()))
                else:
                    if verbosity>1: warnings.warn("exit1: No file was found ({}).".format(y))
                    if os.path.exists(curoutpath_inprog): os.remove(curoutpath_inprog)
                    continue
                del found_files
                #print(gt.flist(cor_files), "\n", cor_dates)
                #return
                jlargs.append([gt.flist(cor_files), cor_dates, _arg['jlcsv']])

            """
            found_files = gt.get_files_paths_using_regex(unc_kwargs['mother_path'], startswith=str(y), pattern=unc_kwargs['file_pattern'])
            if found_files:
                unc_dates, unc_files = list(zip(*found_files.items()))
            else:
                if verbosity>1: warnings.warn("exit1: No file was found ({}).".format(y))
                if os.path.exists(curoutpath_inprog): os.remove(curoutpath_inprog)
                continue
            del found_files
            jlargs += [gt.flist(unc_files), unc_dates, unc_kwargs['jlcsv']]
            """
            if jlargs:
                try:
                    gt.mkdirs(os.path.dirname(output_path))
                    __temp__ = script(*jlargs, a, curoutpath)
                except Exception as e: #ValueError:
                    if verbosity>1: warnings.warn("exit2: Error when running script ({}).".format(y))
                    if verbosity>1: warnings.warn(str(e))
                    #if os.path.exists(curoutpath_inprog): os.remove(curoutpath_inprog)
                    #continue

            if os.path.exists(curoutpath_inprog): os.remove(curoutpath_inprog)
            if verbosity: print(y, f'{int(100*i/len(ymd))} %', end='\n')
    del script

def DEPRECATED_run_uncertainty(ymd, raw_kwargs, output_path, averaging=[30], overwrite=False, verbosity=1):
    if verbosity: print('\nRUNNING UNCERTAINTY\n')
    t0 = time.time()
    #ymd = np.unique(['{}{}{}'.format(d.year, str(d.month).zfill(2), str(d.day).zfill(2)) for d in list_of_dates(*ymd)])
    ymd = np.unique(savefrequency(list_of_dates(*ymd), '1D')[1])
    todo = []
    print("ymd", time.time()-t0)
    for a in averaging:        
        for i, y in enumerate(ymd):
            s = raw_kwargs['suffix'] if 'suffix' in raw_kwargs.keys() else ''
            if not overwrite and os.path.exists(output_path.format(s, y, str(a).zfill(2))):
                if verbosity>1: warnings.warn("exists: File already exists ({}).".format(y))
                continue
            else:
                todo += [y]
    if not todo:
        if verbosity>1: warnings.warn("exists: All files already exists.")
        return
    print("todo", time.time()-t0)

    jlfunc = gg.LazyCallable("Lib/open_flux/scripts/avg_fast.jl", "randomuncertainty").__get__().fc

    print("jlfunc", time.time()-t0)
    for a in averaging:
        for i, y in enumerate(todo):
            curoutpath = output_path.format(s, y, str(a).zfill(2))
            curoutpath_inprog = curoutpath.rsplit(".", 1)[0] + ".inprogress"
            if gt.checkifinprogress(curoutpath_inprog):
                continue

            t1 = time.time()
            found_files = gt.get_files_paths_using_regex(raw_kwargs['mother_path'], startswith=str(y), pattern=raw_kwargs['file_pattern'])
            if not i: print("found_files", len(found_files.keys()), time.time()-t0)
            if found_files:
                cor_dates, cor_files = list(zip(*found_files.items()))
            else:
                if verbosity>1: warnings.warn("exit1: No file was found ({}).".format(y))
                if os.path.exists(curoutpath_inprog): os.remove(curoutpath_inprog)
                continue
            del found_files
            
            try:
                __temp__ = jlfunc(gt.flist(cor_files), cor_dates, raw_kwargs['jlcsv'], 
                                a, curoutpath)
            except Exception as e: #ValueError:
                if verbosity>1: warnings.warn("exit2: Error when running script ({}).".format(y))
                if verbosity>1: warnings.warn(str(e))
                if os.path.exists(curoutpath_inprog): os.remove(curoutpath_inprog)
                continue

            if os.path.exists(curoutpath_inprog): os.remove(curoutpath_inprog)
            if verbosity: print(y, [len(cor_files), len(cor_dates)], f'{np.ceil(100*i/len(todo))} %', "{}mn".format(np.round((time.time()-t1)/60, 1)), end='\n')
    del jlfunc

def consolidate_yearly(ymd, path, pattern, output_path, averaging=None):
    print('\nCONSOLIDATING DATASET\n')
    #assert type(path) == type(pattern) == type(output_path)
    if isinstance(path, str): path = [path]
    if isinstance(pattern, str): pattern = [pattern]
    if isinstance(output_path, str): output_path = [output_path]
    
    for path_, patt_, outp_ in list(zip(path, pattern, output_path)):
        if averaging == None:
            # 'C:/Users/phherigcoimb/Desktop/INRAE_longfiles/PAUL/PARIS-Jus/output/EC/openflux/'
            averaging = set([re.findall('\.([0-9]+)mn\.', p)[0] for p in os.listdir(path_) if re.findall('\.([0-9]+mn)\.', p)])
        p = '' # ['', '_unc']

        for a in averaging: 
            for y in ymd:
                datf = pd.DataFrame()
                    
                try:
                    dates, files = list(zip(*gt.get_files_paths_using_regex(
                        path_, startswith=str(y), pattern=patt_.format(str(a).zfill(2))).items()))
                except ValueError:
                    continue

                if files:
                    print(os.path.basename(outp_.format(p, y, str(a).zfill(2))), ' '*15, end='\r')
                    
                for i, fs in enumerate(files): # for f in gt.flist([files]):
                    for f in fs:
                        # check if is not an empty file
                        if os.path.getsize(f):
                            tmp_ = pd.read_csv(f)
                            datf = pd.concat((datf, tmp_))
                
                gt.mkdirs(outp_.format(p, str(y), str(a).zfill(2)))
                datf.to_csv(outp_.format(p, str(y), str(a).zfill(2)), index=False)

                print(os.path.basename(outp_.format(p, y, str(a).zfill(2))), ': Saved.', ' '*15, end='\n', sep='')
                    
                del datf

def run_wt(ymd, varstorun, raw_kwargs, output_path, wt_kwargs={}, 
           method="dwt", Cφ=1, nan_tolerance=.3,
           averaging=[30], condsamp=[], integrating=30*60, 
           overwrite=False, saveraw=False, file_duration="1D", verbosity=1):
    """
    fs = 20, f0 = 1/(3*60*60), f1 = 10, fn = 100, agg_avg = 1, 
    suffix = "", mother = pycwt.wavelet.MexicanHat(),
    **kwargs):
    """
    assert method in [
        'dwt', 'cwt', 'fcwt'], "Method not found. Available methods are: dwt, cwt, fcwt"
    if verbosity: print(f'\nRUNNING WAVELET TRASNFORM ({method})\n')
    if method in ["cwt", "fcwt"]:
        if method == "fcwt" or "mother" not in wt_kwargs.keys() or wt_kwargs.get("mother") in ['morlet', 'Morlet', pycwt.wavelet.Morlet(6)]:
            Cφ = 5.271
        else:
            Cφ = 16.568
    
    dt = 1 / wt_kwargs.get("fs", 20)
    suffix = raw_kwargs['suffix'] if 'suffix' in raw_kwargs.keys() else ''
    
    _, _, _f = ymd
    ymd = gt.list_time_in_period(*ymd, file_duration)
    if method in ['dwt']:
        buffer = wc.bufferforfrequency_dwt(
            N=pd.to_timedelta(file_duration)/pd.to_timedelta("1S") * dt**-1,
            n_=_f, **wt_kwargs)/2
    else:
        buffer = wc.bufferforfrequency(wt_kwargs.get("f0", 1/(3*60*60))) / 2


    for i, yl in enumerate(ymd):
        date = re.sub('[-: ]', '', yl.strftime('%')[0])
        if file_duration.endswith("D"): date = date[:8]
        if file_duration.endswith("H") or file_duration.endswith("Min"): date = date[:12]
        
        # recheck if files exist and overwrite option
        # doesn't save time (maybe only save 5min)
        if not overwrite:
            avg_ = []
            for a in averaging:
                if not overwrite and os.path.exists(output_path.format(suffix, date, str(a).zfill(2))):
                    avg_ += [a]
            avg_ = list(set(averaging)-set(avg_))
            if not avg_:
                if verbosity > 1: warnings.warn("exists: File already exists ({}).".format(date))
                continue
        else:
            avg_ = [a for a in averaging]
        
        curoutpath_inprog = output_path.format(suffix, str(date), "").rsplit(".", 1)[
            0] + ".inprogress"
        if gt.checkifinprogress(curoutpath_inprog): continue
        
        # load files
        # data = get_rawdata.open_flux(lookup=yl, **raw_kwargs).data
        data = wc.loaddatawithbuffer(
            yl, d1=None, freq=_f, buffer=buffer, **raw_kwargs)
        if data.empty:
            if verbosity>1: warnings.warn("exit1: No file was found ({}).".format(date))
            if os.path.exists(curoutpath_inprog): os.remove(curoutpath_inprog)
            continue
        
        # ensure time is time
        data.TIMESTAMP = pd.to_datetime(data.TIMESTAMP)
        
        # ensure continuity
        data = pd.merge(pd.DataFrame({"TIMESTAMP": pd.date_range(*gt.nanminmax(data.TIMESTAMP), freq=f"{dt}S")}),
                            data, on="TIMESTAMP", how='outer').reset_index(drop=True)

        # main run
        # . collect all wavelet transforms
        # . calculate covariance
        # . conditional sampling (optional)
        # . save in dataframe and .csv
        φ = {}
        μ = {}
        dat_fullspectra = {a: [] for a in avg_}
        dat_fluxresult = {a: [] for a in avg_}

        # run by couple of variables (e.g.: co2*w -> mean(co2'w'))
        for xy, condsamp in [(v.split('*')[:2], v.split('*')[2:3]) for v in varstorun]:
            # run wavelet transform
            for v in xy + condsamp:
                if v not in φ.keys():
                    signal = np.array(data[v])
                    signan = np.isnan(signal)
                    N = len(signal)
                    Nnan = np.sum(signan)
                    if Nnan:
                        if (nan_tolerance > 1 and Nnan > nan_tolerance) or (Nnan > nan_tolerance * N):
                            warnings.warn(
                                f"Too much nans ({np.sum(signan)}, {np.sum(signan)/len(signal)}%) in {date}.")
                    if Nnan and Nnan < N:
                        signal = np.interp(np.linspace(0, 1, N), 
                                  np.linspace(0, 1, N)[signan == False],
                                  signal[signan==False])
                    φ[v], sj = wc.universal_wt(signal, method, **wt_kwargs, inv=True)
                    # apply despiking (Mauder et al.)

                    def __despike__(X):
                        N = len(X)
                        X = mauder2013(X)
                        Xna = np.isnan(X)
                        X = np.interp(np.linspace(0, 1, N), 
                                           np.linspace(0, 1, N)[Xna == False],
                                  X[Xna==False])
                        return X 
                    φ[v] = np.apply_along_axis(__despike__, 1, φ[v])
                    μ[v] = signan *1

            # calculate covariance
            Y12 = np.array(φ[xy[0]]) * np.array(φ[xy[1]]).conjugate() * Cφ
            print(date, ''.join(xy), Y12.shape, round(Y12.shape[1] / (24*60*60*20), 2), buffer)
            φs = {''.join(xy): Y12}
            μs = {''.join(xy): np.where(np.where(
                np.array(μ[xy[0]]), 0, 1) * np.where(np.array(μ[xy[1]]), 0, 1), 0, 1)}

            # conditional sampling
            φc = [np.array(φ[xy[0]]) * np.array(φ[c]).conjugate() for c in condsamp]
            φc = mld.conditional_sampling(Y12, *φc) if φc else {}
            φs.update({k.replace("xy", ''.join(xy)).replace(
                'a', ''.join(condsamp)): v for k, v in φc.items()})

            # repeats nan flag wo/ considering conditional sampling variables
            μs.update(
                {k: μs[k if k in μs.keys() else [k_ for k_ in μs.keys() if k.startswith(k_)][0]] for k in φs.keys()})

            # array to dataframe for averaging
            def __arr2dataframe__(Y, qc=np.nan, prefix=''.join(xy), 
                                  id=np.array(data.TIMESTAMP), icolnames=sj):
                colnames = ["{}_{}".format(prefix, l) for l in icolnames] if icolnames is not None else None
                __temp__ = wc.matrixtotimetable(id, Y, columns=colnames)
                __temp__["{}_qc".format(prefix)] = qc
                __temp__ = __temp__[__temp__.TIMESTAMP > min(yl)]
                __temp__ = __temp__[__temp__.TIMESTAMP <= max(yl)]
                return __temp__

            __temp__ = reduce(lambda left, right: pd.merge(left, right[['TIMESTAMP'] + list(right.columns.difference(left.columns))], on="TIMESTAMP", how="outer"),
                              [__arr2dataframe__(Y, μs[n], prefix=n) for n, Y in φs.items()])
            
            for a in avg_:
                __tempa__ = copy.deepcopy(__temp__)
                __tempa__["TIMESTAMP"] = pd.to_datetime(np.array(__tempa__.TIMESTAMP)).ceil(
                    str(a)+'Min')
                __tempa__ = __tempa__.groupby("TIMESTAMP").agg(np.nanmean).reset_index()

                maincols = ["TIMESTAMP", ''.join(xy)]
                if φc:
                    for c in ['++', '+-', '--', '-+']:
                        maincols += [''.join(xy) + c + ''.join(condsamp)]
                        __tempa__.insert(1, ''.join(xy) + c + ''.join(condsamp), np.sum(__tempa__[[
                            "{}_{}".format(''.join(xy) + c + ''.join(condsamp), l) for l in sj if dt*2**l < integrating]], axis=1))
                __tempa__.insert(1, ''.join(xy), np.sum(__tempa__[[
                    "{}_{}".format(''.join(xy), l) for l in sj if dt*2**l < integrating]], axis=1))

                dat_fullspectra[a] += [__tempa__]
                dat_fluxresult[a] += [__tempa__[maincols]]
                del __tempa__
        
        for a in avg_:
            dat_fullspectra[a] = reduce(lambda left, right: pd.merge(left, right, on="TIMESTAMP", how="outer"),
                                 dat_fullspectra[a])
            dat_fluxresult[a] = reduce(lambda left, right: pd.merge(left, right, on="TIMESTAMP", how="outer"),
                                 dat_fluxresult[a])
            
            gt.mkdirs(output_path.format(suffix, str(date), str(a).zfill(2)))
            dat_fullspectra[a].to_csv(output_path.format(
                suffix + "_full_cospectra", str(date), str(a).zfill(2)), index=False)
            dat_fluxresult[a].to_csv(output_path.format(
                suffix, str(date), str(a).zfill(2)), index=False)
                
        if os.path.exists(curoutpath_inprog): os.remove(curoutpath_inprog)
        if verbosity:
            print(date, len(yl), f'{int(100*i/len(ymd))} %', end='\n')
    return

def run_dwt(ymd, varstorun, raw_kwargs, output_path, dwv_kwargs={}, averaging=[30], condsamp=[],
                                integrating=15, overwrite=False, saveraw=False, file_duration="1D", verbosity=1):
    if verbosity: print('\nRUNNING MULTILEVEL DECOMPOSITION\n')
    lvl = dwv_kwargs.get("level", integrating+3)
    dwv_kwargs.update({"level": lvl})
    
    suffix = raw_kwargs['suffix'] if 'suffix' in raw_kwargs.keys() else ''
    
    # from START, END, FREQ to list of dates
    # ymd = np.unique(['{}{}{}'.format(d.year, str(d.month).zfill(2), str(d.day).zfill(2)) for d in list_of_dates(*ymd)])
    
    d0, d1, _f = ymd
    ymd = gt.list_time_in_period(*ymd, file_duration)    
    buffer = wc.bufferforfrequency(20/(2**dwv_kwargs['level'])) / 2

    for i, yl in enumerate(ymd):
        y = re.sub('[-: ]', '', yl.strftime('%')[0])
        if file_duration.endswith("D"): y = y[:8]
        if file_duration.endswith("H") or file_duration.endswith("Min"): y = y[:12]
        #y = '{}{}{}'.format(yl[0].year, str(yl[0].month).zfill(2), str(yl[0].day).zfill(2))

        # check if files exist and overwrite option
        if not overwrite:
            avg_ = []
            for a in averaging:
                if not overwrite and os.path.exists(output_path.format(suffix, y, str(a).zfill(2))):
                    avg_ += [a]
            avg_ = list(set(averaging)-set(avg_))
            if not avg_:
                if verbosity > 1: warnings.warn("exists: File already exists ({}).".format(y))
                continue
        else:
            avg_ = [a for a in averaging]
        
        curoutpath_inprog = output_path.format(suffix, str(y), "").rsplit(".", 1)[0] + ".inprogress"
        if gt.checkifinprogress(curoutpath_inprog): continue
        
        # find files that follows pattern (optional)
        """
        found_files = gt.get_files_paths_using_regex(raw_kwargs['mother_path'], startswith=str(y), pattern=raw_kwargs['file_pattern'])
        
        if found_files:
            cor_dates, cor_files = list(zip(*found_files.items()))
        else:
            if verbosity>1: warnings.warn("exit1: No file was found ({}).".format(y))
            if os.path.exists(curoutpath_inprog): os.remove(curoutpath_inprog)
            continue
        del found_files
        """
        # load files
        #df = get_rawdata.open_flux(lookup=yl, **raw_kwargs).data
        df = wc.loaddatawithbuffer(
            yl, d1=None, freq=_f, buffer=buffer, **raw_kwargs)
        
        if df.empty:
            if verbosity>1: warnings.warn("exit1: No file was found ({}).".format(y))
            if os.path.exists(curoutpath_inprog): os.remove(curoutpath_inprog)
            continue

        #df = reduce(lambda left, right: pd.concat([left, right]), [pd.read_csv(p) for p in gt.flist(cor_files)])
        df.TIMESTAMP = pd.to_datetime(df.TIMESTAMP)
        
        df = pd.merge(pd.DataFrame({"TIMESTAMP": pd.date_range(*gt.nanminmax(df.TIMESTAMP), freq="0.05S")}),
                            df,
                            on="TIMESTAMP", how='outer').reset_index(drop=True)

        dat_fullspectra = {a: [] for a in avg_}
        dat_fluxresult = {a: [] for a in avg_}

        # run by couple of variables (e.g.: co2*w -> mean(co2'w'))
        for xy, condsamp in [(v.split('*')[:2], v.split('*')[2:3]) for v in varstorun]:
            Xs = []
            Xnas = []
            for xy_ in xy:
                X = df[xy_]  # .interpolate(limit_direction="both")
                Xna = np.isnan(X)
                X = np.interp(np.linspace(0, 1, len(X)),
                                np.linspace(0, 1, len(X))[Xna == False],
                                X[Xna == False])
                Xnas += [np.isnan(X)]
                Xs += [X - np.mean(X)]

            # run multilevel decomposition
            #Y12, level = mld.multilevelflux.get_flux(*Xs, **dwv_kwargs)
            #Y12[:, (X1na | X2na)] = np.nan
            [Y1, Y2], level = mld.decompose(*Xs, **dwv_kwargs)
            Y12 = Y1 * Y2.conjugate()
            Ys = {''.join(xy): Y12}

            Xcs = []
            for cs_ in condsamp:
                X = df[cs_]  # .interpolate(limit_direction="both")
                Xna = np.isnan(X)
                X = np.interp(np.linspace(0, 1, len(X)),
                              np.linspace(0, 1, len(X))[Xna == False],
                              X[Xna == False])
                Xcs += [X - np.mean(X)]
            if len(Xcs)==2:
                [Xc1, Xc2], level = mld.decompose(*Xcs, **dwv_kwargs)
            elif len(Xcs) == 1:
                [Xc2], level = mld.decompose(*Xcs, **dwv_kwargs)
                Xc1 = Y1
            if Xcs:
                Xc12 = Xc1 * Xc2.conjugate()
            
            Ycs = mld.conditional_sampling(Y12, Xc12) if Xcs else {}
            nnames = {}
            for k in Ycs.keys():
                nname = str(k)
                for i, cs_ in enumerate(condsamp):
                    nname = nname.replace(f"x{i}", cs_)
                nnames[k] = nname
            for old, new in nnames.items():
                Ycs[new] = Ycs.pop(old)
            Ys.update({k.replace("xy", ''.join(xy)).replace('a', ''.join(condsamp)): v for k, v in Ycs.items()})

            # ARRAY TO DATAFRAME THEN AVG
            def __arr2dataframe__(Y, prefix=''.join(xy), id=np.array(df.TIMESTAMP)):
                __temp__ = wc.matrixtotimetable(id, Y, columns=[
                        "{}_{}".format(prefix, l) for l in range(level)] + ["{}_r".format(prefix)])
                __temp__["qc_{}".format(prefix)] = (Xnas[0] | Xnas[1]) * 1
                __temp__ = __temp__[__temp__.TIMESTAMP > min(yl)]
                __temp__ = __temp__[__temp__.TIMESTAMP <= max(yl)]
                return __temp__

            __temp__ = reduce(lambda left, right: pd.merge(left, right[['TIMESTAMP'] + list(right.columns.difference(left.columns))], on="TIMESTAMP", how="outer"),
                              [__arr2dataframe__(Y, n) for n, Y in Ys.items()])
            
            for a in avg_:
                __tempa__ = copy.deepcopy(__temp__)
                __tempa__["TIMESTAMP"] = pd.to_datetime(np.array(__tempa__.TIMESTAMP)).ceil(
                    str(a)+'Min')
                __tempa__ = __tempa__.groupby("TIMESTAMP").agg(np.nanmean).reset_index()
                #__tempcond = (nan_tolerance < 1 and np.array(__tempa__["qc"]) > nan_tolerance) + (np.array(__tempa__["qc"]) > (nan_tolerance * len(__tempa__) / len(__temp__)))
                #__tempa__.loc[__tempcond>0, [c for c in __tempa__.columns if c != "TIMESTAMP"]] = np.nan

                if Xcs:
                    for c in ['++', '+-', '--', '-+']:
                        __tempa__.insert(1, ''.join(xy) + c + ''.join(condsamp), np.sum(__tempa__[[
                            "{}_{}".format(''.join(xy) + c + ''.join(condsamp), l) for l in range(level) if l < integrating]], axis=1))
                __tempa__.insert(1, ''.join(xy), np.sum(__tempa__[[
                    "{}_{}".format(''.join(xy), l) for l in range(level) if l < integrating]], axis=1))

                dat_fullspectra[a] += [__tempa__]
                cs_ = [''.join(xy) + c + ''.join(condsamp) for c in ['++','+-', '--','-+']] if Xcs else []
                dat_fluxresult[a] += [__tempa__[["TIMESTAMP", ''.join(xy)] + cs_]]
                del __tempa__  # , __tempcond
        
        for a in avg_:
            dat_fullspectra[a] = reduce(lambda left, right: pd.merge(left, right, on="TIMESTAMP", how="outer"),
                                 dat_fullspectra[a])
            dat_fluxresult[a] = reduce(lambda left, right: pd.merge(left, right, on="TIMESTAMP", how="outer"),
                                 dat_fluxresult[a])
            
            gt.mkdirs(output_path.format(suffix, str(y), str(a).zfill(2)))
            dat_fullspectra[a].to_csv(output_path.format(suffix + "_full_cospectra", str(y), str(a).zfill(2)), index=False)
            dat_fluxresult[a].to_csv(output_path.format(suffix, str(y), str(a).zfill(2)), index=False)
                
        if os.path.exists(curoutpath_inprog): os.remove(curoutpath_inprog)
        if verbosity: print(y, len(yl), f'{int(100*i/len(ymd))} %', end='\n')


def run_cwt(ymd, varstorun, output_path, pathtoraw,
             fs=20, f0=1/(3*60*60), f1=10, fn=100, agg_avg=1, file_duration="1D",
             overwrite=False, averaging=[30], suffix="", verbosity=1, nan_tolerance=0.10,
             method='torrencecompo', mother=pycwt.wavelet.MexicanHat(),
             **kwargs):
    if verbosity: print(f'\nRUNNING CWT ({method})\n')
    if method=="fcwt": mother=pycwt.wavelet.Morlet(6)

    motherdict = {pycwt.wavelet.Morlet(6): ["morlet", "mor"],
     pycwt.wavelet.MexicanHat(): ["mexican hat", "mhat"]}
    if isinstance(mother, str):
        for k, v in motherdict.items():
            if str(mother).lower() in v: mother=k

    # variables
    # sj = s0*2^(j*dj), j=0,1,...,J
    # f0**-1 = f1**-1 * 2^((fn-1)*dj)
    dj = np.log2(f1/f0)/(fn-1)
    # J = np.log2(sj_max/s0)/dj
    # J = np.log2(30*60*f1)/dj
    d0, d1, _f = ymd
    ymd_ = gt.list_time_in_period(*ymd, file_duration)
    todo = []
    for a in averaging:        
        for i, dhm in enumerate(ymd_):
            date = re.sub('[-: ]', '', dhm.strftime('%')[0])
            if file_duration.endswith("D"): date = date[:8]
            if file_duration.endswith("H") or file_duration.endswith("Min"): date = date[:12]
            #date = saven.format(dhm[0].year, str(dhm[0].month).zfill(2), str(dhm[0].day).zfill(2), 
            #                       str(dhm[0].hour).zfill(2), str(dhm[0].minute).zfill(2))
            if not overwrite and os.path.exists(output_path.format(suffix, date, str(a).zfill(2))):
                if verbosity>1: warnings.warn("exists: File already exists ({}).".format(date))
                continue
            else:
                todo += [dhm]
    if not todo:
        if verbosity>1: warnings.warn("exists: All files already exists.")
        return
    
    buffer = wc.bufferforfrequency(f0) / 2
    
    for i, dhm in enumerate(todo):
        date = re.sub('[-: ]', '', dhm.strftime('%')[0])
        if file_duration.endswith("D"): date = date[:8]
        if file_duration.endswith("H") or file_duration.endswith("Min"): date = date[:12]

        #date = saven.format(dhm[0].year, str(dhm[0].month).zfill(2), str(dhm[0].day).zfill(2),
        #                       str(dhm[0].hour).zfill(2), str(dhm[0].minute).zfill(2))
        print(date)
        curoutpath_inprog = output_path.format(suffix, date, "").rsplit(".", 1)[0] + ".inprogress"
        if gt.checkifinprogress(curoutpath_inprog): continue
        
        data = wc.loaddatawithbuffer(
            dhm, d1=None, freq=_f, buffer=buffer, path=pathtoraw)
        if data is None or data.empty:
            if os.path.exists(curoutpath_inprog): os.remove(curoutpath_inprog)
            continue
        print(data.columns)
        xy = [v.split('*')[:2] for v in varstorun]
        xy = [(v,v) if len(v)==1 else v for v in xy]

        W = {}
        #Wxy = {}
        FWxy = {}

        dat_fullspectra = {a: [] for a in averaging}
        dat_fluxresult = {a: [] for a in averaging}

        #coi = wc.calculate_coi(len(data))
        #coi_ = wc.coitomask(coi, (fn, len(data)), 1/freqs)

        for xy in [v.split('*')[:2] for v in varstorun]:
            for v in xy:
                if v not in W.keys():
                    signal = np.array(data[v])
                    signan = np.isnan(signal)
                    N = len(signal)
                    Nnan = np.sum(signan)
                    # signal = np.array(signal[signan == False]) if np.sum(signan) else np.array(data[v])
                    
                    if Nnan:
                        signal = np.interp(np.linspace(0, 1, N), 
                                  np.linspace(0, 1, N)[signan == False],
                                  signal[signan==False])
                        
                    if method=='fcwt':
                        _j, wave = wc.cwt(signal, fs, f0, f1, fn, **kwargs)
                        sj = _j**-1
                        dt = fs
                    if method=='torrencecompo':
                        wave, sj, _, _, _, _ = pycwt.cwt(signal, dt=1/fs, s0=2/fs, dj=dj, J=fn-1, wavelet=mother)
                        dt = fs**-1
                    
                    if Nnan:                        
                        #wave = gt.insert_in_array(wave, [i for i, s in enumerate(signan) if s])
                        wave[:, signan] = np.nan
                        if (nan_tolerance > 1 and Nnan > nan_tolerance) or (Nnan > nan_tolerance * N):
                            warnings.warn(f"Too much nans ({np.sum(signan)}, {np.sum(signan)/len(signal)}%) in {date}.")

                    """
                    # Reconstruction (Torrence and Compo)
                    if method=='fcwt':
                        #dj = wc.djfromscales(sj, fn)
                        dt = fs
                        C_d = 0.776
                        Y_00 = np.pi**(-1/4)
                    elif method=='torrencecompo':
                        dt = fs**-1
                        C_d = mother.cdelta
                        Y_00 = mother.psi(0).real

                    a, b = wave.shape
                    c = sj.size
                    if a == c:
                        sj_ = (np.ones([b, 1]) * (sj**-1)).transpose()
                    elif b == c:
                        sj_ = np.ones([a, 1]) * (sj**-1)
                    
                    iwave = (dj * np.sqrt(dt) / C_d * Y_00 * (wave.real / (sj_ ** .5)).T)
                    """
                    #iwave = (wave.real.T / sj ** -.5).T * ((dj * dt ** .5) / (C_d * Y_00))
                    #iwave = np.array([dj*((1/fs)**(1/2))/((sj**(-1/2))*0.776*(np.pi**1/4))]).T * wave
                    W[v] = wave
            
            a, b = W[xy[0]].shape
            c = sj.size
            if a == c:
                N = b
                sj_ = (np.ones([b, 1]) * sj).transpose()
            elif b == c:
                N = a
                sj_ = np.ones([a, 1]) * sj
            k = ((dj * dt) / mother.cdelta)
            FWxy = W[xy[0]] * W[xy[1]].conjugate() / sj_ * k
            FWxy = FWxy.real

            #FWxy = cw.icwt(W[xy[0]] * W[xy[1]].conjugate(), sj, 1/fs, dj, wavelet=mother).real

            # apply coi
            #coi = wc.calculate_coi(N, dt=1/fs)
            #print("N", N, "a", a, "b", b, "c", c, np.array(coi).shape)
            #coi = wc.coitomask(coi, FWxy.shape, sj, false=np.nan)
            
            #FWxy = FWxy*coi

            __temp__ = wc.matrixtotimetable(np.array(data.TIMESTAMP), FWxy, columns=[
                "{}_{}".format(''.join(xy), l) for l in sj**-1])
            __temp__["qc_{}".format(''.join(xy))]=signan * 1
            __temp__ = __temp__[__temp__.TIMESTAMP > min(dhm)]
            __temp__ = __temp__[__temp__.TIMESTAMP <= max(dhm)]
            
            for a in averaging:
                __tempa__ = copy.deepcopy(__temp__)
                __tempa__["TIMESTAMP"] = pd.to_datetime(np.array(__tempa__.TIMESTAMP)).ceil(
                    str(a)+'Min')
                __tempa__ = __tempa__.groupby("TIMESTAMP").agg(np.nanmean).reset_index()
                #__tempcond = (nan_tolerance < 1 and np.array(__tempa__["qc"]) > nan_tolerance) + (np.array(__tempa__["qc"]) > (nan_tolerance * len(__tempa__) / len(__temp__)))
                #__tempa__.loc[__tempcond>0, [c for c in __tempa__.columns if c != "TIMESTAMP"]] = np.nan

                __tempa__.insert(1, ''.join(xy), np.sum(__tempa__[[
                    "{}_{}".format(''.join(xy), l) for l in sj**-1 if l > 1/(a*60)]], axis=1))

                dat_fullspectra[a] += [__tempa__]
                dat_fluxresult[a] += [__tempa__[["TIMESTAMP", ''.join(xy)]]]
                del __tempa__  # , __tempcond
        
        for a in averaging:
            dat_fullspectra[a] = reduce(lambda left, right: pd.merge(left, right, on="TIMESTAMP", how="outer"),
                                 dat_fullspectra[a])
            dat_fluxresult[a] = reduce(lambda left, right: pd.merge(left, right, on="TIMESTAMP", how="outer"),
                                 dat_fluxresult[a])
            
            gt.mkdirs(output_path.format(suffix, date, str(a).zfill(2)))
            dat_fullspectra[a].to_csv(output_path.format(suffix + "_full_cospectra", date, str(a).zfill(2)), index=False)
            dat_fluxresult[a].to_csv(output_path.format(suffix, date, str(a).zfill(2)), index=False)

        if os.path.exists(curoutpath_inprog): os.remove(curoutpath_inprog)
        if verbosity: print(date, f'{int(100*i/len(todo))} %', len(todo), end='\n')
    return
        
def DEPRECATED_run_wavelets(ymd, output_path, varstorun, prefix, wv_kwargs, prepare_kwargs, agg_avg=1,
                 overwrite=False, verbosity=1, nan_tolerance=0.01, **kwargs):
    if verbosity: print('\nRUNNING WAVELETS\n')

    output_path = os.path.join(output_path, prefix) if output_path else output_path
    wv_kwargs_ = {"dt": 0.05, "wavelet": pycwt.wavelet.MexicanHat(), "dj": 1/8}
    wv_kwargs_.update(wv_kwargs)
    wv_kwargs = wv_kwargs_
    if isinstance(wv_kwargs['mother'], str):
        wv_kwargs['mother'] = pycwt.wavelet.MexicanHat() if str(wv_kwargs['mother']).replace(' ', '').lower() in [
            'mh', 'mexicanhat', 'ricker'] else pycwt.wavelet.Morlet(6)
    
    #ymd = list(set(['{}{}{}{}{}'.format(d.year, str(d.month).zfill(2), str(d.day).zfill(2),str(d.hour).zfill(2),str(d.minute).zfill(2)) for d in list_of_dates(*ymd)]))
    ymd = np.unique(['{}'.format(d.strftime('%Y%m%d%H%M')) for d in list_of_dates(*ymd)])
    
    for i in range(len(ymd)-1):
        print(ymd[i+1])  
        if output_path and not overwrite and os.path.exists(output_path.format(ymd[i+1])):
            if verbosity>1: warnings.warn("exists: File already exists ({}).".format(ymd[i+1]))
            continue
        
        prepare_kwargs.update({"d0": ymd[i], "d1": ymd[i+1]})
        Fwv = wc.waveletflux(**wv_kwargs).prepare_data(
            **prepare_kwargs
            #'201906041130', '201906041200',
            #max_period=3*60*60,
            #path=r'C:\Users\phherigcoimb\OneDrive\INRAe\thesis-project\wavelets_for_flux\data\ICOS\FR-Gri\input\EC\2019'
        )
        
        if 'TIMESTAMP' not in Fwv.__dict__.keys() or Fwv.data.empty:
            continue
        
        #print("FW mean", [(k, np.nanmean(Fwv.data[k]), np.mean(Fwv.data[k])) for k in ["w", "co2", "h2o", "t_sonic"]])
        Fwv.get_flux(varstorun, tol=nan_tolerance)
        
        if Fwv.FWxy == {}:
            continue
        #print("FW mean", [(k, np.nanmean(w)) for k, w in Fwv.FWxy.items()])
        Fwv.calculate_coi()
        Fwv.calculate_coimask(Fwv.coi)
        Fwv.screen(
            valarr=(pd.DatetimeIndex(Fwv.TIMESTAMP) > pd.to_datetime(Fwv.meta['d0'])) *
            (pd.DatetimeIndex(Fwv.TIMESTAMP) <= pd.to_datetime(Fwv.meta['d1'])),
            affected=['TIMESTAMP', 'coi_mask', 'FWxy'])

        if 'TIMESTAMP' not in Fwv.__dict__.keys() or len(Fwv.TIMESTAMP) < 1:
            continue
        
        #print('> still', Fwv.data)
        Fwv.TIMESTAMP = pd.to_datetime(Fwv.TIMESTAMP)
        
        #Fwv.data = Fwv.data.reset_index(drop=True)
        del Fwv.data

        if output_path:
            if agg_avg:
                Fwv.average_flux(set_avg_period=int(agg_avg*60)).dump(output_path.format(ymd[i+1]))
            else:
                Fwv.dump(output_path.format(ymd[i+1]))
            #.rsplit('.', 1)[0] + '.rwflx')
        else:
            return Fwv
        """
        for a in averaging:
            Fwva = Fwv.copy().average_flux(
                set_avg_period=a*60).integrate_flux(
                max_freq=period*60)
            Fwva.to_DataFrame(all_freqs=all_freqs).to_csv(
                output_path.format(ymd[i+1]), index=False)
        """

    """
    if savefig:
        wv_x.plot(significance_level=None)
        #pyplot.axhline(np.log2(max_period),
        #                       ls=':', color='k')
        pyplot.axvline(min(t_), ls=':', color='k')
        pyplot.axvline(min(t_) + datetime.timedelta(days=1), ymax=np.log2(max_period),
                               ls=':', color='k')
        pyplot.savefig(
                    output_path + '.'.join(prefix.split('.')[:-1]) + ".png")
        pyplot.close()
    """


def DEPRACTED_run_crosswavelets(ymd, output_path, varstorun, prefix, agg_avg=1,
                      overwrite=False, verbosity=1, zero_tol=0.05, **kwargs):
    output_path = os.path.join(
        output_path, prefix) if output_path else output_path
    
    _fr = ymd.freq if isinstance(ymd, pd.DatetimeIndex) else ymd[2]
    ymd = np.unique(['{}'.format(d.strftime('%Y%m%d%H%M')) for d in list_of_dates(*ymd)])
    
    for _vtr in varstorun:
        for i in range(len(ymd)-1):
            print(ymd[i+1])
            if output_path and not overwrite and os.path.exists(output_path.format(ymd[i+1])):
                if verbosity>1: warnings.warn("exists: File already exists ({}).".format(ymd[i+1]))
                continue
            
            Fwvx = run_wavelets(ymd=pd.DatetimeIndex([ymd[i]], freq=_fr), 
                                varstorun=[_vtr[0]], output_path=None, agg_avg=None, **kwargs)
            Fwvy = run_wavelets(ymd=pd.DatetimeIndex([ymd[i]], freq=_fr), 
                                varstorun=[_vtr[1]], output_path=None, agg_avg=None, **kwargs)
            
            Fwvx.FWxypos = copy.deepcopy(np.array(Fwvx.FWxy))
            Fwvx.FWxypos[np.array(Fwvy.FWxy) < zero_tol] = np.nan
            
            Fwvx.FWxyneg = copy.deepcopy(np.array(Fwvx.FWxy))
            Fwvx.FWxyneg[np.array(Fwvy.FWxy) > -zero_tol] = np.nan

            Fwvx.FWxyres = copy.deepcopy(np.array(Fwvx.FWxy))
            Fwvx.FWxyres[(np.array(Fwvy.FWxy) <= -zero_tol) *
                         (np.array(Fwvy.FWxy) >= zero_tol)] = np.nan
            return Fwvx
            if agg_avg:
                Fwv.average_flux(set_avg_period=int(agg_avg*60)).dump(output_path.format(ymd[i+1]))
            else:
                Fwv.dump(output_path.format(ymd[i+1]))


def DEPRECATED_consolidate_wavelets(ymd, path, output_path, filefreq="30min", averaging=[30], integrating=[60], verbosity=1):
    if verbosity: print('\nCONSOLIDATING WAVELETS\n')
    
    print(len(re.findall("{}", output_path)))
    l = len(re.findall("{}", output_path))
    averaging = [averaging[0]] if l < 3 else averaging
    integrating = [integrating[0]] if l < 2 else integrating
    
    for a in averaging:
        for p in integrating:
            for y in ymd:
                wv_datf = pd.DataFrame()
                if verbosity: print(y)
                
                for d in pd.date_range(start=str(y)+"-01-01", end=str(y)+"-12-31", freq=filefreq):
                    if os.path.exists(path.format(d.strftime('%Y%m%d%H%M'))):
                        wv_datf = pd.concat([wv_datf, gt.datahandler.load(
                            path.format(d.strftime('%Y%m%d%H%M'))).average_flux(
                                set_avg_period=a*60).integrate_flux(
                                max_freq=p*60).to_DataFrame(all_freqs=True)])
                    else:
                        print(d)
                    continue
                v = [a, p, y][-l:]
                
                gt.mkdirs(output_path.format(*v))
                wv_datf.to_csv(output_path.format(*v), index=False)
                """
                    #[print(path.format(v.replace('_', ''), d.strftime('%Y%m%d%H%M'))) for v in varstorun]
                    t_0 = {v: gt.datahandler.load(path.format(v.replace('*', ''), d.strftime('%Y%m%d%H%M'))) for v in varstorun}
                    #print(t_0)
                    t_1 = [[
                        w.average_flux(set_avg_period=30*60).integrate_flux(60*60).to_DataFrame(wv=k, all_freqs=True),
                        #*[w.average_flux(set_avg_period=30*60).integrate_flux(i*60*30).to_DataFrame(wv=k+'_'+str(int(i*0.5))) \
                        #for i in [2,4,6,8,10,12,24,48]]
                        ] for k, w in t_0.items() if w]
                    #print([t.head(1) for t in t_1])
                    if t_1:
                        t_2 = reduce(lambda left, right: pd.merge(left, right, on=['TIMESTAMP'], how='outer'),
                            (gt.flist(t_1))).sort_values('TIMESTAMP')
                        wv_datf = wv_datf.append(t_2)
                        del t_2
                    del t_0, t_1
                #print(output_path.format(y))
                wv_datf.to_csv(output_path.format(y), index=False)
                """


def run_postprocessing(ymd, path, prefix, output_path, **kw):
    posp.unify_dataset(ymd, path, prefix, output_path.format(""), **kw)
    posp.flag_dataset(ymd, path, prefix, output_path.format("_flagged"), **kw)
    return


def unify_dataset(*a, **kw):
    return posp.unify_dataset(*a, **kw)


def flag_dataset(*a, **kw):
    return posp.flag_dataset(*a, **kw)


def gap_filling(*a, **kw):
    return posp.gap_filling(*a, **kw)


def revisit_rawdata_dateformat(path, pattern, datefrom, dateto):
    """correct date from raw file"""
    listtorun = gt.flist(gt.get_files_paths_using_regex(path, pattern=pattern).values())
    for i, v in enumerate(listtorun):
        print(v, i, len(listtorun), ' '*10, end='\r')
        df_ = pd.read_csv(v)
        try:
            pd.to_datetime(str(df_.loc[1, 'TIMESTAMP']), format=dateto)
            print(v, i, len(listtorun), 'ignored', df_.loc[1, 'TIMESTAMP'], ' '*10, end='\r')
            continue
        except ValueError:
            try:
                df_.loc[:, 'TIMESTAMP'] = df_.loc[:, 'TIMESTAMP'].apply(lambda e: pd.to_datetime('%.2f' % e, format=datefrom).strftime(dateto))
                
                gt.mkdirs(v)
                df_.to_csv(v, index=False)
                print(v, i, len(listtorun), 'saved', ' '*10, end='\r')
            except:
                print(v, i, len(listtorun), ' '*10)
                print(type(df_.loc[:, 'TIMESTAMP']), df_.loc[:2, 'TIMESTAMP'])


def revisit_rawdata_filtercols(path, pattern, keepcols=[], excludecols=[]):
    """filter columns to keep (good to save space)"""
    listtorun = gt.flist(gt.get_files_paths_using_regex(path, pattern=pattern).values())
    for i, v in enumerate(listtorun):
        print(v, i, len(listtorun), ' '*10, end='\r')
        df_ = pd.read_csv(v)
        if keepcols != []:
            df_ = df_[:, keepcols]
        if excludecols != []:
            df_ = df_[:, [c for c in df_.columns if c not in excludecols]]
        
        gt.mkdirs(v)
        df_.to_csv(v, index=False)
        

if __name__ == '__main__':
    #args = [a for a in sys.argv[3:] if '=' not in a]
    #kwargs = dict([a.split('=') for a in sys.argv[3:] if '=' in a])
    mm.main(path="Lib/open_flux/setup/readme.yaml", lib="open_flux", welcometxt="OpenFLUX", font="small")
