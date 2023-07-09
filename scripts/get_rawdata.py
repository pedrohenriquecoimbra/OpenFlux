"""
Functions to retrieve RAW data from sites and put in a standard format.
"""

import multiprocess as mp
from functools import reduce

import warnings
import datetime
from Lib.OpenFlux.scripts import common as tt
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_object_dtype
import os
import re
import numpy as np
from math import sqrt
import pathlib
import zipfile
from io import StringIO


DEFAULT_EDDYPRO = {
    "FILE_RAW":{
        "date_format": "%Y%m%d-%H%M",
        "file_pattern": "([0-9]{8}-[0-9]{4})_raw_dataset_.*.txt",
        "dt": 0.05,
        },
    "READ_CSV":{
        "sep": "\s+",
        "skiprows": 8,
        "na_values": [-9999],
        },
}

DEFAULT_FILE_RAW = {
    'file_pattern': '.*_raw_dataset_([0-9]{12}).csv', 
    'date_format': '%Y%m%d%H%M', 
    'dt': 0.05, 
    'tname': "TIMESTAMP", 
    'id': None,
    'datefomatfrom': '%Y%m%d%H%M%S.%f', 
    'datefomatto': '%Y-%m-%dT%H:%M:%S.%f'
}

DEFAULT_READ_CSV = {
}

DEFAULT_READ_GHG = {
    'skiprows': 7,
    'sep': r"\t"
}

DEFAULT_FMT_DATA = {
}


class FluxTowerRawData(tt.datahandler):
    def __init__(self, data=None, dt=None, **kwargs):
        if data is None:
            loopvar = kwargs.pop('lookup', [])
            result = tt.multiprocess_framework(
                open_flux, multiprocess=1, loopvar=loopvar, varname="lookup", fc_kw=kwargs, verbosity=0)
            self.__dict__.update(result.__dict__)
        
        else:
            assert dt is not None, 'Missing measurement frequency (dt).'
            self.data = data
            self.dt = dt
            self.__dict__.update(**kwargs)

    '''def __get__(fn, **kwargs):
        fn = re.sub('[ -]', '_', fn.lower())
        fc = tt.LazyCallable('wavelets_for_flux.scripts.get_rawdata.' + fn)
        
        return fc.__call__(**kwargs)'''

    """
    def __get__(self, fn, lookup, multiprocess=1, **kwargs):
        #fn = re.sub('[ -]', '_', fn.lower())
        fc = tt.LazyCallable(os.path.join(pathlib.Path(__file__).parent.resolve(), fn)) #'wavelets_for_flux.scripts.get_rawdata.'

        result = tt.multiprocess_framework(
            lambda i, y: {i: fc.__call__(lookup=y, **kwargs)}, multiprocess=multiprocess, loopvar=lookup, verbosity=0)
        
        '''
        if multiprocess <= 1:
            multiprocess = False
            result = fc.__call__(file_list=file_list, **kwargs)
        else:
            npools = min(multiprocess, mp.cpu_count()-1)
            pool = mp.Pool(npools)

            callback_dict = {}
            n_ = int(np.ceil(len(file_list)/npools))
            for i in range(npools):
                pool.apply_async(fc.__call__, kwds=dict(kwargs, **{'id': i, 'file_list': file_list[i*n_:(i+1)*n_]}),
                                 callback=lambda x: callback_dict.update(x))
            pool.close()
            pool.join()
            
            wrklist = sorted(callback_dict.items())
            wrklist = list(np.array(wrklist)[:, 1])
            result = FluxTowerRawData(
                pd.concat([l.data for l in wrklist]), dt=wrklist[0].dt)
        '''
        return result
    """
    def filter(self, items: dict):
        for k, v in items.items():
            if isinstance(v, tuple):
                self.data = self.data.loc[(self.data[k] > v[0])
                                          & (self.data[k] < v[1])].copy()
            else:
                self.data = self.data[self.data[k].isin(v)].copy()
        return self

    def rename(self, names: dict):
        self.data = self.data.rename(columns=names)
        return self

    def modify(self, items: dict):
        for k, v in items.items():
            self.data[k] = v
        return self

    def format(self, 
               cols={'t':'ts'}, 
               keepcols=['u', 'v', 'w', 'ts', 'co2', 'co2_dry', 'h2o', 'h2o_dry', 'ch4', 'n2o'],
               addkeep=[],
               colsfunc=str.lower, cut=False, **kwargs):
        
        if isinstance(self, pd.DataFrame):
            formated = self
        else:
            fmt_clas = FluxTowerRawData(**self.__dict__)
            formated = fmt_clas.data

        if colsfunc is not None:
            if isinstance(colsfunc, (list, tuple)):
                colsfunc = tt.LazyCallable(*colsfunc).__get__().fc
            formated.columns = map(colsfunc, formated.columns)
        #cols.update(kwargs)
        cols.update({v.lower(): k.lower() for k, v in kwargs.items() if isinstance(v, list)==False})
        cols = {v: k for k, v in {v: k for k, v in cols.items()}.items()}
        cols.update({'timestamp': 'TIMESTAMP'})
        #formated.TIMESTAMP = formated.TIMESTAMP.apply(np.datetime64)
        if cut:
            #formated = formated[[
            #    c for c in formated.columns if c in cols.keys()]]
            formated = formated.loc[:, np.isin(formated.columns, keepcols+addkeep+list(cols.keys()))]
        
        formated = formated.rename(columns=cols)

        if isinstance(self, pd.DataFrame):
            return formated
        else:
            fmt_clas.data = formated
            return fmt_clas
    
    def interpolate(self, cols=["co2", "w"], qcname="qc"):
        interpolated = FluxTowerRawData(**self.__dict__)
        interpolated.data[qcname] = 0
        for c_ in list(cols):
            interpolated.data[qcname] = interpolated.data[qcname] + 0 * \
                np.array(interpolated.data[c_])
            interpolated.data.loc[np.isnan(interpolated.data[qcname]), qcname] = 1
            interpolated.data[qcname] = interpolated.data[qcname].astype(int)
            interpolated.data[c_] = interpolated.data[c_].interpolate(method='pad')
            
        return interpolated
    '''
    def to_wv_class(self, xname="co2", yname="w", tname="TIMESTAMP", qcname="qc", **kwargs):
        import Lib.OpenFlux.scripts.wavelet_calculate as wv_c
        print('yes')
        result = wv_c.wv(x=np.array(self.data[xname]), 
                         y=np.array(self.data[yname]),
                         t=np.array(self.data[tname]),
                         #qc=np.array(self.data[qcname]),
                         dt=self.dt)
        print('yes2')
        result.__dict__.update(**kwargs)
        return result
    '''
    '''
    def to_bm_class(self, *args, **kwargs):
        import wavelets_for_flux.scripts.get_bm as get_bm
        result = get_bm.BMDataset(self, *args, **kwargs)
        return result
    '''


'''
Site-specific functions to organise data into a dataframe
Once harmonized it should be ready to be piped through the FluxTowerRawData class
Attention, it should include variables such as:
- gas concentration and wind on the right units;
Attention, variables should be continuous (gapfilled if needed);
'''


def _open_flux(path, date_format='%Y%m%d%H%M', dt=0.05, tname="TIMESTAMP", id=None, **kwargs):
    kwargs.pop('file_list', None)
    df_site = pd.DataFrame()

    for t, ps in path.items():
        assert len(ps) == 1, 'Unexpected input! More than one file for the same timestamp.'
        for p in ps:
            if os.path.exists(p):
                df_td = pd.read_csv(p, **kwargs)
                if tname not in df_td.columns:
                    df_td[tname] = pd.to_datetime(
                        t, format=date_format) + datetime.timedelta(seconds=dt) * df_td.index
                df_td.loc[:, 'file'] = t
                df_site = df_site.append(df_td)

    if id is not None:
        return {id: FluxTowerRawData(df_site, dt=dt)}
    else:
        return FluxTowerRawData(df_site, dt=dt)


def open_flux(path, lookup=[], fill=False, fmt={}, onlynumeric=True, verbosity=1, fkwargs={}, **kwargs):
    df_site = pd.DataFrame()
    
    folders = [path + p + '/' for p in os.listdir(path) if os.path.isdir(path + p)]
    folders = folders if folders else [path]
    
    for path_ in folders:
        df_td = pd.DataFrame()

        # read tips file        
        kw_ = tt.update_nested_dicts({"FILE_RAW": DEFAULT_FILE_RAW, "READ_CSV": DEFAULT_READ_CSV, "FMT_DATA": DEFAULT_FMT_DATA}, 
                                              os.path.join(path, 'readme.txt'), os.path.join(path_, 'readme.txt'),
                                              {"FILE_RAW": fkwargs, "READ_CSV": kwargs, "FMT_DATA": fmt},
                                              fstr=lambda d: tt.readable_file(d).safe_load().to_dict())
        kw = tt.metadata(**kw_['FILE_RAW'])
        kw_csv = kw_['READ_CSV']
        
        try:
            if ('header_file' in kw_csv.keys()) and (os.path.exists(kw_csv['header_file'])):
                kw_csv['header_file'] = "[" + open(kw_csv['header_file']).readlines()[0].replace("\n", "") + "]"
        except:
            None
        
        lookup_ = list(set([f.strftime(kw.date_format) for f in lookup]))
        files_list = {}

        for root, directories, files in os.walk(path_):
            for name in files:
                dateparts = re.findall(kw.file_pattern, name, flags=re.IGNORECASE)
                if len(dateparts) == 1:
                    files_list[dateparts[0]] = os.path.join(root, name)
                    
        for td in set(lookup_) & files_list.keys() if lookup_ != [] else files_list.keys():
            path_to_tdfile = files_list[td]
            if os.path.exists(path_to_tdfile):
                if path_to_tdfile.endswith('.gz'): kw_csv.update(**{'compression': 'gzip'})
                elif path_to_tdfile.endswith('.csv'): kw_csv.pop('compression', None)
                if path_to_tdfile.endswith('.ghg'):
                    with zipfile.ZipFile(path_to_tdfile, 'r') as zip_ref:
                        datafile = [zip_ref.read(name) for name in zip_ref.namelist() if name.endswith(".data")][0]
                    datafile = str(datafile, 'utf-8')
                    path_to_tdfile = StringIO(datafile)
                    # DEFAULT_READ_GHG
                    kw_csv.update(DEFAULT_READ_GHG)
                try:
                    df_td = pd.read_csv(path_to_tdfile, **kw_csv)
                    #for c in df_td.columns:
                    #    if c not in ['TIMESTAMP']:
                    #        df_td.loc[:, c] = pd.to_numeric(df_td.loc[:, c], errors = 'ignore')
                except Exception as e:# (EOFError, pd.errors.ParserError, pd.errors.EmptyDataError):
                    try:
                        if verbosity>1: warnings.warn(f'{e}, when opening {path_to_tdfile}, using {kw_csv}. Re-trying using python as engine and ignoring bad lines.')
                        df_td = pd.read_csv(path_to_tdfile, on_bad_lines='warn', engine='python', **kw_csv)
                    except Exception as ee:
                        warnings.warn(f'{ee}, when opening {str(path_to_tdfile)}, using {kw_csv}')
                        continue
                
                """
                if kw.tname in df_td.columns:
                    try:
                        df_td.loc[:, kw.tname] = pd.to_datetime(df_td.loc[:, kw.tname].astype(str))
                        print(max(df_td[kw.tname].dt.year), max(df_td[kw.tname]), min(df_td[kw.tname].dt.year))
                        assert max(df_td[kw.tname].dt.year) > 1990 and min(df_td[kw.tname].dt.year) > 1990
                    except:
                        df_td.rename({kw.tname+'_orig': kw.tname})
                """
                if kw.datefomatfrom == 'drop':
                    df_td = df_td.rename({kw.tname: kw.tname+'_orig'})
                
                if kw.tname not in df_td.columns or kw.datefomatfrom == 'drop':
                    if "date" in df_td.columns and "time" in df_td.columns:
                        df_td[kw.tname] = pd.to_datetime(
                            df_td.date + " " + df_td.time, format='%Y-%m-%d %H:%M')
                    else:
                        df_td[kw.tname] = pd.to_datetime(
                            td, format=kw.date_format) - datetime.timedelta(seconds=kw.dt) * (len(df_td)-1 + -1*df_td.index)
                            #td, format=kw.date_format) + datetime.timedelta(seconds=kw.dt) * (df_td.index)
                        df_td[kw.tname] = df_td[kw.tname].dt.strftime(
                            kw.datefomatto)
                else:
                    """
                    try:
                        # check if already in the good dateformat
                        #pd.to_datetime(df_td.loc[:, 'TIMESTAMP'].astype(
                        #    str), format=kw.datefomatto)
                        df_td.loc[:, kw.tname] = df_td.loc[:, kw.tname].apply(lambda e: pd.to_datetime('%.2f' % e, format=kw.datefomatto).strftime(kw.datefomatto))
                        print('to worked')
                    except ValueError:
                    """
                    try:
                        #print(df_td[kw.tname].dtypes, df_td[kw.tname][0], df_td[kw.tname].astype(str)[0], kw.datefomatfrom, 
                        #      pd.to_datetime(df_td[kw.tname][0])#, pd.to_datetime(df_td[kw.tname]).strftime(kw.datefomatto)[0]
                        #      )
                        if is_numeric_dtype(df_td[kw.tname]):
                            df_td.loc[:, kw.tname] = df_td.loc[:, kw.tname].apply(lambda e: pd.to_datetime('%.2f' % e, format=kw.datefomatfrom).strftime(kw.datefomatto))
                        #elif is_string_dtype(df_td[kw.tname]):
                        #    df_td.loc[:, kw.tname] = pd.to_datetime(df_td[kw.tname], format=kw.datefomatfrom).strftime(kw.datefomatto)
                        elif is_object_dtype(df_td[kw.tname]):
                            df_td.loc[:, kw.tname] = df_td.loc[:, kw.tname].apply(lambda e: pd.to_datetime(e).strftime(kw.datefomatto))
                        else:
                            df_td.loc[:, kw.tname] = pd.to_datetime(df_td[kw.tname], format=kw.datefomatfrom).strftime(kw.datefomatto)
                            #pd.to_datetime(df_td[kw.tname]).strftime(kw.datefomatto)
                        # df_td[kw.tname] = pd.to_datetime(df_td[kw.tname], format=kw.datefomatfrom).dt.strftime(kw.datefomatto)
                    except:
                        warnings.warn(f'error when converting {kw.tname} from {kw.datefomatfrom} to {kw.datefomatto}.')
                        continue
                
                df_td['file'] = td
                #df_site = df_site.append(df_td)
                df_site = pd.concat([df_site, df_td], ignore_index=True).reset_index(drop=True)
        
        if df_td.empty == False:
            break
        
    #print('df_td.empty ', df_td.empty)
    if onlynumeric:
        valcols = [i for i in df_site.columns if i.lower() not in [kw.tname.lower(), 'file']]
        _bf = df_site.dtypes
        #df_site.loc[:, valcols] = df_site.loc[:, valcols].apply(pd.to_numeric, errors='coerce')
        df_site[valcols] = df_site[valcols].apply(pd.to_numeric, errors='coerce')
        _af = df_site.dtypes
        if verbosity>1:
            _bfaf = []
            for (k, b) in _bf.items():
                if b!=_af[k]:
                    _nonnum = [s for s in np.unique(df_site[k].apply(lambda s: str(s) if re.findall('[A-z/]+', str(s)) else '')) if s]
                    _bfaf += ['{}, changed from {} to {}. ({})'.format(k, b, _af[k], ', '.join(_nonnum) if _nonnum else 'All numeric')]
            if _bfaf:
                warnings.warn(', '.join(_bfaf))
    """
    kw_fmt = DEFAULT_FMT_DATA
    kw_fmt = update_dict_using_readable_file(kw_fmt, os.path.join(path, 'readme.txt'), ['FMT_DATA'])
    kw_fmt = update_dict_using_readable_file(kw_fmt, os.path.join(path_, 'readme.txt'), ['FMT_DATA'])
    kw_fmt.update(fmt)
    """
    #if kw_fmt:
    df_site = FluxTowerRawData.format(df_site, **kw_['FMT_DATA'])

    if fill:
        if lookup:
            minmax = [min(lookup), max(lookup)]
        else:
            minmax = [np.nanmin(df_site[kw.tname]),
                      np.nanmax(df_site[kw.tname])]
        df_site = df_site.set_index(kw.tname).join(pd.DataFrame({kw.tname: pd.date_range(*minmax, freq=str(kw.dt) + ' S')}).set_index(kw.tname),
                how='outer').ffill().reset_index()
        #if 'co2' in df_site.columns and (abs(np.max(df_site.co2)) < 1000) and (abs(np.min(df_site.co2)) < 1000):
        #    df_site.loc[:, "co2"] = df_site.loc[:, "co2"] * 1000  # mmol/m3 -> μmol/m3
    
    if kw.id is not None:
        return {kw.id: FluxTowerRawData(df_site, dt=kw.dt)}
    else:
        return FluxTowerRawData(df_site, dt=kw.dt)

'''
def icos(path, file_pattern, file_list, date_format='%Y%j%H%M', dt=0.05, tname="TIMESTAMP",
            subfolder=True, cols=None, id=None, **kwargs):
    file_list = [f.strftime(date_format) for f in file_list]
    df_site = pd.DataFrame()
    files_list = {}

    for root, directories, files in os.walk(path):
        for name in files:
            dateparts = re.findall(file_pattern,  name, flags=re.IGNORECASE)
            if len(dateparts) == 1:
                files_list[dateparts[0]] = os.path.join(root, name)
    
    for td in set(file_list) & files_list.keys():
        path_to_tdfile = files_list[td]
        if os.path.exists(path_to_tdfile):
            # sep=sep, skiprows=8, na_values=[-9999])
            df_td = pd.read_csv(path_to_tdfile, **kwargs)
            if df_td.empty:
                continue
            df_td[tname] = pd.to_datetime(
                td, format=date_format) + datetime.timedelta(seconds=dt) * df_td.index
            df_td['file'] = td
            df_site = df_site.append(df_td)
    
    if id is not None:
        return {id: FluxTowerRawData(df_site, dt=dt)}
    else:
        return FluxTowerRawData(df_site, dt=dt)

"""
def generic(path, file_pattern, file_list, dt=0.05, tname="TIMESTAMP", id=None, **kwargs):
    df_site = pd.DataFrame()

    for td in np.array(file_list):
        path_to_tdfile = path.replace(file_pattern, td)
        if os.path.exists(path_to_tdfile):
            df_td = pd.read_csv(path_to_tdfile, **kwargs) # sep=sep, skiprows=8, na_values=[-9999])
            df_td[tname] = pd.to_datetime(
                td) + datetime.timedelta(seconds=dt) * df_td.index
            df_td['file'] = td
            df_site = df_site.append(df_td)

    if id is not None:
        return {id: FluxTowerRawData(df_site, dt=dt)}
    else:
        return FluxTowerRawData(df_site, dt=dt)
"""

def eddypro_raw_datasets(*args, **kwargs):
    kwargs.update({'date_format': '%Y%m%d-%H%M', 'file_pattern': '([0-9]{8}-[0-9]{4})_raw_dataset_.*.txt', 'sep': '\s+', 'skiprows': 8, 'na_values': [-9999]})
    #kwargs.update({'sep': '\s+', 'skiprows': 8, 'na_values': [-9999]})
    lookup = kwargs.pop('lookup', None)
    
    dta = open_flux(*args, **kwargs)
    if dta.data.empty:
        return dta
    
    #gapfilling
    dta.data = dta.data.set_index('TIMESTAMP').join(
        pd.DataFrame({'TIMESTAMP': pd.date_range(
                min(lookup), periods=len(np.array(lookup).ravel())*20*60*30, freq=str(0.05) + ' S')}).set_index('TIMESTAMP'),
                how='outer').ffill().reset_index()
    if 'co2' in dta.data.columns and (abs(np.max(dta.data.co2)) < 1000) and (abs(np.min(dta.data.co2)) < 1000):
        dta.data.loc[:, "co2"] = dta.data.loc[:, "co2"] * 1000  # mmol/m3 -> μmol/m3
    return dta

def fr_gri(path, file_list, dt=0.05, file_pattern='{TIMESTAMP}',
           tname="TIMESTAMP", expected_rows=20*60*30, id=None, **kwargs):
    df_site = pd.DataFrame()

    for td in np.array(file_list):
        #print(td, '  ', end='\r')
        if isinstance(path, str) and path.endswith('.csv'):
            path_to_tdfile = path.replace(file_pattern, td)
        else:
            path_to_tdfile = [path + e for e in os.listdir(path) if re.findall(
                td + file_pattern,  e, flags=re.IGNORECASE)]
            assert len(path_to_tdfile)<=1
            if len(path_to_tdfile)==1:
                path_to_tdfile = path_to_tdfile[0]
            else:
                continue

        if os.path.exists(path_to_tdfile):
            df_td = pd.read_csv(path_to_tdfile, na_values=[-9999], **kwargs)
            df_td[tname] = pd.to_datetime(
                td) + datetime.timedelta(seconds=dt) * df_td.index
            df_td['file'] = td

            #gapfilling
            original_data = df_td[tname]
            df_td = df_td.set_index(tname).join(
                pd.DataFrame({tname: pd.date_range(
                    str(td), periods=expected_rows, freq=str(dt) + ' S')}).set_index(tname),
                how='outer').ffill().reset_index()
            df_td['qc'] = [
                0 if e in original_data else 1 for e in df_td[tname]]

            #append
            df_site = df_site.append(df_td)
    
    if 'co2' in df_site.columns and (abs(np.max(df_site.co2)) < 1000) and (abs(np.min(df_site.co2)) < 1000):
            df_site.co2 = df_site.co2 * 1000  # mmol/m3 -> μmol/m3

    result = FluxTowerRawData(df_site, dt=dt)
    if id is not None:
        return {id: result}
    else:
        return result


def hu_hhs(path, file_pattern, file_list, dt=0.25, tname="TIMESTAMP", verbosity=0, id=None, **kwargs):
    df_site = pd.DataFrame()

    for td in np.array(file_list):
        path_to_tdfile = path.replace(file_pattern, td)
        if os.path.exists(path_to_tdfile):
            df_td = pd.read_csv(path_to_tdfile, sep='\s+',
                                names=['co2', 'h2o', 'w', 'ts'], na_values=[-9999, -999], **kwargs)

            if verbosity > 0: 
                print(len(df_td), df_td.index)
            
            """
            if len(df_td) < 24*60*60*(1/dt):
                for i in range(24*60*60*(1/dt)-len(df_td)):
                    df_td = df_td.append(pd.Series(dtype=object), ignore_index=True)
            """
            
            df_td = df_td[:24*60*60*4]
            df_td[tname] = pd.to_datetime(
                td, format="%y%m%d") + datetime.timedelta(seconds=dt) * df_td.index
            df_td['file'] = td

            if verbosity > 0:
                print(min(df_td[tname]), max(df_td[tname]), np.count_nonzero(np.unique(df_td[tname])))
            df_site = df_site.append(df_td)
    
    if id is not None:
        return {id: FluxTowerRawData(df_site, dt=dt)}
    else:
        return FluxTowerRawData(df_site, dt=dt)
'''
