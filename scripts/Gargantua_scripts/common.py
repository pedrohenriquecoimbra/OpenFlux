"""
Common functions
"""

import pathlib
import codecs
from contextlib import contextmanager
import inspect
#import imp
import sys
import os
import re
import time
import pickle
import json
import copy
#import time
import ast
from matplotlib.pyplot import isinteractive
#import warnings
import numpy as np
import pandas as pd
from itertools import chain
import itertools
import importlib
import multiprocess as mp
import folium
from matplotlib import cm
import geopandas as gpd
from shapely import geometry, ops
from pyproj import Transformer
from fiona.drvsupport import supported_drivers
supported_drivers['KML'] = 'rw'

from folium.plugins import BeautifyIcon
from matplotlib.colors import to_hex 

cfp = pathlib.Path(__file__).parent.resolve()


'''
UNIVERSALCALLS
'''
UNIVERSALCALLS = None
del UNIVERSALCALLS


def import_from_anywhere(module, package=None, n=None, caller=None):
    # get a handle on the module
    mdl = importlib.import_module(module, package)

    # is there an __all__?  if so respect it
    if "__all__" in mdl.__dict__:
        names = mdl.__dict__["__all__"]
    else:
        # otherwise we import all names that don't begin with _
        names = [x for x in mdl.__dict__ if not x.startswith("_")]

    if n:
        names = [x for x in names if not x in n]

    # now drag them in
    if not caller:
        globals().update({k: getattr(mdl, k) for k in names})
    else:
        globals().update({caller: {k: getattr(mdl, k) for k in names}})


def importlib_to_globals(fpath, **kwargs):
    if os.path.isfile(fpath) and fpath.endswith('.py'):
        name = os.path.basename(fpath)
        import_from_anywhere(os.path.relpath(os.path.dirname(fpath), pathlib.Path.cwd()).replace(
            '\\', '.') + '.' + name.split('.', 1)[0], pathlib.Path.cwd(), **kwargs)
        
    elif os.path.isdir(fpath):
        for root, _, files in os.walk(fpath):
            for name in files:
                if os.path.isfile(os.path.join(root, name)) and name.endswith('.py'):
                    #print(name)
                    #print(os.path.relpath(root, pathlib.Path.cwd()))
                    import_from_anywhere(os.path.relpath(root, pathlib.Path.cwd()).replace(
                        '\\', '.') + '.' + name.split('.', 1)[0], pathlib.Path.cwd(), **kwargs)


'''
MISC
'''
MISC = None
del MISC

def flist(lst):
    flst = []

    def _fl(lst):
        for l in lst:
            if isinstance(l, (list, np.ndarray)):
                _fl(l)
            else:
                flst.append(l)
    _fl(lst)
    return flst
#flist = lambda x: list(itertools.chain.from_iterable([x]))

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout
            

def popup(message='Done.'):
    import tkinter as tk
    from tkinter import messagebox
    popup = tk.Tk()

    tk.Frame(popup).pack()
    messagebox.showinfo('Window', message)
    popup.destroy()

    popup.mainloop()

def mkdirs(filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)

def covariance(x, y):
    # Subtracting mean from the individual elements
    sub_x = x - np.mean(x)
    sub_y = y - np.mean(y)
    cov = np.sum(sub_x*sub_y) / (len(x)-1)
    return cov

def minmax(x):
    return [np.min(x), np.max(x)]

def nanminmax(x):
    return [np.nanmin(x), np.nanmax(x)]

def agg(x, fc):
    return [f(x) for f in fc]

def symetric_quantile(x, q=0.95):
    return max(abs(np.nanpercentile(x, q*100)), abs(np.nanpercentile(x, (1-q)*100)))


def checkifinprogress(path, LIMIT_TIME_OUT=30*60):
    if os.path.exists(path) and (time.time()-os.path.getmtime(path)) < LIMIT_TIME_OUT:
        return 1
    else:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a+"):
            pass
        return 0

def nearest(items, pivot, direction=0):
    if direction == 0:
        nearest = min(items, key=lambda x: abs(x - pivot))
        difference = abs(nearest - pivot)
        
    elif direction == -1:
        nearest = min(items, key=lambda x: abs(x - pivot) if x<pivot else pd.Timedelta(999, "d"))
        difference = (nearest - pivot)
        
    elif direction == 1:
        nearest = min(items, key=lambda x: abs(x - pivot) if x>pivot else pd.Timedelta(999, "d"))
        difference = (nearest - pivot)
    
    return nearest, difference


def update_nested_dict(d, u):
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = update_nested_dict(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def update_nested_dicts(*ds, fstr=None):
    r = {}
    for d in ds:
        if isinstance(d, str) and fstr:
            try:
                d = fstr(d)
            except Exception as e:
                continue
        r = update_nested_dict(r, d)
    return r

def trygetfromdict(d, keys, default=None):
    try:
        d_ = d
        for k in keys:
            d_ = d_[k]
        return d_
    except:
        return default

def list_time_in_period(tmin, tmax, fastfreq, slowfreq, include='both'):
    if include=="left":
        return [(pd.date_range(max(p, pd.to_datetime(tmin)), min(p + pd.Timedelta(slowfreq), pd.to_datetime(tmax)),
          freq=fastfreq)[:-1]) for p in pd.date_range(tmin, tmax, freq=slowfreq).floor(slowfreq)]
    elif include == "right":
        return [(pd.date_range(max(p, pd.to_datetime(tmin)), min(p + pd.Timedelta(slowfreq), pd.to_datetime(tmax)),
          freq=fastfreq)[1:]) for p in pd.date_range(tmin, tmax, freq=slowfreq).floor(slowfreq)]
    elif include == "both":
        return [(pd.date_range(max(p, pd.to_datetime(tmin)), min(p + pd.Timedelta(slowfreq), pd.to_datetime(tmax)),
          freq=fastfreq)) for p in pd.date_range(tmin, tmax, freq=slowfreq).floor(slowfreq)]
    return


def where(arr, vl):
    '''returns index for the nearest value'''
    return np.where(arr == nearest(arr, vl)[0])[0]


def sum_nan_arrays(a, b):
    ma = np.isnan(a)
    mb = np.isnan(b)
    return np.where(ma & mb, np.nan, np.where(ma, 0, a) + np.where(mb, 0, b))


def prioritize_list(list, renames):
    match = [re.search(renames, el) for el in list]
    match = [el != None for el in match]
    match = np.array(list)[match]
    match = [list.index(el) for el in match]
    for el in match:
        list.insert(0, list.pop(el))
    return list

def replace_with_dict(ar, dic):
    # Extract out keys and values
    k = np.array(list(dic.keys()))
    v = np.array(list(dic.values()))

    # Get argsort indices
    sidx = k.argsort()
    
    # Drop the magic bomb with searchsorted to get the corresponding
    # places for a in keys (using sorter since a is not necessarily sorted).
    # Then trace it back to original order with indexing into sidx
    # Finally index into values for desired output.
    return v[sidx[np.searchsorted(k,ar,sorter=sidx)]]
    

def get_files_paths_using_regex(path, loopupitems=[], startswith=None, pattern='.*_raw_dataset_([0-9]{12}).csv', date_format='%Y%m%d%H%M'):
    loopupitems = [f.strftime(date_format) if isinstance(f, pd.Timestamp) else f for f in loopupitems]
    folder_files = {}
    found_files = {}
    for root, _, files in os.walk(path):
        for name in files:
            dateparts = re.findall(pattern, name, flags=re.IGNORECASE)
            if len(dateparts) > 0:
                if dateparts[0] in folder_files.keys():
                    folder_files[str(dateparts[0])] += [os.path.join(root, name)]
                else:
                    folder_files[str(dateparts[0])] = [os.path.join(root, name)]
    
    if loopupitems != []:
        [found_files.update({td: folder_files[td]}) for td in set(loopupitems) & folder_files.keys()]            
    elif startswith is not None:
        [found_files.update({td: folder_files[td]}) for td in folder_files.keys() if td.startswith(startswith)] 
    else: 
        found_files = folder_files
    
    return found_files

def append_multi_dataframes(list, *args, **kwargs):
    return

def insert_in_array(a, index, b=None, axis=0):
    #a = np.ones((3, 9))
    #index = [1, 3]
    axis = 1 if axis==1 else 0
    if axis==0:
        n_b = a.shape[1] + len(index)
        not_index = np.array([k for k in range(n_b) if k not in index])
        b = np.zeros((a.shape[0], n_b), dtype=a.dtype) if b is None else b
        b[:, not_index] = a
    else:
        n_b = a.shape[0] + len(index)
        not_index = np.array([k for k in range(n_b) if k not in index])
        b = np.zeros((n_b, a.shape[1]), dtype=a.dtype) if b is None else b
        b[not_index, :] = a

    return b

'''
MULTIPROCESSING
'''
MULTIPROCESSING__ = 0
del MULTIPROCESSING__

def multiprocess_framework(fc, multiprocess, loopvar, varname=None, multiappend=None, result=True, append=True, verbosity=1, fc_kw={}, **kwargs):
    raw_lst = []
    kwargs.update(fc_kw)

    if multiprocess <= 1 or multiprocess == False:
        multiprocess = False
    if not multiprocess:
        for i, e in enumerate(loopvar):
            if varname: 
                kwargs.update({varname: e})
                raw_i = {i: fc(**kwargs)}
            else:
                raw_i = {i: fc(e, **kwargs)}
            if raw_i != None:
                raw_lst += [raw_i[i]]
    
    if multiprocess:
        """Run assynchronously with a tip to reorder after"""
        pool = mp.Pool(min(multiprocess, mp.cpu_count()-1))
        callback_dict = {}

        for i, e in enumerate(loopvar):
            if varname:
                kwargs.update({varname: e})
                pool.apply_async(fc, kwds=kwargs,
                                    callback=lambda x, j=i: callback_dict.update({j: x}))
            else:
                pool.apply_async(fc, args=(e), kwds=kwargs,
                                    callback=lambda x, j=i: callback_dict.update({j: x}))
            
        pool.close()
        pool.join()

    if not result:
        return

    # put into good order
    if multiprocess:
        if verbosity: print('Put back into the good order.')
        raw_lst = sorted(callback_dict.items())
        raw_lst = list(np.array(raw_lst)[:, 1])

    if not append:
        return raw_lst

    if not multiappend:
        multiappend = multiprocess

    if verbosity: print('000 / 000, appending class.'+' '*20, end='\r')    
    raw_dat = append_class_in_list(multiappend, raw_lst, verbosity=verbosity)
    if verbosity: print('')
    return raw_dat

def append_class_in_list(npools, el_lst, verbosity=1):
    """
    Separate in multiprocess groups and then
    Run assynchronously with a tip to reorder after
    """    
    def one_loop(i_, el_lst_):
        N = len(el_lst_)
        if N==0:
            return {i_: None}
        i = 0
        for _, w in enumerate(el_lst_):
            if verbosity: print(str(i).zfill(3), '/', str(N).zfill(3), end='\r')
            if w is None:
                continue
            if i == 0:
                el_clas = w
            else:
                for k in el_clas.__dict__.keys():
                    #print(type(el_clas.__dict__[k]))
                    if (str(el_clas.__dict__[k]) == str(w.__dict__[k])) and \
                                (('col_vars' in el_clas.__dict__.keys() and (k not in el_clas.col_vars)) or
                                    ('col_vars' not in el_clas.__dict__.keys())):
                            #(el_clas.__dict__[k] is None)):
                        #print('? append!')
                        continue
                    elif isinstance(el_clas.__dict__[k], float) or isinstance(el_clas.__dict__[k], int):
                        #print('int append!')
                        continue
                    elif isinstance(el_clas.__dict__[k], str):
                        #print('str append!')
                        continue
                    elif el_clas.__dict__[k] is None:
                        #print('el_clas.__dict__[k] is None!')
                        continue
                    elif isinstance(el_clas.__dict__[k], list):
                        #print('list append!')
                        el_clas.__dict__[k] += w.__dict__[k]
                    #elif isinstance(wv_Fx.__dict__[k], np.ndarray):
                    elif isinstance(el_clas.__dict__[k], pd.DataFrame):
                        #print('dataframe append!')
                        #el_clas.__dict__[k] = el_clas.__dict__[
                        #        k].append(w.__dict__[k])
                        el_clas.__dict__[k] = pd.concat([el_clas.__dict__[
                                k], w.__dict__[k]])
                    #elif isinstance(wv_Fx.__dict__[k], np.ndarray):
                    else:
                        el_clas.__dict__[k] = np.append(
                                el_clas.__dict__[k], w.__dict__[k])
                    #wv_Fx.__dict__ = dict([(k, [wv_Fx.__dict__[k]] + [w.__dict__[k]]) for k in wv_Fx.__dict__])
            i += 1
        return {i_: el_clas}

    def cascading_loop(npools, el_lst_):
        npools = min(npools, mp.cpu_count()-1)
        pool = mp.Pool(npools)
        callback_dict = {}
        n_ = int(np.ceil(len(el_lst_)/npools))
        for i in range(npools):
            pool.apply_async(one_loop, args=(i, el_lst_[i*n_:(i+1)*n_]),
                             callback=lambda x: callback_dict.update(x))
        pool.close()
        pool.join()

        el_lst_ = sorted(callback_dict.items())
        el_lst_ = list(np.array(el_lst_)[:, 1])
        return el_lst_

    active_pools = npools
    while active_pools >= 3:
        if verbosity: print(str(active_pools).zfill(3), end='\r')
        el_lst = cascading_loop(active_pools, el_lst)
        active_pools = int(np.ceil(active_pools/2))
        
    cl_dat = one_loop(0, el_lst)
    cl_dat = cl_dat[0]
        
    return cl_dat
        

def fc_mp_along_axes(fc, npools, *args, **kwargs):
    if npools<=1:
        return(fc(*args, **kwargs))
    else:
        if False in [len(a) for a in args]:
            print('Warning: not all args have same length.')

        npools = min(npools, mp.cpu_count()-1)
        pool = mp.Pool(npools)

        callback_dict = {}
        n_ = int(np.ceil(args[0].shape[0]/npools))

        def fc_mp(MP_ID, *args, **kwargs):
            return {MP_ID: fc(*args, **kwargs)}
        
        #for i in range(len(args[0])):
        for i in range(npools):
            pool.apply_async(fc_mp, args=[i] + [a[i*n_:(i+1)*n_] for a in args], kwds=kwargs,
                             callback=lambda x: callback_dict.update(x))
        pool.close()
        pool.join()

        wrklist = sorted(callback_dict.items())
        wrklist = np.concatenate(np.array(wrklist)[:, 1])
        return wrklist


def fc_mp_varying_parameter(fc, npools, param_loc, *args, **kwargs):
    if npools <= 1:
        args = list(args)
        param = args[param_loc]
        result = []
        for p in args[param_loc]:
            args[param_loc] = p
            result = result + [fc(*args, **kwargs)]
        return np.array(result)
    else:
        args = list(args)
        param = args[param_loc]
        
        npools = min(npools, mp.cpu_count()-1)
        pool = mp.Pool(npools)

        callback_dict = {}

        def fc_mp(MP_ID, *args, **kwargs):
            return {MP_ID: fc(*args, **kwargs)}

        for i in param:
            print(i)
            args[param_loc] = i
            pool.apply_async(fc_mp, args=args, kwds=kwargs,
                             callback=lambda x: callback_dict.update(x))
        pool.close()
        pool.join()

        wrklist = sorted(callback_dict.items())
        wrklist = list(np.array(wrklist)[:, 1])
        return wrklist


"""
# Log system
def print_metadata(logfile):
    if os.path.exists(logfile):
        print(open(logfile, 'r').read())
    else:
        print('No logfile found.')


def check_metadata(vartest, valtest, logfile):
    if os.path.exists(logfile):
        log = ast.literal_eval(open(logfile, 'r').read())
        #if [valtest].isin(log[vartest]):
        if isinstance(vartest, list)==False:
            vartest = [vartest]
            valtest = [valtest]
        for i in range(len(vartest)):
            if vartest[i] in log.keys() and valtest[i] == log[vartest[i]]:
                continue
            else: 
                return False
        return True # need to check if it works
    else:
        return False


def write_metadata(varwrite, valwrite, logfile, announce=False):
    if isinstance(varwrite, list)==False:
        varwrite = [varwrite]
        valwrite = [valwrite]
        
    if os.path.exists(logfile):
        log = ast.literal_eval(open(logfile, 'r').read())
        for i in range(len(varwrite)):
            log[varwrite[i]] = valwrite[i]
    else:
        for i in range(len(varwrite)):
            log = "{'" + varwrite[i] + "': " + str(valwrite[i]) + "}"
        
    with open(logfile, 'w') as logf:
        #print(logf)
        if announce:
            logf.write(str(log))
        else:
            with suppress_stdout():
                logf.write(str(log))
"""

'''
CLASSES
'''

class datahandler():
    def __init__(self, data=None, **kwargs):
        self.data = data
        self.update(**kwargs)

    def update(self, **kwargs):
        kwargs = {k: v.to_dict() if isinstance(v, pd.DataFrame) else v for k, v in kwargs.items()}
        self.__dict__.update(kwargs)
        return

    def cols(self, cols=None, exclude=None):
        nself = type(self)(**self.__dict__)
        kcols = nself.data.columns
        if cols:
            kcols = list(set(kcols)&set(cols))
        if exclude:
            kcols = list(set(kcols)-set(exclude))
        kcols = [k for k in nself.data.columns if k in kcols]
        nself.data = nself.data[kcols]
        return nself
        
    def update(self, **kwargs):
        kwargs = {k: v.to_dict() if isinstance(v, pd.DataFrame) else v for k, v in kwargs.items()}
        self.__dict__.update(kwargs)
        return

    def vars(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith('__') and not k.endswith('__')}

    def select(self, attrs: list, fill=False):
        if fill:
            return metadata(**{k: self.vars().pop(k, None) for k in attrs})
        else:
            return metadata(**{k: self.vars().pop(k) for k in set(attrs) & set(self.vars().keys())})
        
    def print(self, limit_cha=100):
        # if len(str(item)) < limit_cha else str(item)[:limit_cha] + '|'
        print('\n'.join(str("%s: %."+str(limit_cha)+"s") % item for item in vars(self).items()))
        return

    def dump(self, path):
        with open(path, 'wb+') as file:
            pickle.dump(self, file)
        return self
    
    def load(path):
        if os.path.exists(path):
            with open(path, 'rb') as file:
                return pickle.load(file)
        else:
            return None

class metadata(datahandler):
    def __init__(self, filepath=None, **kwargs):
        self.__filepath__ = filepath
        if kwargs:
            self.update(**kwargs)
        #self.__dict__.update(kwargs)

    def update(self, **kwargs):
        kwargs = {k: v.to_dict() if isinstance(v, pd.DataFrame) else v for k, v in kwargs.items()}
        self.__dict__.update(kwargs)
        return

    def check(self, attrs=None):
        if os.path.exists(self.__filepath__):
            try:
                log = pickle.load(open(self.__filepath__, 'rb'))
            except OSError:
                raise OSError('File exists but not accessible')
            if attrs:
                
                check = metadata(**vars(self))
                return {k: vars(check).pop(k, None) for k in attrs} == {
                    k: vars(log).pop(k, None) for k in attrs}
                """
                return [vars(self)[k] == vars(log)[k] for k in attrs \
                    if k in vars(self).keys() and k in vars(log).keys()]
                """
            else:
                return vars(self) == vars(log)
        else:
            return False

    def write(self):
        pickle.dump(self, open(self.__filepath__, 'wb+'))
        return

    def print_file(self):
        if os.path.exists(self.__filepath__):
            log = pickle.load(open(self.__filepath__, 'rb'))
            print('\n'.join("%s: %s" % item for item in vars(log).items()))
        else:
            print('No logfile found.')
        return


class ECnetwork(datahandler):
    def __init__(self, path="data/info/stations.xlsx"):
        stations = pd.read_excel(path)
        stations.columns = map(str.lower, stations.columns)
        self.name = stations["name"].to_list()
        self.__dict__.update(**stations.set_index('name').to_dict())
        '''self.loc = {}
        for st in self.name:
            if np.isfinite(self.Latitude[st] + self.Longitude[st]):
                """get site location"""
                site_loc = [self.Longitude[st]] + [self.Latitude[st]]
                latlon = ops.transform(Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True).transform,
                                    geometry.Point((site_loc)))  # (y, x)
                self.loc.update({st: list(latlon.coords[0])})'''
        return
    
    def plot(self, zoom_start=4, **kwargs):
        ECmap = folium.Figure(height=300)
        ECmap = folium.Map(location=[48, 8], zoom_start=zoom_start,
                           #scrollWheelZoom=False,
                           **kwargs).add_to(ECmap)

        for st in self.name:
            if np.isfinite(self.latitude[st] + self.longitude[st]):
                folium.Marker([self.latitude[st], self.longitude[st]],
                              tooltip=st + '<br>' + '<br>'.join("%s: %s" % item for item in {k: vars(self)[k][st] for k in ['project', 'country', 'latitude', 'longitude', 'tile'] if k in vars(self).keys()}.items())).add_to(ECmap)

                #folium.Marker([site.geometry.y, site.geometry.x],
                #              tooltip=st).add_to(ECmap)
        return ECmap


class ECsite(datahandler):
    def __init__(self, SiteName, mapath="data/info/", stpath="stations.xlsx", lupath="landuse.xlsx"):
        stpath = os.path.join(mapath, stpath)
        lupath = os.path.join(mapath, lupath)

        stations = pd.read_excel(stpath)
        stations.columns = map(str.lower, stations.columns)
                
        if SiteName in stations["sitename"]:
            lookup = "sitename"

        else:#elif SiteName not in stations["name"]:
            lookup = "name"
            SiteName = prioritize_list(stations["name"].to_list(), "("+SiteName+")")[0]
        
        self.name = SiteName
        
        # screen site name
        stations = stations[(stations[lookup] == SiteName)]
        attrs = stations.set_index(lookup).to_dict()
        attrs = {el: attrs[el][SiteName] for el in attrs.keys()}
        self.__dict__.update(**attrs)

        # get site location
        site_loc = list(stations["longitude"].values) + \
            list(stations["latitude"].values)
        latlon = ops.transform(Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True).transform,
                               geometry.Point((site_loc)))  # (y, x)
        self.loc = list(latlon.coords[0])

        # get site tile
        #if pd.isnull(stations.satelitetile).any():
        #    stations.loc[pd.null(stations.satelitetile), 'satelitetile'] = add_sentinel_tile_in_dataframe(
        #        stations[pd.null(stations.satelitetile)])
        self.tile = stations.satelitetile.to_list()[0]

        if self.targetareafilepath not in [None, np.nan]:            
            if os.path.isfile(self.targetareafilepath):
                self.targetarea = gpd.read_file(
                    self.targetareafilepath, driver='KML')
                self.targetarea = self.targetarea.to_crs("EPSG:3035")
                self.targetarea = geometry.Polygon(self.targetarea.geometry[0])
            else:
                self.targetarea = 'Err: NotFound'
        else:
            self.targetarea = np.nan

        if os.path.exists(lupath):
            luinfo = pd.read_excel(lupath, index_col='CO')
            luinfo.columns = map(str.lower, luinfo.columns)
            luinfo = luinfo.to_dict()
            self.lu_path = luinfo['landusefilepath'][self.co]
            self.lu_metapath = luinfo['landusemetafilepath'][self.co]
            self.lu_resolution = luinfo['resolution'][self.co]
        
    def get(self):
        print('\n'.join("%s: %s" % item for item in vars(self).items()))

    def plot(self, **kwargs):
        if np.isfinite(self.latitude + self.longitude):
            ECmap = folium.Figure(height=300)
            ECmap = folium.Map(location=[self.latitude, self.longitude],
                               #zoom_start=8, scrollWheelZoom=False,
                               **kwargs).add_to(ECmap)

            folium.Marker([self.latitude, self.longitude],
                          tooltip='<br>'.join("%s: %s" % item for item in {k: vars(self)[k] for k in ['name', 'country', 'latitude', 'longitude', 'tile']}.items())).add_to(ECmap)
        return ECmap
