"""
Common functions
"""

from contextlib import contextmanager
import inspect
#import imp
import sys
import os
import re
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
flist = lambda x: list(itertools.chain.from_iterable(x))

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
    from tkinter import messagebox
    messagebox.showinfo('Message', message)


class LazyCallable(object):
    def __init__(self, name, fcn=None):
        self.modn, self.fc = name, None
        self.n = fcn

    def __get_jl__(self, *a, **k):
        from julia.api import Julia
        jl = Julia(compiled_modules=True)

        fc = jl.eval('include("{}"); {}'.format(self.modn, self.n))
        return fc
    
    def __get_py__(self, *a, **k):
        if self.n is None:
            self.modn, self.n = self.modn.rsplit('.', 1)
        
        if self.modn not in sys.modules:
            __import__(self.modn)
        fc = getattr(sys.modules[self.modn], self.n)
        return fc
    
    def __get_R__(self, *a, libs=[], **k):
        from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage as RPackage, importr

        """
        for l in libs:
            try:
                __rlib__ = RPackage(''.join(open(l, 'r').readlines()), '__rlib__')
            except:
                print("Warning! Library '{}' not loaded.".format(l))
        """
        
        preamble = ['library({})'.format(l) for l in libs]
        fc = RPackage(''.join(preamble + open(self.modn, 'r').readlines()), 'rpy2_fc')
        
        if not self.n:
            print("Warning! No function name given, return function with the same name as file. If not found returns first function.")
            
            fcn = [c for c in fc.__dict__['_exported_names'] if (c == self.modn.rsplit('/', 1)[1].rsplit('.', 1)[0]) and (c in fc.__dict__.keys())]
            fcn = fcn[0] if fcn else list(fc.__dict__['_exported_names'])[0] if list(fc.__dict__['_exported_names'])[0] in fc.__dict__.keys() else ''
            if fcn:
                self.n = fcn
        
        if self.n and self.n != '*':
            fc = fc.__dict__[self.n]
        
        return fc
    
    def __get__(self, *a, **k):
        if (self.fc is None):
            if (self.modn.lower().endswith('.r')) or ((self.n is not None) and (self.n.lower().endswith('.r'))):
                self.fc = self.__get_R__(*a, **k)
            elif self.modn.lower().endswith('.jl'):
                self.fc = self.__get_jl__(*a, **k)
            else:
                self.fc = self.__get_py__(*a, **k)
        return self

    def __call__(self, *a, **k):
        _g = self.__get__(*a, **k)
        
        if (self.modn.lower().endswith('.r')) or ((self.n is not None) and (self.n.lower().endswith('.r'))):
            from rpy2.robjects.conversion import rpy2py
            from rpy2.robjects import pandas2ri
            pandas2ri.activate()
            #print('r')
            #print(type(robjects.conversion.rpy2py(self.__get__().fc(*a, **k))))
            #print(self.__get__().fc(*a, **k)[0].head())
            #print(self.__get__().fc(*a, **k)[1].head())
            #print('r2')
            #print(robjects.conversion.rpy2py(self.__get__().fc(*a, **k)))
            '''
            for i, j in enumerate(a):
                if isinstance(j, pd.DataFrame) and 'TIMESTAMP' in j.columns:
                    #a[i].TIMESTAMP = j.TIMESTAMP.apply(np.datetime64)
                    print(j.dtypes)
                    a[i].TIMESTAMP = j.TIMESTAMP.as_type('%Y%m%d%H%M')
                    print(j.dtypes)
            '''
            return rpy2py(_g.fc(*a, **k))
        else:
            return _g.fc(*a, **k)

def covariance(x, y):
    # Subtracting mean from the individual elements
    sub_x = x - np.mean(x)
    sub_y = y - np.mean(y)
    cov = np.sum(sub_x*sub_y) / (len(x)-1)
    return cov

def minmax(x):
    return [np.min(x), np.max(x)]

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


def list_time_in_period(tmin, tmax, fastfreq, slowfreq):
    return [(pd.date_range(p, p + pd.Timedelta(slowfreq), freq=fastfreq, closed='left')) for p in pd.date_range(tmin, tmax, freq=slowfreq)]


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
            if len(dateparts) == 1:
                if dateparts[0] in folder_files.keys():
                    folder_files[dateparts[0]] += [os.path.join(root, name)]
                else:
                    folder_files[dateparts[0]] = [os.path.join(root, name)]
    
    if loopupitems != []:
        [found_files.update({td: folder_files[td]}) for td in set(loopupitems) & folder_files.keys()]            
    elif startswith is not None:
        [found_files.update({td: folder_files[td]}) for td in folder_files.keys() if td.startswith(startswith)] 
    else: 
        folder_files.keys()

    return found_files

def append_multi_dataframes(list, *args, **kwargs):
    return


'''
MULTIPROCESSING
'''


def multiprocess_framework(fc, multiprocess, loopvar, multiappend=None, result=True, append=True, verbosity=1, **kwargs):
    if multiprocess <= 1:
        multiprocess = False
    if not multiappend:
        multiappend = multiprocess
    
    if multiprocess:
        """Run assynchronously with a tip to reorder after"""
        pool = mp.Pool(min(multiprocess, mp.cpu_count()-1))
        callback_dict = {}

    raw_lst = []
    for i, e in enumerate(loopvar):
        if multiprocess:
            pool.apply_async(fc, args=(i, e), kwds=kwargs,
                                 callback=lambda x: callback_dict.update(x))
        else:
            raw_i = fc(i, e, **kwargs)
            if raw_i != None:
                raw_lst += [raw_i[i]]

    if multiprocess:
            pool.close()
            pool.join()

    if not result:
        return

    if multiprocess:
        if verbosity: print('Put back into the good order.')
        raw_lst = sorted(callback_dict.items())
        # [el[1] for el in bm_lst]
        raw_lst = list(np.array(raw_lst)[:, 1])

    if not append:
        return raw_lst

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
                        el_clas.__dict__[k] = el_clas.__dict__[
                                k].append(w.__dict__[k])
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

class readable_file():
    def __init__(self, __path__, **kwargs):
        self.__path__ = os.path.join(__path__, 'readme.txt') if os.path.isdir(__path__) else __path__
        self.__text__ = 'MACHINE READ FILE. PLEASE DO NOT MODIFY SPACES AND LINE JUMPS.'
        self.__iden__ = ''
        self.__dict__.update(kwargs)
        if isinstance(self.__text__, dict):
            self.__text__ = '\n'.join(["'{}': {}".format(k, v) for k, v in self.__text__.items()])
        else:
            self.__text__ = str(self.__text__)

    def dump(self, max_iteration=1000, **kwargs):
        def check(body, text = '', title=''):

            for k, v in {k: v for k, v in body.items(
            ) if isinstance(v, dict) == False}.items():
                if text[-2:] == '\n\n':
                    text = text + title + '\n'
                elif text:
                    text = text + '\n'
                add = '"{}": "{}"'.format(k, v) if isinstance(
                    v, str) else '"{}": {}'.format(k, v)
                text = text + add

            for k, v in {k: v for k, v in body.items(
            ) if isinstance(v, dict)}.items():
                if text and (text[-2:] != '\n\n'):
                    text = text + '\n\n'
                text = check(v, text, title + k + '::')
            return text

        if isinstance(self, dict):
            txt = self
        else:
            try:
                txt = self.to_dict()
            except:
                print("Nothing was done! Can only deal with dictionary and 'readable_file' Class. Try using 'readable_file(...)'.")
                return
        
        readable = self.__text__ + '\n\n'
        
        readable = readable + check(txt) + '\n\n'

        #readable = readable + '\n\n'.join([k + '::\n' + '\n'.join(['"{}": "{}"'.format(k_, v_) if isinstance(v_, str) else '"{}": {}'.format(k_, v_) for k_, v_ in v.items()]) for k, v in txt.items()])
        readable = readable + str(self.__iden__)

        with open(self.__path__, 'wb+') as f:
            f.write(readable.encode('utf-8'))
        return self

    def load_py(self, **kwargs):
        with open(self.__path__, 'r') as f:
            fil = f.read()

        function_marker = 'def (.+)\((.*)\):'
        fcs = re.findall(function_marker, fil)
        for fc in fcs:
            walking_dic = self.__dict__
            walking_dic.update({fc[0]: fc[1]})
            print(walking_dic)

        return self

    def load(self, **kwargs):
        if self.__path__.endswith('.py'):
            return self.load_py(**kwargs)
        with open(self.__path__, 'r') as f:
            fil = f.read()
        pts = fil.split('\n\n')
        self.__text__ = pts[0]
        self.__iden__ = pts[-1] if len(pts) > 2 else None
        for i, txt in enumerate(pts[1:-1]):
            tit = txt.split('\n', 1)[0].split('::')[:-1] if '::' in txt.split('\n', 1)[0] else []
            walking_dic = self.__dict__
            for t in tit[:-1]:
                if t not in walking_dic.keys():
                    walking_dic[t] = {}
                walking_dic = walking_dic[t]
            
            txt = txt.split('::\n', 1)[-1] if '::' in txt.split('\n', 1)[0] else txt
            txt = ast.literal_eval('{' +  txt.replace('\n', ', ') + '}')
            walking_dic.update({tit[-1]: txt} if tit else txt)
        return self

    def check_id(self, id, **kwargs):
        check = self.load(**kwargs)
        return str(check.__iden__) == str(id)

    def print(self):
        print('\n'.join("%s: %s" % item for item in vars(self).items()))
        return

    def to_dict(self, ix=0):
        #{k_: v_ for k_, v_ in v.items()} if isinstance(v, dict) else
        return {k: v for k, v in self.__dict__.items() if not k.startswith('__') and not k.endswith('__')}
    

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
    def __init__(self):
        stations = pd.read_excel(
            "data/info/stations.xlsx")
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
    def __init__(self, SiteName):
        stations = pd.read_excel(
            "data/info/stations.xlsx")
        stations.columns = map(str.lower, stations.columns)
                
        if SiteName in stations["sitename"]:
            lookup = "sitename"

        elif SiteName not in stations["name"]:
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
        if pd.isnull(stations.satelitetile).any():
            stations.loc[pd.null(stations.satelitetile), 'satelitetile'] = add_sentinel_tile_in_dataframe(
                stations[pd.null(stations.satelitetile)])
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
        
        luinfo = pd.read_excel(
            "data/info/landuse.xlsx", index_col='CO')
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
