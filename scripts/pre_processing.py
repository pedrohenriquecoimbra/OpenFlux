"""
Functions to retrieve Bioclimatology data from sites and put in a standard format.
"""

from functools import reduce
import os
import ast
import warnings
import copy
import numpy as np
import pandas as pd
from Lib.OpenFlux.scripts.corrections import tilt_rotation, despike, time_lag
from Lib.OpenFlux.scripts import get_data, get_rawdata, common as tt
import pathlib
cfp = pathlib.Path(__file__).parent.resolve()
#checkifinprogress = tt.LazyCallable(os.path.abspath(os.path.join(cfp, "..", "main.py")), "checkifinprogress").__get__().fc

import importlib
importlib.reload(tt)
importlib.reload(get_data)
importlib.reload(tilt_rotation)

PRE_CALC_DATES_LEVEL = {}

def default_lag(length=711, diameter=5.3, pump=15, dt=20):
    # default time lag in number of data points
    return int(np.round((length * (np.pi * (diameter/2)**2) * (10**-6) / pump) * 60 * dt))

def not_ignored_files(list_date, path_output, file_name, steps="", dateformat='%Y%m%d%H%M', found=False, verbosity=0):
    list_new_date = []
    
    for file_date in list_date:
        notskip = True
        desc_date = min(file_date).strftime(dateformat)+'_'+max(file_date).strftime(dateformat) if isinstance(file_date, (list, np.ndarray, pd.DatetimeIndex)) else file_date.strftime(dateformat)
        if verbosity==1: print("Screening: ", desc_date, ' '*20, end='\r')
        
        for f in [steps]:#_, f in enumerate(reversed(steps)):
            if path_output is not None and file_name is not None:
                path_output_full = os.path.join(path_output, f, file_name.format(desc_date))

                if verbosity==2: print(path_output_full, ' '*10, end='\r')

                #if default_ignore is False and os.path.isfile(path_output_full): 
                #    default_ignore = True

                # if save and overwrite: ignore = False
                if found:
                    if (notskip) and (os.path.isfile(path_output_full)): #default_ignore == False
                        list_new_date += [file_date]
                        notskip = False
                else:
                    if (notskip) and (os.path.isfile(path_output_full)==False): #default_ignore == False
                        list_new_date += [file_date]
                        notskip = False
            else:
                print('Warning! Running No file name nor path were given, no search was attempted.')
            
    return list_new_date

"""
def callinloop(*a, **k):
    return universalCaller.call_loop(*a, **k)

class universalCaller(tt.datahandler):
    def __init__(self, *args, **kwargs):
        "pre processing"
        args = list(args)
        # print(args[0], isinstance(args[0], get_rawdata.FluxTowerRawData))
        if args:  # and isinstance(args[0], get_rawdata.FluxTowerRawData):
            self.__dict__.update(**args[0].__dict__)
            args.pop(0)
            self.get(*args, **kwargs)
        self.__dict__.update(**kwargs)
"""
def first_call(file_date, **kwargs):
    fs = DEFAULT_FIRST_STEP
    fs.update(kwargs)
    fc = fs["func"]
    kw = fs["kwargs"]
    try:
        if not callable(fc):
            fc = tt.LazyCallable(*fc).__get__().fc
        kw[fs['date_var']] = file_date if isinstance(
                    file_date, (list, np.ndarray, pd.DatetimeIndex)) else [file_date]
        current_dta = fc(**kw)
    except Exception as e:
        warnings.warn(str(e))
    return current_dta


def steps_call(cordta, file_date, func, outputpath, fckw={}, overwrite=False, save=False, _rerun_args=None, **kwargs):
    if 'buffer_period' in kwargs.keys():
        _ = kwargs.pop('avg_period', None)

    if not callable(func):
        func = tt.LazyCallable(*func).__get__().fc

    if not callable(func):
        warnings.warn(
            'Function named {} not callable, thus not run.'.format(str(func)))
        return None
    
    # IF PRE-FUNCTION
    if 'avg_period' in kwargs.keys():
        if func.__name__ in PRE_CALC_DATES_LEVEL.keys() and (file_date >= min(PRE_CALC_DATES_LEVEL[func.__name__])) and (file_date <= max(PRE_CALC_DATES_LEVEL[func.__name__])):
            # skip
            return None

        __date_range = (min(tt.flist(kwargs['avg_period'])), max(
            tt.flist(kwargs['avg_period'])))
        assert (file_date >= __date_range[0]) and (file_date <= __date_range[1]), \
            'Date outside of averaging period. Date is {} and averaging period range is {}.'.format(file_date, __date_range)

        # select dates to run step in which the running date is in
        _ymd = pd.DatetimeIndex([p for p in kwargs['avg_period'] if file_date in p][0])
        globals()['PRE_CALC_DATES_LEVEL'].update({func.__name__: _ymd})

        #print(_rerun_args.keys())
        _rerun_args.update({'ymd': _ymd, 'verbosity': 0})
        _tmpdta = call_loop(**_rerun_args)
        #_tmpdta = tt.multiprocess_framework(
        #    universalcall, multiprocess=1, **_rerun_args).data

        fckw.update(pre_calculus=True, save_date=_ymd)
        _ = func(_tmpdta, **fckw)

        fckw.update(pre_calculus=False)
        del _tmpdta, _rerun_args

    if 'buffer_period' in kwargs.keys():
        if isinstance(kwargs['buffer_period'], str):
            kwargs['buffer_period'] = pd.Timedelta(
                kwargs['buffer_period'])
        # buffer means that for a normal run it will use border values (to avoid missing when shifting data)
        _ymd = pd.DatetimeIndex((file_date-kwargs['buffer_period'],
                    file_date, file_date+kwargs['buffer_period']))
        
        _rerun_args.update({'ymd': _ymd, 'verbosity': 0, 'save': False, 'result': True})
        #print(_rerun_args.keys())
        _tmpdta = call_loop(**_rerun_args)
        #tt.multiprocess_framework(universalcall, multiprocess=1, **_rerun_args).data
        
        _tmpdta = func(_tmpdta, **fckw)

        cordta = _tmpdta[np.isin(pd.to_datetime(_tmpdta.TIMESTAMP, errors='coerce'), pd.to_datetime(
            cordta.TIMESTAMP, errors='coerce'))].copy()
        #cordta = _tmpdta[(pd.to_datetime(_tmpdta.TIMESTAMP) >= pd.to_datetime(file_date)) * \
        #(pd.to_datetime(_tmpdta.TIMESTAMP) < pd.to_datetime(file_date + kwargs['buffer_period']))].copy()
        del _tmpdta, _rerun_args

    else:
        cordta = func(cordta, **fckw)

    if cordta.empty:
        warnings.warn(
            'Warning! Dataframe is empty ({}).'.format(func.__name__))
        return None
    
    if save:
        # Attempt to save
        if cordta.empty:
            warnings.warn('Warning! Dataframe is empty no saving was attempted ({}).'.format(func.__name__))
            return None
        elif outputpath is not None:
            if (not os.path.isfile(outputpath)) or (os.path.isfile(outputpath) and overwrite):
                #if not os.path.exists(os.path.dirname(outputpath)):
                #    os.makedirs(os.path.dirname(outputpath), exist_ok=True)
                tt.mkdirs(outputpath)
                cordta.to_csv(outputpath, index=False)
            else:
                warnings.warn('Warning! ?')
        else:
            warnings.warn('Warning! No file name nor path were given or found, saving was not possible.')
    return cordta

def universalcall(file_date, path_output=None, file_name=None, 
            first_step={}, steps_order=[], steps=None, last_step={}, 
            from_raw=False, verbos=0, result=True, naive=True, **kwargs):
    if verbos: print(file_date, ' '*1, end='\r')
    curoutpath_inprog = ""
    
    # Get default steps if None
    if steps is None: steps = DEFAULT_STEPS
    
    # Put order into steps
    if isinstance(steps, dict):
        steps = [steps[s] for s in steps_order] if steps_order else list(steps.values())

    #fs_ = DEFAULT_FIRST_STEP
    #fs_.update(first_step)
    #first_step = fs_
    #if first_step is None: first_step = DEFAULT_FIRST_STEP
            
    saved_args = {k: v for k, v in locals().items()} #copy.deepcopy()
    
    
    if path_output is not None and file_name is not None:
        if isinstance(file_date, (list, np.ndarray, pd.DatetimeIndex)):
            path_name = file_name.format(min(file_date).strftime('%Y%m%d%H%M')+'_'+max(file_date).strftime('%Y%m%d%H%M'))
        else:
            path_name = file_name.format(file_date.strftime('%Y%m%d%H%M'))
        
        curoutpath_inprog = os.path.join(path_output, path_name.rsplit(".", 1)[0] + ".inprogress")
        if not sum([tt.trygetfromdict(s_, ['buffer_period'], False) for s_ in steps]):
            if tt.checkifinprogress(curoutpath_inprog):
                return

    # READ STEPS IN REVERSE TO ADD PATH AND CHECK LAST AVAILABLE
    steps_r = []
    default_ignore = False
    if not from_raw:
        path_to_load = None
        for i, s in enumerate(reversed(steps)):
            if path_output is not None and file_name is not None:
                #if isinstance(file_date, (list, np.ndarray, pd.DatetimeIndex)):
                #    path_name = file_name.format(min(file_date).strftime('%Y%m%d%H%M')+'_'+max(file_date).strftime('%Y%m%d%H%M'))
                #else:
                #    path_name = file_name.format(file_date.strftime('%Y%m%d%H%M'))
                path_output_full = os.path.join(path_output, s['name'], path_name)
                
                if len(steps)==1 and s['save']:
                    if os.path.exists(curoutpath_inprog): os.remove(curoutpath_inprog)
                    curoutpath_inprog = path_output_full.rsplit(".", 1)[0] + ".inprogress"
                    if tt.checkifinprogress(curoutpath_inprog):
                        return

                if default_ignore is False and os.path.isfile(path_output_full): 
                    default_ignore = True
                    path_to_load = path_output_full
                    
                # if save and overwrite: ignore = False
                steps_r = steps_r + [{'name': s['name'], 'path': path_output_full, 'available': os.path.isfile(path_output_full), 'ignore': default_ignore}]
            else:
                print('Warning! Running No file name nor path were given, no loading was attempted.')
    else:
        steps_r = [{'name': s['name'], 'path': path_output + s['name'] + '/' + path_name, 
                    'available': False, 'ignore': False} for s in reversed(steps)]

    if sum([s['ignore']==0 for s in steps_r]) or True:#result==False:
        if path_to_load is not None:
            cordta = pd.read_csv(path_to_load)

        if 'cordta' not in locals():
            # move it to -> universalCaller._first_call(file_date, **first_step)
            try:
                print(file_date, first_step)
                cordta = first_call(file_date=file_date, **first_step)
                """
                if not callable(first_step['fc']):
                    first_step['fc'] = tt.LazyCallable(*first_step['fc']).__get__().fc
                first_step['kwargs'][first_step['date_var']] = file_date if isinstance(file_date, (list, np.ndarray, pd.DatetimeIndex)) else [file_date]
                cordta = first_step['fc'](**first_step['kwargs'])
                """
            except Exception as e:
                cordta = pd.DataFrame()
                warnings.warn(str(e))
                """if isinstance(file_date, (list, np.ndarray, pd.DatetimeIndex)):
                    cordta = get_rawdata.open_flux(lookup=file_date, **first_step['kwargs']).data
                else:
                    cordta = get_rawdata.open_flux(lookup=[file_date], **first_step['kwargs']).data"""
        
        if cordta is None or cordta.empty:
            #print('empt')
            warnings.warn(f'{file_date} (Empty)')
            if os.path.exists(curoutpath_inprog): os.remove(curoutpath_inprog)
            return pd.DataFrame() #if i_ is None else {i_: tt.datahandler(pd.DataFrame())}
        else:
            None
            #print('full')
            #print(file_date, ' '*15, end='\n')
        
        # check if attributed time (in file path) is in fact data time (in file), 
        # if data came with no TIMESTAMP, comparison is spurious

        # file_date in cordta.TIMESTAMP
        # else segment data frame in pieces

        # REVERSE BACK TO NORMAL ORDER
        steps_r = [c for c in reversed(steps_r)]

        # RUN STEPS
        for i, f in enumerate(steps):
            #print(i, f['fc'])
            #print(file_date, cordta.columns)
            #print(cordta[["u", "v", "w", "co2", "h2o"]].head(2))
            #print("___")
            if steps_r[i]['ignore'] or 'skip' in f.keys() and f['skip']:
                # skip
                continue
            
            _rerun_args = copy.deepcopy(saved_args)
            #_rerun_args.pop('i_')
            _rerun_args.pop('file_date')
            _rerun_args['steps'] = _rerun_args['steps'][:i]
            
            fc_ = f.pop('fc', None)
            if fc_:
                f.update({"func": fc_})
            fc_ = f.pop('kwargs', None)
            if fc_:
                f.update({"fckw": fc_})
            cordta = steps_call(cordta, file_date, **f, outputpath=steps_r[i]["path"], _rerun_args=_rerun_args)
            if cordta is None: return pd.DataFrame()

            """
            if 'kwargs' not in f.keys():
                f['kwargs'] = {}

            if 'buffer_period' in f['kwargs'].keys():
                _ = f['kwargs'].pop('avg_period', None)
            
            if not callable(f['fc']):
                f['fc'] = tt.LazyCallable(*f['fc']).__get__().fc
            
            if not callable(f['fc']):
                warnings.warn('Function named {} not callable, thus not run.'.format(str(f)))
                continue
            else:
                # IF PRE-FUNCTION         
                if 'avg_period' in f['kwargs'].keys():
                    if f['name'] in PRE_CALC_DATES_LEVEL.keys() and (file_date >= min(PRE_CALC_DATES_LEVEL[f['name']])) and (file_date <= max(PRE_CALC_DATES_LEVEL[f['name']])):
                        # skip
                        continue

                    __date_range = (min(tt.flist(f['kwargs']['avg_period'])), max(tt.flist(f['kwargs']['avg_period'])))
                    assert (file_date >= __date_range[0]) and (file_date <= __date_range[1]), \
                    'Date outside of averaging period. Date is {} and averaging period range is {}.'.format(file_date, __date_range)

                    # select dates to run step in which the running date is in
                    _loopvar = [p for p in f['kwargs']['avg_period'] if file_date in p][0]
                    globals()['PRE_CALC_DATES_LEVEL'].update({f['name']: _loopvar})

                    _rerun_args = copy.deepcopy(saved_args)
                    _rerun_args.pop('i_')
                    _rerun_args.pop('file_date')
                    _rerun_args['steps'] = _rerun_args['steps'][:i]
                    _rerun_args.update({'loopvar':_loopvar, 'verbosity': 0})
                    #print(_rerun_args.keys())
                    _tmpdta = tt.multiprocess_framework(universalcall, multiprocess=1, **_rerun_args).data

                    f['kwargs'].update(pre_calculus=True, save_date=_loopvar)
                    _ = f['fc'](_tmpdta, **f['kwargs'])

                    f['kwargs'].update(pre_calculus=False)
                    del _tmpdta, _rerun_args          
                
                if 'buffer_period' in f['kwargs'].keys():
                    if isinstance(f['kwargs']['buffer_period'], str):
                        f['kwargs']['buffer_period'] = pd.Timedelta(f['kwargs']['buffer_period'])
                    # buffer means that for a normal run it will use border values (to avoid missing when shifting data)
                    _loopvar = (file_date-f['kwargs']['buffer_period'], file_date, file_date+f['kwargs']['buffer_period'])
                    
                    _rerun_args = {k: v for k, v in saved_args.items()}
                    _rerun_args.pop('i_')
                    _rerun_args.pop('file_date')
                    _rerun_args['steps'] = _rerun_args['steps'][:i]
                    _rerun_args.update({'loopvar': _loopvar, 'verbosity': 0, 'save': False})
                    #print(_rerun_args.keys())
                    _tmpdta = tt.multiprocess_framework(universalcall, multiprocess=1, **_rerun_args).data
                    
                    _tmpdta = f['fc'](_tmpdta, **f['kwargs'])
                    
                    cordta = _tmpdta[np.isin(pd.to_datetime(_tmpdta.TIMESTAMP, errors='coerce'), pd.to_datetime(cordta.TIMESTAMP, errors='coerce'))].copy()
                    #cordta = _tmpdta[(pd.to_datetime(_tmpdta.TIMESTAMP) >= pd.to_datetime(file_date)) * \
                    #(pd.to_datetime(_tmpdta.TIMESTAMP) < pd.to_datetime(file_date + f['kwargs']['buffer_period']))].copy()
                    del _tmpdta, _rerun_args
                    
                else:
                    cordta = f['fc'](cordta, **f['kwargs'])
                    
            if cordta.empty:
                warnings.warn('Warning! Dataframe is empty ({}).'.format(f['name']))
                return None
            
            f.update(f['kwargs'])
            if 'save' in f.keys() and f['save']:
                # Attempt to save
                if cordta.empty:
                    warnings.warn('Warning! Dataframe is empty no saving was attempted ({}).'.format(f['name']))
                    return None
                elif steps_r[i]['path'] is not None:
                    if (not os.path.isfile(steps_r[i]['path'])) or (os.path.isfile(steps_r[i]['path']) and 'overwrite' in f.keys() and f['overwrite']):
                        if not os.path.exists(os.path.dirname(steps_r[i]['path'])):
                            os.makedirs(os.path.dirname(steps_r[i]['path']), exist_ok=True)
                        cordta.to_csv(steps_r[i]['path'], index=False, compression="gzip")
                    else:
                        warnings.warn('Warning! ?')
                else:
                    warnings.warn('Warning! No file name nor path were given or found, saving was not possible.')
            """
    else:
        warnings.warn('already exists.')

    if last_step:
        try:
            if not callable(last_step.get('fc', '')):
                last_step['fc'] = tt.LazyCallable(*last_step['fc']).__get__().fc
            last_step['fc'](**last_step['kwargs'])
        except Exception as e:
            if os.path.exists(curoutpath_inprog): os.remove(curoutpath_inprog)
            warnings.warn(str(e))
        
    if 'cordta' in locals():
        if os.path.exists(curoutpath_inprog): os.remove(curoutpath_inprog)
        return cordta #if i_ is None else {i_: tt.datahandler(cordta)}
    else:
        if os.path.exists(curoutpath_inprog): os.remove(curoutpath_inprog)
        return
    
def call_loop(ymd, multiprocess=False, verbosity=0, bystep=False, **kwargs):
    print("\nCALL LOOP")
    
    if isinstance(ymd, pd.DatetimeIndex):
        None
    elif isinstance(ymd, (list, tuple)) and len(ymd) == 3:
        s, e, f = ymd
        ymd = pd.date_range(s, e, freq=f)

    #kwargs.update({"result": False})
    #_ = tt.multiprocess_framework(universalcall, multiprocess=multiprocess,
    #                                loopvar=ymd, verbosity=verbosity, **kwargs)
    so = kwargs.pop("steps_order", [])
    if bystep:
        so = so if so else [""]
        r = [[universalcall(y, steps_order=[s] if s else [], **kwargs) for y in ymd] for s in so][-1]
    else:
        r = [universalcall(y, steps_order=so, **kwargs) for y in ymd]

    return pd.concat(r)

"""
    def _openinloop(*args, **kwargs):
        #Prepare rawdata and run get
        def o_loop(i_, file_date, path_output=None, file_name=None, steps=DEFAULT_STEPS, level=1, fcs=[], overwrite=False, save=True, from_raw=False, raw_kwargs={}, fmt_kwargs={}, **kwargs):
            if save == False: overwrite = False
            saved_args = copy.deepcopy(locals())
            #saved_args.update(**saved_args.pop('kwargs'))
            assert level > 0, 'Level chosen: {}. Negative level not allowed.'.format(level)
            level_init = 0
            
            # OPEN INITIAL DATA
            path_level_x = None
            if not from_raw and path_output is not None and file_name is not None:
                if isinstance(file_date, (list, np.ndarray, pd.DatetimeIndex)):
                    file_name = file_name.format(min(file_date).strftime('%Y%m%d%H%M')+'_'+max(file_date).strftime('%Y%m%d%H%M'))
                else:
                    file_name = file_name.format(file_date.strftime('%Y%m%d%H%M'))

                path_output = path_output + 'level_{}/' + file_name
                #path_availab = [path_output + 'level_' + str(l+1) + '/' + file_name for l in range(level)]
                #path_availab = {i: {'path': p, 'available': os.path.isfile(p)} for i, p in enumerate(path_availab)}
                #print(level, path_output, file_name)
                path_availab = {l: {'path': path_output.format(str(l+1)), 'available': os.path.isfile(path_output.format(str(l+1)))} for l in range(level)}
                
                path_level_x = path_output.format(str(level)) if fcs == [] else path_output.format('x')

                if os.path.isfile(path_level_x) and overwrite == False:
                    # open corrected file and return it
                    cordta = pd.read_csv(path_level_x)
                    return cordta if i_ is None else {i_: tt.datahandler(cordta)}
                
                elif np.sum([v['available'] for v in path_availab.values()]) > 0:
                    # open last available level
                    antecipated_load = [(l, path_availab[l]['path']) for l in reversed(range(level-1)) if path_availab[l]['available']][0]
                    cordta = pd.read_csv(antecipated_load[1])
                    level_init = antecipated_load[0]
            
            else:
                print('Warning! Running No file name nor path were given, no loading was attempted.')
            
            if 'cordta' not in locals():
                if isinstance(file_date, (list, np.ndarray, pd.DatetimeIndex)):
                    cordta = get_rawdata.icos(file_list=file_date, **raw_kwargs).format(**fmt_kwargs).data
                else:
                    cordta = get_rawdata.icos(file_list=[file_date], **raw_kwargs).format(**fmt_kwargs).data
            
            if cordta.empty:
                return None
            #print(cordta.head())
            # SELECT FUNCTIONS TO RUN
            fkw = []
            
            ## TRANSLATE LEVEL INTO A FUNCTION
            if fcs: #isinstance(level, list) and (isinstance(el, str) for el in level):
                # assert (isinstance(el, (str, list, tuple, set)) for el in fcs), 'Warning! Functions type not allowed.'
                fkw = [{} for _ in fcs]
                for i, f in enumerate(fcs):
                    fkw[i] = kwargs[f.rsplit('.', 1)[-1]] if str(f.rsplit('.', 1)[-1]) in kwargs.keys() else fkw[i]
                    fkw[i] = kwargs[f.rsplit('/', 1)[-1]] if str(f.rsplit('/', 1)[-1]) in kwargs.keys() else fkw[i]
                    fkw[i] = kwargs[f.rsplit('/', 1)[-1].rsplit('.', 1)[0]] if str(f.rsplit('/', 1)[-1].rsplit('.', 1)[0]) in kwargs.keys() else fkw[i]
                fcs = [tt.LazyCallable(f).__call__ if isinstance(f, str) else tt.LazyCallable(f[0], module=f[1]).__call__ if isinstance(f, (str, list, tuple, set)) else f for f in fcs]
            
            if isinstance(level, (int, float)) and level > 1:
                '''
                if level == 1:
                    cordta = get_rawdata.icos(file_list=[file_date], **raw_kwargs).format(**fmt_kwargs).data
                
                else:
                    # open level_x-1
                    saved_args.update({'i_': None, 'level':level-1, 'overwrite':False, 'save':False})
                    cordta = o_loop(**saved_args)
                    
                    if i_ is not None and cordta.empty:
                        return None
                    
                    # calculate level_x
                    saved_args.pop('i_')
                    LEVELS['level_' + str(level)](cordta, **saved_args)
                '''
                fkk = {k.upper(): {s: saved_args[s] for s in LEVELS_REQ[k.lower()]} for k in kwargs.keys() if k.lower() in LEVELS_REQ.keys()}
                fkk = {k.upper(): {**fkk[k.upper()], **v} for k, v in kwargs.items() if k.upper() in fkk.keys()}
                #[fkk['LEVEL_' + str(l+1)].update(fkk['LEVEL_' + str(l+1)]) for l in range(1, level) if 'LEVEL_' + str(l+1) in fkk.keys()]
                fkw = [fkk['LEVEL_' + str(l)] if 'LEVEL_' + str(l) in fkk.keys() else {} for l in range(2, level+1)] + fkw

                fcs = [LEVELS['level_' + str(l)] for l in range(2, level+1)] + list(fcs)
            
            for i, f in enumerate(fcs):
                if i < level_init:
                    continue
                                
                # IF PRE-FUNCTION
                if 'avg_period' in fkw[i].keys():
                    if str(level) in PRE_CALC_DATES_LEVEL.keys() and (file_date >= min(PRE_CALC_DATES_LEVEL[str(level)])) and (file_date <= max(PRE_CALC_DATES_LEVEL[str(level)])):
                        # skip
                        continue

                    __date_range = (min(tt.flist(fkw[i]['avg_period'])), max(tt.flist(fkw[i]['avg_period'])))
                    assert (file_date >= __date_range[0]) and (file_date <= __date_range[1]), \
                    'Date outside of averaging period. Date is {} and averaging period range is {}.'.format(file_date, __date_range)
                    _loopvar = [p for p in fkw[i]['avg_period'] if file_date in p][0]
                    globals()['PRE_CALC_DATES_LEVEL'].update({str(level): _loopvar})
                    _rerun_args = {k: v for k, v in saved_args.items()}
                    _rerun_args.pop('i_')
                    _rerun_args.pop('file_date')
                    _rerun_args.update({'loopvar':_loopvar, 'level':level-1, 'verbosity': 0, 'save': False})
                    #print(_rerun_args.keys())
                    _tmpdta = PPDataset.openinloop(multiprocess=1, **_rerun_args).data
                    fkw[i].update(pre_calculus=True, save_date=_loopvar)
                    _ = f(_tmpdta, **fkw[i])
                    fkw[i].update(pre_calculus=False)
                    del _tmpdta, _rerun_args
                
                if not callable(f):
                    print('Function named {} not callable, thus not run.'.format(str(f)))
                else:
                    cordta = f(cordta, **fkw[i])
            
            #else:
            #    print('Warning! Nothing was done, cannot handle level choice.')
            
            if save:
                # Attempt to save
                if path_level_x is not None and cordta.empty==False:
                    if (not os.path.isfile(path_level_x)) or (os.path.isfile(path_level_x) and overwrite):
                        if not os.path.exists(os.path.dirname(path_level_x)):
                            os.makedirs(os.path.dirname(path_level_x), exist_ok=True)
                        cordta.to_csv(path_level_x, index=False)
                else:
                    print('Warning! No file name nor path were given or found, saving was not possible.')
                
            return cordta if i_ is None else {i_: tt.datahandler(cordta)}

        return tt.multiprocess_framework(o_loop, *args, **kwargs)
        """
"""
def level_2(self, cols=['u', 'v', 'w', 'co2', 'h2o'], script='Py', fup2=.1, *args, **kwargs):
    #ncol = tt.multiprocess_framework(lambda i, x: {i: tt.LazyCallable("despiking", "scripts/RFlux-scripts/despiking.R").__call__(
    #        self[x], mfreq=20, variant="v3", wsignal=20*60*30/6, wscale=20*60*30/6)[0]}, loopvar=cols, append=False)
    # self[c] = ncol[i]
    plausibility_range = {'u': 3.5, 'v': 3.5, 'w': 5, 'co2': 3.5, 'h2o': 3.5}
    for c in cols:
        #import re
        #print(c, self[c][[re.match('^[0-9\.-]*$', str(t)) == None for t in self[c]]])
        if script=='RFlux':
            self.loc[:, c] = tt.LazyCallable(os.path.join(cfp, "RFlux-scripts/despiking.R"), "despiking").__call__(
                self.loc[:, c], mfreq=20, variant="v3", wsignal=7, wscale=20*60*30/6, zth=plausibility_range[c])[0] #20*60*30/6
        elif script=='OCE':
            self.loc[:, c] = tt.LazyCallable(os.path.join(cfp, "corrections/oce_despike.R"), 'despike').__call__(self.loc[:, c], n=plausibility_range[c])
        else:
            # Mauder et al. 2013
            self.loc[:, c] = tt.LazyCallable(os.path.join(cfp, "corrections/despiking.py"), "despike").__call__(self.loc[:, c], **kwargs)
    
        #if (np.sum(np.isnan(self[c])) / len(self)) <= fup2:
        #    self.loc[:, c] = self.set_index('TIMESTAMP').loc[:, c].ffill().loc[:, c]

    return self

def level_3(self, *args, **kwargs):
    return self

def level_4(self, *args, **kwargs):
    return self

def level_5(self, save_date=None, pre_calculus=False, *args, **kwargs):
    #assert 'tilt_kwargs' in kwargs.keys(), 'Assign options for tilt correction.'
    if 'tilt_kwargs' in kwargs.keys():
        kwargs.update(kwargs['tilt_kwargs'])
    
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
            '{}_{}'.format(min(save_date).strftime('%Y%m%d%H%M'), max(save_date).strftime('%Y%m%d%H%M')) 
                if (isinstance(save_date, (list, np.ndarray, pd.DatetimeIndex))) 
                else save_date.strftime('%Y%m%d%H%M'))
        #save_date.strftime('%Y%m%d%H%M'))
        
        if (os.path.exists(save_path)==False) or (
            ('__iden__' in kwargs.keys()) and (not tt.readable_file(save_path).check_id(kwargs['__iden__']))):
            if not pre_calculus:
                print('Warning! Using opened dates ({}) for calculating tilt as well as returning corrected data.'.format(
                            '{} to {}'.format(min(save_date).strftime('%Y-%m-%d %H:%M'), max(save_date).strftime('%Y-%m-%d %H:%M')) 
                            if (isinstance(save_date, (list, np.ndarray))) and (len(save_date) > 1) 
                            else save_date.strftime('%Y-%m-%d %H:%M')
                ))
            
            _, _, _, _theta, _phi = tiltcorrection.tiltcorrection(self['u'], self['v'], self['w'], method=kwargs['method'])

            '''
            if len(loopvar)==1:
                _, _, _, _theta, _phi = tiltcorrection.tiltcorrection(self['u'], self['v'], self['w'], method=kwargs['method'])
            else:
                _tmp = PPDataset.openinloop(1, loopvar, verbosity=0, **kwargs).data
                #self['u'], self['v'], self['w'] = tiltcorrection.tiltcorrection(self['u'], self['v'], self['w'], method='2r')
                _, _, _, _theta, _phi = tiltcorrection.tiltcorrection(_tmp['u'], _tmp['v'], _tmp['w'], method=kwargs['method'])
            '''
            if kwargs['setup_name']:
                tt.readable_file(save_path, angles={'theta': _theta, 'phi':_phi},
                **{k: v for k, v in kwargs.items() if k in ['__text__', '__iden__']}).dump()
            
            if pre_calculus:
                return
        
        tilt_setup = tt.readable_file(save_path).load().to_dict()['angles']
        _theta=tilt_setup['theta']
        _phi=tilt_setup['phi']
    else:
        _, _, _, _theta, _phi = tiltcorrection.tiltcorrection(self['u'], self['v'], self['w'], method=kwargs['method'])

        
    self.loc[:, 'u'], self.loc[:, 'v'], self.loc[:, 'w'], _, _ = tiltcorrection.tiltcorrection(
            self.loc[:, 'u'], self.loc[:, 'v'], self.loc[:, 'w'], method=kwargs['method'], _theta=_theta, _phi=_phi)
    
    return self

def level_6(self, default=0, *args, **kwargs):
    default = default if default > 0  else default_lag()
    #"scripts/RFlux-scripts/tlag_detection.R"
    self.loc[:, 'co2'] = self.co2.shift(-default)
    #self.loc[:, 'co2_conc'] = self.co2_conc.shift(-default)

    f = tuple([(np.isnan(self.co2)==False) * (np.isnan(self.w)==False)])
    x = np.array(self.loc[f, 'co2'])
    y = np.array(self.loc[f, 'w'])

    try:
        tlag_opt = tt.LazyCallable(os.path.join(cfp, 'RFlux-scripts/tlag_detection.R'), 'tlag_detection').__call__(
                x, y, mfreq=20)
        tlag_opt = int(tlag_opt[3][0])
        self.loc[:, 'co2'] = self.co2.shift(-tlag_opt)
        #self.loc[:, 'co2_conc'] = self.co2_conc.shift(-tlag_opt)
    except:
        tlag_opt = default
        print('err')
    
    '''
    if self.ylag > 0:
        # ignore first timestamp since we jumped 5 observations in the beginning
        flux.wvlst = flux.wvlst[:, (period-self.ylag):]
        # last part depends on the time delay for max corr between co2 and w
        self.t = self.t[(period-self.ylag):]
    '''
    return self

def level_7(self, *args, **kwargs):
    return self

def level_personalized(self, *args, **kwargs):
    '''
    self should be pd.DataFrame type.
    use args and kwargs at will.
    '''
    return self

def set_default(**kwargs):
    globals().update(kwargs)
    return

#LEVELS = {'level_2': level_2,'level_3': level_3, 'level_4': level_4, 'level_5': level_5, 'level_6': level_6, 'level_7': level_7}
#LEVELS_REQ = {'level_2': [],'level_3': [], 'level_4': [], 'level_5': [], 'level_6': [], 'level_7': []}
"""

#def DEFAULT_IMPORT(*args, **kwargs):
#    return get_rawdata.open_flux(*args, **kwargs).data

DEFAULT_FIRST_STEP = {
    'func': lambda *a, **kw: get_rawdata.open_flux(*a, **kw).data,
    'date_var': 'lookup'
}

DEFAULT_STEPS = [{
    'name': 'level_2',
    'ds': 'despike',
    'fc': despike,
    'skip': False,
    'kwargs': {'fill': pd.Series.interpolate},
},{
    'name': 'level_3',
    'ds': '-',
    'fc': lambda *args, **kwargs: args[0],
    'skip': False,
    'kwargs': {},
},{
    'name': 'level_4',
    'ds': '-',
    'fc': lambda *args, **kwargs: args[0],
    'skip': False,
    'kwargs': {},
},{
    'name': 'level_5',
    'ds': 'tilt axis',
    'fc': tilt_rotation,
    'skip': False,
    'kwargs': {'method': '2r'},
},{
    'name': 'level_6',
    'ds': 'time lag',
    'fc': time_lag,
    'skip': False,
    #'kwargs': {'buffer_period': pd.Timedelta('30Min')},
},{
    'name': 'level_7',
    'ds': '-',
    'fc': lambda *args, **kwargs: args[0],
    'skip': False,
    'kwargs': {},
},
]
