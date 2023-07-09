"""
ADD FLAGS
"""

#from .import_all import meteo_functions
from Lib.OpenFlux.scripts import common as tt, get_data, get_rawdata, QA_QC
import numpy as np
import pandas as pd
import copy
import warnings
from functools import reduce
import os
import pathlib
import yaml
cfp = pathlib.Path(__file__).parent.resolve()


def unify_dataset(ymd, path, prefix, output_path, averaging=[30], overwrite=False, metadata={}, 
                  include={"CWT": ("CWT", "cwt_"),
                         "EC": ("EC", ""),
                         "ECunc": ("EC", ""),
                         "DWT": ("DWT", "dwt_"),
                         "BM": ("BM", "")},
                  sel_cols={},#{"BM": ['TIMESTAMP', 'SW_IN', "NETRAD", 'PPFD_IN', "TA", "RH","WD"]}, 
                  verbosity=1):
    if verbosity:
        print('\nMERGING DATASETS\n')

    # collect flux datasets
    v_dta = {}
    for var, (fld, pr) in include.items():
        _path = os.path.join(path, fld, prefix+"_"+var+"_{}.{}mn.csv")
        #import glob
        #print(glob.glob(_path.replace("{}", "*")))
        #print([[(_path.format(*f), os.path.exists(_path.format(*f))) for f in [[y, str(a).zfill(2)] for y in ymd]] for a in averaging])
        _data = [reduce(lambda left, right: pd.concat([left, right]),
                        [pd.read_csv(_path.format(*f))
                        if os.path.exists(_path.format(*f)) else pd.DataFrame()
                        for f in [[y, str(a).zfill(2)] for y in ymd]])
                for a in averaging]
        _data = [w for w in _data if w.empty == False]
        if var in sel_cols.keys(): _data = [e[[c for c in sel_cols[var] if c in e.columns]] for e in _data]
        for i in range(len(_data)):
            _data[i].columns = [c if c in ["TIMESTAMP"] else f'{pr}{c}' for c in _data[i].columns]
        for w in _data:
            if not w.empty: w['TIMESTAMP'] = pd.to_datetime(w.TIMESTAMP)
        if verbosity: print(f'{var} ({len(_data)})')
        v_dta[var] = _data
    
    # collect dynamic metadata (e.g., measurement and vegetation height changing in time)
    mtpath = os.path.join(path, 'META/', prefix+"_{}_dynamic_metadata.csv")
    mt_data = [reduce(lambda left, right: pd.concat([left, right]),
                      [pd.read_csv(mtpath.format(y))
                       if os.path.exists(mtpath.format(y)) else pd.DataFrame()
                       for y in ymd])]
    mt_data = [w for w in mt_data if w.empty == False]
    for w in mt_data:
        if "TIMESTAMP" not in w.columns and "date" in w.columns and "time" in w.columns:
            w['TIMESTAMP'] = pd.to_datetime(
                w.date.astype(str) + " " + w.time.astype(str))
            w.drop(columns=["date", "time"], inplace=True)

        w.rename(columns={"master_sonic_height": "zm"}, inplace=True)
        w["z0"] = round(0.1 * w["canopy_height"] / 100, 2)
    if verbosity: print(f'Metadata ({len(mt_data)})')

    # collect metadata information (e.g., measurement and crop height fixed in time)
    mt_info = os.path.join(path, 'META/', prefix+"_metadata.yaml")
    if os.path.exists(mt_info):
        with open(mt_info, "r") as file:
            mt_info = yaml.safe_load(file)

    # collect EddyPro results
    if os.path.exists(os.path.join(path, "EP")) and len([f for f in os.listdir(os.path.join(path, "EP")) if os.path.isfile(os.path.join(path, "EP", f))]):
        try:
            """ep_data = [get_data.icos(path=path + "EP/").filter(
                #{'TIMESTAMP': (min(post_setup.time_list), max(post_setup.time_list) + pd.Timedelta('1D'))}
                {'TIMESTAMP': (pd.to_datetime('20190101'),
                            pd.to_datetime('20301231'))}
            ).data]"""
            ep_data = [get_rawdata.open_flux(path=os.path.join(path, "EP"), 
                                            lookup=pd.date_range(str(min(ymd))+'0101', str(max(ymd))+'1231', freq='30Min'), 
                                            onlynumeric=False).data]
            
            if 'EP' in sel_cols.keys():
                ep_data = [e[[c for c in sel_cols['EP'] if c in e.columns]]
                           for e in ep_data]
            
            """ep_data = [e[[c for c in ['TIMESTAMP', 'co2_flux', 'qc_co2_flux', #'Rg', 'PPFD', 'air_temperature',
                                    'rand_err_co2_flux', 'VPD', 'RH', 'LE', 'H',
                                    'h', 
                                    'air_molar_volume', 'air_pressure', 'e'] if c in e.columns]] for e in ep_data]"""
            if verbosity:
                print(f'EddyPro ({len(ep_data)})')
        except Exception as e:
            warnings.warn(str(e))
            ep_data = []
    else:
        ep_data = []
    
    # merge all collected datasets
    fluxResult = reduce(lambda left, right: pd.merge(left, right, on=['TIMESTAMP'],
                                                     how='outer', suffixes=(None, "y")),
                        (*tt.flist(list(v_dta.values())),#*wv_data, *ec_data, *bm_data
                         *ep_data,
                        )).sort_values('TIMESTAMP').reset_index(drop=True)
    fluxResult = reduce(lambda left, right: pd.merge(left, right, on=['TIMESTAMP'],
                                                     how='left'),
                        (fluxResult,
                        *mt_data)).sort_values('TIMESTAMP').reset_index(drop=True)
    
    if isinstance(mt_info, dict):
        for k, v in mt_info.items():
            fluxResult[k] = v

    # GUARANTEE FULL YEAR
    fluxResult = reduce(lambda left, right: pd.merge(left, right, on=['TIMESTAMP'],
                                                     how='inner'),
                        (fluxResult,
                        pd.DataFrame({"TIMESTAMP": pd.date_range(str(min(ymd))+'0101', str(max(ymd))+'1231', freq='30Min')})
                        )).sort_values('TIMESTAMP').reset_index(drop=True)
    

    # UNITS

    if 'air_molar_volume' in fluxResult.columns and 'air_pressure' in fluxResult.columns and 'e' in fluxResult.columns:
        for c in set(["cov_wco2", "cwt_co2w"] + [c for c in fluxResult.columns if c.startswith("dwt_wco2")]) & set(fluxResult.columns):
            # need to multiply by the dry air molar volume (Vd, m3mol-1)
            Vd = (fluxResult.air_molar_volume * fluxResult.air_pressure /
                  (fluxResult.air_pressure-fluxResult.e))
            fluxResult.loc[:, c] = fluxResult[c] / Vd
    
    #for c in ["cov_wco2", "dwt_co2w", "cwt_co2w"]:
    #    fluxResult.loc[:, c] = fluxResult[c] * 10**3

    """
    if 'co2_flux' in fluxResult.columns:
        for c in ["cov_wco2", "dwt_co2w", "cwt_co2w"]:
            r_ep_of = np.mean(abs(fluxResult["co2_flux"] / fluxResult[c]))
            if r_ep_of > 100:
                # pass from microgramC (remainder of ppm unit) to gC
                fluxResult[c] = fluxResult[c] * (r_ep_of//100)
    """
    

    # common units
    """
    for f in ["co2_w", "cov_wco2", "co2_flux"]:
        if (f in fluxResult.columns):
            if ((fluxResult[f].quantile(0.95) < 1) or (fluxResult[f].quantile(0.05) > -10)):
                fluxResult.loc[:, f] = fluxResult[f] * 10**3
            # drop absurd values
            fluxResult.loc[fluxResult[f] > 1.5*fluxResult[f].quantile(0.999), f] = np.nan
            fluxResult.loc[fluxResult[f] < 1.5*fluxResult[f].quantile(0.001), f] = np.nan
    """
    # save
    tt.mkdirs(output_path)
    fluxResult.to_csv(output_path, index=False)

    return fluxResult


def flag_dataset(ymd, path, prefix, output_path, postfix="_flagged", overwrite=False, verbosity=1):
    if verbosity:
        print('\nQA : QC\n')

    if isinstance(path, pd.DataFrame):
        fluxResult = path
    else:
        if 'fluxResult' not in globals():
            fluxResult = pd.read_csv(os.path.join(
                path, output_path.format("") if "{}" in output_path else output_path.replace(postfix, "")))
            fluxResult.TIMESTAMP = pd.to_datetime(fluxResult.TIMESTAMP)
        else:
            warnings.warn('fluxResult already in memory.')

    if verbosity:
        print(f'N: {len(fluxResult)}')
    
    filepath = os.path.join(path, "EC/", prefix + "_EC_{}.{}mn.csv")

    _dta30 = reduce(lambda left, right: pd.concat((left, right), axis=0), [
        pd.read_csv(filepath.format(y, '30')) for y in ymd if os.path.exists(filepath.format(y, '30'))])
    _dta30['TIMESTAMP'] = pd.to_datetime(_dta30.TIMESTAMP)
    _dta30 = pd.merge(_dta30, fluxResult[['TIMESTAMP', 'zm', 'z0']], how='left', on='TIMESTAMP')
    #_dta30['zm'] = 2#37
    #_dta30['z0'] = 0.02#18.76

    _dta30 = QA_QC.fITC(_dta30, 48.86, ustar='us')  # post_setup.site.latitude)

    if verbosity:
        print(f'fITC: {np.nansum(_dta30.fITC == False)}')

    _dta05 = reduce(lambda left, right: pd.concat((left, right), axis=0), [
        pd.read_csv(filepath.format(y, '05')) for y in ymd if os.path.exists(filepath.format(y, '05'))])
    _dta05['TIMESTAMP'] = pd.to_datetime(_dta05.TIMESTAMP)
    _dta30 = QA_QC.fSTA(_dta30, _dta05, bydate='TIMESTAMP')[
        ['TIMESTAMP', 'zm', 'z0', 'zL', 'sTp_Tpstar_mo', 'sw_ustar', 'itc_co2', 'fITC', 'sum_wco2', 'stat_wco2', 'fSTA']]

    if sum([os.path.exists(os.path.join(path, "DWT/", prefix + f"_DWT_{y}.{a}mn.csv")) for a in ['30', '05'] for y in ymd]):
        dwtpath = os.path.join(path, "DWT/", prefix + "_DWT_{}.{}mn.csv")
        _dta30dwt = reduce(lambda left, right: pd.concat((left, right), axis=0), [
            pd.read_csv(dwtpath.format(y, '30')) for y in ymd if os.path.exists(dwtpath.format(y, '30'))])
        _dta30dwt['TIMESTAMP'] = pd.to_datetime(_dta30dwt.TIMESTAMP)
        _dta05dwt = reduce(lambda left, right: pd.concat((left, right), axis=0), [
            pd.read_csv(dwtpath.format(y, '05')) for y in ymd if os.path.exists(dwtpath.format(y, '05'))])
        _dta05dwt['TIMESTAMP'] = pd.to_datetime(_dta05dwt.TIMESTAMP)

        _dta30dwt = QA_QC.fSTA(_dta30dwt, _dta05dwt, cov_name='co2w', bydate='TIMESTAMP')[
            ['TIMESTAMP', 'sum_wco2', 'stat_wco2', 'fSTA']]

    if verbosity:
        print(f'fSTA: {np.nansum(_dta30.fSTA == 0)}')

    fluxResult = pd.merge(fluxResult, _dta30, on=['TIMESTAMP'], how='outer')
    fluxResult = pd.merge(fluxResult, _dta30dwt, on=['TIMESTAMP'], how='outer', suffixes=(None, '_dwt'))

    del _dta30, _dta05

    fluxResult['overall_flag'] = np.max(
        (fluxResult.fITC, fluxResult.fSTA), axis=0)

    """
    # flag next to a gap
    completeDf = pd.merge(pd.DataFrame({'TIMESTAMP': pd.date_range(*tt.nanminmax(fluxResult.TIMESTAMP), freq='30min')}),
                          fluxResult, 
                          on='TIMESTAMP', how='outer').reset_index(drop=True)
    misDHM = np.isnan(completeDf.overall_flag)
    gapDHM = misDHM + np.array([1] + list(misDHM[1:])) + np.array(list(misDHM[:-1]) + [1])
    fluxResult['fNGap'] = [d in set(completeDf[gapDHM>0].TIMESTAMP) for d in fluxResult.TIMESTAMP]
    """
    fluxResult['fNGap'] = 0

    if verbosity:
        print(f'fNGap: {np.nansum(fluxResult.fNGap == 0)}')

    # save
    tt.mkdirs(output_path.format(postfix))
    fluxResult.to_csv(output_path.format(postfix), index=False)

    return


def gap_filling(path, output_path, latitude, longitude,
                cols={'CWT': {'var': 'co2w_x', 'flag': ['ITC']},
                      'DWT': {'var': 'co2w_y', 'flag': ['ITC']},
                      'EC': {'var': 'cov_wco2', 'flag': ['ITC', 'STA']}},
                overwrite=False, verbosity=1):
    latitude = float(latitude)
    longitude = float(longitude)
    REddyProcsMDSGapFill = tt.LazyCallable(
        os.path.join(cfp, "gapfilling.R"), 'REddyProcsMDSGapFill')
    r2gapfill = tt.LazyCallable(os.path.join(cfp, "gapfilling.R"), '*').__get__().fc
    r2gapfill.sink_reset()
    #r2gapfill.silenceR()

    # prepare data
    fluxResult = pd.read_csv(path)
    fluxResult.TIMESTAMP = pd.to_datetime(fluxResult.TIMESTAMP)

    fluxData = get_data.FluxTowerData(fluxResult)
    #fluxData = get_data.FluxTowerData(fluxResult10[fluxResult10.TIMESTAMP.dt.year==2019])


    fluxData.data = fluxData.data.rename(columns=
        {'air_temperature': 'Tair', 'SW_IN': 'Rg', 'RH': 'rH', 'ustar': 'Ustar', 'us': 'Ustar'})
    
    fluxData.data["Tair"] = fluxData.data.Tair-273 if np.nanmean(fluxData.data.Tair) > 100 else fluxData.data.Tair
    fluxData.data["Tsoil"] = fluxData.data.Tair
    #    #'VPD': fluxData.data.VPD/1000,
    #    #'date': fluxData.data.TIMESTAMP.dt.strftime('%Y%m%d')
    #    }).data
            
    if "VPD" not in fluxData.data.columns:
        fluxData.data["VPD"] = tt.vapour_deficit_pressure(
            fluxData.data.Tair, fluxData.data.rH)

    #fluxData.data.loc[np.isnan(fluxData.data.Ustar), 'Ustar'] = 0
    #fluxData.data = fluxData.data[np.isnan(fluxData.data.Ustar)]
    '''
    fluxData = get_data.FluxTowerData.load(
        path="./wavelets_for_flux/data/"+site_ec.sitename+"/doublerotation/FR-Fon_full_output_gapfilled.30mn.dat")
    '''

    # assert continuous time series
    #print('len(fluxData.data)', len(fluxData.data))
    fluxData.data = pd.merge(pd.DataFrame({'TIMESTAMP': pd.date_range(*tt.minmax(fluxData.data.TIMESTAMP), freq='30min')}),
                            fluxData.data,
                            on='TIMESTAMP', how='outer').reset_index(drop=True)
    #print('len(fluxData.data)', len(fluxData.data), 'duplicated:', ''.join("{} ({})".format(d, len(
    #    fluxData.data[fluxData.data.TIMESTAMP == d])) for d in set(fluxData.data[fluxData.data.duplicated("TIMESTAMP")].TIMESTAMP)))
    fluxData.data.drop_duplicates("TIMESTAMP", keep="first", inplace=True)
    
    #print('len(fluxData.data)', len(fluxData.data))

    #fluxData.data['TIMESTAMP_END'] = fluxData.data.TIMESTAMP + \
    #    pd.Timedelta('29.95min')

    # do it
    fluxData.fdta = {}

    #fluxData.data['Ustar'] = fluxData.data['us']
    #'co2_flux': 'EP', , 'co2_w_y': 'WV_Mo'
    for k, e in cols.items():
        print(k, end='\r')
        v = e.get('var', '')
        f = e.get('flag', [])
        l = e.get('flux', [])
        print(v)

        fill_data = copy.deepcopy(fluxData.data)
        fill_data = fill_data.rename(columns={v: 'NEE', 'qc_' + v: 'qc_NEE'})

        # drop flagged data
        for flag in f:
            for flux in ['NEE'] + l:
                fill_data.loc[fill_data[flag] > 0, flux] = np.nan
        """
        if 'fITC' in f:
            fill_data.loc[fill_data.fITC > 0, 'NEE'] = np.nan
        if 'fSTA' in f:
            fill_data.loc[fill_data.fSTA > 0, 'NEE'] = np.nan
        fill_data.loc[np.isnan(fill_data.fITC), 'NEE'] = np.nan
        """

        fill_data.loc[fill_data.Rg < 0, 'Rg'] = 0

        fill_data.TIMESTAMP = fill_data.TIMESTAMP.dt.strftime('%Y%m%d%H%M')
        #fill_data.TIMESTAMP_END = pd.to_datetime(fill_data.TIMESTAMP_END).dt.strftime(
        #    '%Y%m%d%H%M')
        #print('4')
        #print(["{}: {} {}".format(c, np.mean(fill_data[c]), np.nanmean(fill_data[c])) for c in ['Tair', 'Rg', 'VPD', 'NEE', 'Ustar', 'rH'] if c in fill_data.columns])
        fill_data = REddyProcsMDSGapFill.__call__(fill_data, ['Tair', 'Rg', 'VPD'], ['NEE'] + l,  # , 'LE'
                                                Lon=longitude, Lat=latitude
                                                ).reset_index(drop=True)
        #print('5')
        #print("fill_data.columns", [
        #      c for c in fill_data.columns if c.upper().startswith('RECO') or c.upper().startswith('GPP')], fill_data.columns)
        #fill_data = fill_data[[c for c in fill_data.columns if ((c in ['TIMESTAMP', 'Tair_f', 'Rg_f', 'VPD_f']) or (
        #    c.startswith('NEE')) or (c.startswith('Reco')) or (c.startswith('GPP')))]]
        #print("fill_data.columns after", [
        #      c for c in fill_data.columns if c.upper().startswith('RECO') or c.upper().startswith('GPP')])

        fluxData.fdta[k] = fill_data
        del k, v, f, fill_data

    # wrap it up
    fluxData.fdata = copy.deepcopy(fluxData.data)

    # climatic variables
    for c in ['Tair_f', 'Rg_f']:  # , 'VPD_f']:
        fluxData.fdata[c] = fluxData.fdta[list(fluxData.fdta.keys())[0]][c]

    # fluxes variables
    for k, d in fluxData.fdta.items():
        for c in d.columns:
            #cs = ['TIMESTAMP']
            if c.startswith('Reco') or c.startswith('GPP') or c.startswith('NEE'):
                #c = c.replace('uStar_', '')
                d.rename(columns={c: f'{k}_{c}'}, inplace=True)
                #d[f'{k}_{c}'] = d[c]
                #cs += [f'{k}_{c}']
            
            fluxData.fdata['TIMESTAMP'] = pd.to_datetime(fluxData.fdata.TIMESTAMP).dt.tz_localize(None)
            d['TIMESTAMP'] = pd.to_datetime(d.TIMESTAMP).dt.tz_localize(None)
            fluxData.fdata = pd.merge(fluxData.fdata, d[['TIMESTAMP']+list(set(d.columns)-set(fluxData.fdata.columns))], on='TIMESTAMP', how='left')
                
        """
        fluxData.fdata[k+'_FCO2_f'] = d.NEE_uStar_f
        fluxData.fdata[k+'_FCO2_o'] = d.NEE_uStar_orig
        if 'GPP_uStar_f' in d.columns:
            fluxData.fdata[k+'_GPP'] = d.GPP_uStar_f
        if 'Reco_uStar' in d.columns:
            fluxData.fdata[k+'_RECO'] = d.Reco_uStar
        """

    # save
    #fluxData.dump(post_setup.mother_folder + post_setup.site.sitename + "_full_output_flagged_gapfilled.30mn.dta")

    # save
    tt.mkdirs(output_path)
    fluxData.fdata.to_csv(output_path, index=False)
