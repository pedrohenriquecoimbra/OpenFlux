"""
Functions to retrieve data from sites and put in a standard format.
"""

from Lib.open_flux.scripts import common as tt
import pandas as pd
import os
import re
import numpy as np
from math import sqrt
import warnings

class FluxTowerData(tt.datahandler):
    def __init__(self, data, **kwargs):
        if isinstance(data, str):
            self.__dict__.update(self.__get__(data, **kwargs).__dict__)
        
        elif isinstance(data, FluxTowerData):
            return FluxTowerData(**data.__dict__)
        
        else:
            self.data = data
            self.__dict__.update(**kwargs)

    def __get__(self, fn, **kwargs):
        fn = re.sub('[ -]', '_', fn.lower())
        fc = tt.LazyCallable('wavelets_for_flux.scripts.get_data.' + fn)
        
        return fc.__call__(**kwargs)

    def filter(self, items: dict):
        for k, v in items.items():
            if k not in self.data.columns:
                warnings.warn(f"Column {k} not found in DataFrame.")
                return self
            if isinstance(v, tuple):
                self.data = self.data.loc[(self.data[k] > v[0])
                                          & (self.data[k] < v[1])]
            elif isinstance(v, list):
                self.data = self.data[self.data[k].isin(v)]
            elif isinstance(v, np.ndarray):
                self.data = self.data[self.data[k].isin(list(v))]
            else:
                self.data = self.data[self.data[k] == v]
        return self

    def rename(self, names: dict):
        self.data = self.data.rename(columns=names)
        return self

    def modify(self, items: dict):
        if self.data.empty:
            return
        for k, v in items.items():
            self.data[k] = v
        return self

    def expand(self, timefreq, key='TIMESTAMP'):
        eself = FluxTowerData(**self.__dict__)
        flag = pd.DataFrame({key: eself.data[key], 'flag': [1]*len(eself.data)})
        eself.data = pd.merge(
            pd.DataFrame(
                {key: pd.date_range(min(eself.data[key]), max(eself.data[key]), freq=timefreq)}),
            eself.data, on=key, how='outer')
        
        eself.data = eself.data.set_index(
            key).interpolate(method='time').reset_index()
        

        eself.data = pd.merge(eself.data, flag, on=key, how='left')
        eself.data.loc[np.isnan(eself.data.flag), 'flag'] = 0
        return eself
    """
    def preventive_drops(self, cols=None, verbosity=0):
        # drop time observation if any value is nan
        self.data = self.data.dropna(subset=cols)
        if verbosity > 1:
            print("drpn", min(self.data["TIMESTAMP"]), max(
                self.data["TIMESTAMP"]), len(self.data))

        # drop time observation if u* < 0.1 (it will either way will be ignored by Kljun's script)
        self.data = self.data[self.data['ustar'] > 0.1]
        if verbosity > 1:
            print("usta", min(self.data["TIMESTAMP"]), max(
                self.data["TIMESTAMP"]), len(self.data))

        # zm/ol (measurement height to Obukhov length ratio) must be equal or larger than -15.5.
        self.data = self.data[self.data['zm']/self.data['ol'] >= -15.5]
        if verbosity > 1:
            print("zm_L", min(self.data["TIMESTAMP"]), max(
                self.data["TIMESTAMP"]), len(self.data))
        
        return self
    """

'''
Site-specific functions to organise data into a dataframe
Once harmonized it should be ready to be piped through the FluxTowerData class
Attention, it should include variables such as:
- Eddy-covariance fluxes;
- Footprint model;
'''


def hu_hhs(flux_path="./wavelets_for_flux/data/HU-Hhs/flux_and_storage/",
           verbosity=0):
    
    bim_files = [re.match(r"^(\d{6})\.rcs", folder).group(
        1) for folder in os.listdir(flux_path) if re.match(r"^(\d{6})\.rcs", folder)]
    for i, el in enumerate(bim_files):
        bim_files[i] = pd.read_csv(flux_path + el + '.rcs',
                                   sep="\s+", na_values=[-999, -999.9])
        bim_files[i]['date'] = el

    df_biom = pd.concat(bim_files)

    flx_files = [re.match(r"^(\d{6})\.out", folder).group(
        1) for folder in os.listdir(flux_path) if re.match(r"^(\d{6})\.out", folder)]
    for i, el in enumerate(flx_files):
        flx_files[i] = pd.read_csv(flux_path + el + '.out',
                                sep="\s+", na_values=[-999, -999.9])
        flx_files[i]['date'] = el
    
    df_flux = pd.concat(flx_files)
    
    df_flux = df_flux.merge(df_biom, on=["date", "time"])

    df_flux['TIMESTAMP'] = df_flux['date'] + \
        df_flux.time.apply(lambda x: str(x).zfill(4))
    df_flux['TIMESTAMP'] = df_flux.TIMESTAMP.apply(
        pd.to_datetime, format='%y%m%d%H%M')

    df_flux["zm"] = 82
    df_flux["z0"] = 0.15  # Barcza et al 2009 AFM (for z0)
    df_flux['ustar'] = df_flux.mflux.apply(sqrt)

    df_flux['co2_flux'] = (df_flux['CO2flux'] + df_flux['rcs']) * 10**-3 / 44 * 10**6

    df_flux = df_flux.rename(columns={"L": "ol", "sdlws": "sigmav"})

    return FluxTowerData(df_flux)


def icos(path='./wavelets_for_flux/data/FR-Gri/',
           sous_path='',
           meta_path=None,
           biom_path=None,
           flux_path=None,
           verbosity=0, modify={}):
    """
    Simplify and put everything that varies in a single df
    """
    if path:
        meta_path = path + sous_path#+ 'output/' + sous_path
        biom_path = path + sous_path#+ 'input/'
        flux_path = path + sous_path#+ 'output/' + sous_path
        gapf_path = path + sous_path#+ 'output/' + sous_path
    
    if isinstance(meta_path, str) and meta_path.endswith('.csv'):
        meta_path = [meta_path]
    else:
        meta_path = [meta_path + e for e in os.listdir(meta_path) if re.findall(
            'metadata',  e, flags=re.IGNORECASE)]

    if isinstance(biom_path, str) and biom_path.endswith('.csv'):
        biom_path = [biom_path]
    else:
        biom_path = [biom_path + e for e in os.listdir(biom_path) if re.findall(
            'biomet',  e, flags=re.IGNORECASE)]

    if isinstance(flux_path, str) and flux_path.endswith('.csv'):
        flux_path = [flux_path]
    else:
        flux_path = [flux_path + e for e in os.listdir(flux_path) if re.findall(
            'full_output',  e, flags=re.IGNORECASE)]

    if isinstance(gapf_path, str) and gapf_path.endswith('.csv'):
        gapf_path = [gapf_path]
    else:
        gapf_path = [gapf_path + e for e in os.listdir(gapf_path) if re.findall(
            'gapfilled_output',  e, flags=re.IGNORECASE)]

    def get_output_eddypro(pathlst, **kwargs):
        df_ep = pd.DataFrame()
        #print(pathlst)
        for p in pathlst:
            if p.endswith('.csv'):
                assert os.path.exists(p)
                df_ep_ = pd.read_csv(p, **kwargs)
                df_ep_["TIMESTAMP"] = pd.to_datetime(
                    df_ep_['date'] + " " + df_ep_['time'])#, format='%Y-%m-%d %H:%M')
                # - 0.8 * (df_meta_["canopy_height"] / 100)
                df_ep = pd.concat([df_ep, df_ep_])
        return df_ep
    '''
    df_meta = pd.DataFrame()
    for m_p in meta_path:
        if m_p.endswith('.csv'):
            assert os.path.exists(m_p)
            df_meta_ = pd.read_csv(m_p, na_values=[-9999])
            df_meta_["TIMESTAMP"] = pd.to_datetime(
                df_meta_['date'] + " " + df_meta_['time'], format='%d/%m/%Y %H:%M')
            # - 0.8 * (df_meta_["canopy_height"] / 100)
            df_meta = df_meta.append(df_meta_)
    '''
    df_meta = get_output_eddypro(meta_path, na_values=[-9999])
    df_flux = get_output_eddypro(flux_path, skiprows=[0, 2], na_values=[-9999])

    df_biom = pd.DataFrame()
    for b_p in biom_path:
        if b_p.endswith('.csv'):
            assert os.path.exists(b_p)
            df_biom_ = pd.read_csv(b_p, skiprows=[1], na_values=[-9999])
            if "TIMESTAMP" in df_biom_.columns:
                df_biom_["TIMESTAMP"] = pd.to_datetime(
                    df_biom_["TIMESTAMP"], format='%Y-%m-%d %H:%M')
            else:
                df_biom_["TIMESTAMP"] = pd.to_datetime(
                    df_biom_[df_biom_.columns[0]] +' ' + df_biom_[df_biom_.columns[1]], format='%Y-%m-%d %H:%M')
            df_biom = df_biom.append(df_biom_)

    '''
    df_flux = pd.DataFrame()
    for f_p in flux_path:
        if f_p.endswith('.csv'):
            assert os.path.exists(f_p)
            df_flux_ = pd.read_csv(f_p, skiprows=[0, 2], na_values=[-9999])
            df_flux_["TIMESTAMP"] = pd.to_datetime(
                df_flux_['date'] + " " + df_flux_['time'], format='%Y-%m-%d %H:%M')
            df_flux = df_flux.append(df_flux_)
    '''
    df_gapf = pd.DataFrame()
    for g_p in gapf_path:
        if g_p.endswith('.csv'):
            assert os.path.exists(g_p)
            df_gapf_ = pd.read_csv(g_p, na_values=[-9999])
            df_gapf = df_gapf.append(df_gapf_)

    for df_ in [df_flux, df_meta, df_biom, df_gapf]:
        if 'TIMESTAMP' in df_.columns:
            '''ignore time zone'''
            df_["TIMESTAMP"] = pd.to_datetime(
                df_["TIMESTAMP"]).dt.tz_localize(None)  # .astype('datetime64[ns]')

    #print('fl', df_flux.TIMESTAMP[df_flux.TIMESTAMP.duplicated()])
    if len(meta_path):
        #print('md', df_meta.TIMESTAMP[df_meta.TIMESTAMP.duplicated()])
        df_flux = df_flux.merge(df_meta, on=["TIMESTAMP"], how='left')
    if len(biom_path):
        #print('bm', df_biom.TIMESTAMP[df_biom.TIMESTAMP.duplicated()])
        df_flux = df_flux.merge(df_biom, on=["TIMESTAMP"], how='left')
    if len(gapf_path):
        #print('gp', df_gapf.TIMESTAMP[df_gapf.TIMESTAMP.duplicated()])
        df_flux = df_flux.merge(df_gapf, on=["TIMESTAMP"], how='left')

    if modify:
        for k, v in modify.items():
            df_flux[k] = v

    if 'master_sonic_height' in df_flux.columns:
        df_flux["zm"] = df_flux["master_sonic_height"]
    if 'canopy_height' in df_flux.columns:
        df_flux["z0"] = 0.1 * (df_flux["canopy_height"] / 100)

    if verbosity > 1:
        print("flux:", min(df_flux["TIMESTAMP"]), max(
            df_flux["TIMESTAMP"]), len(df_flux),
            "\nmeta:", min(df_meta["TIMESTAMP"]), max(
            df_meta["TIMESTAMP"]), len(df_meta),
            "\nbiom:", min(df_biom["TIMESTAMP"]), max(
            df_biom["TIMESTAMP"]), len(df_biom))

    df_flux = df_flux.rename(columns={"L": "ol", "u*": "ustar", "wind_dir": "wd"})
    if "v_var" in df_flux.columns:
        df_flux['sigmav'] = np.sqrt(df_flux.v_var)
    
    #df_flux['PPFD'] = np.nanmean(df_flux[[c for c in df_flux.columns if re.match('PPFD_\d_\d_\d', c, flags=re.IGNORECASE)]], axis=1)
    #df_flux['Rg'] = np.nanmean(df_flux[[c for c in df_flux.columns if re.match('RG_\d_\d_\d', c, flags=re.IGNORECASE)]], axis=1)

    return FluxTowerData(df_flux, flux_path=flux_path)
