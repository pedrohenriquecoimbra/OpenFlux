import numpy as np
import pandas as pd

def ppm_to_umol_m3(Xgas, Ta, Pa, Xh2o=None):
    # PV=nRT
    # Pa * m3 = mol * J K-1 mol-1 * K

    if np.quantile(Ta, 0.1) < 100 and np.quantile(Ta, 0.9) < 100:
        Ta = Ta + 274.15
    if np.quantile(Pa, 0.1) < 10 and np.quantile(Pa, 0.9) < 10:
        Pa = Pa * 10**3

    # optical_cell_volume (LICOR 7200: 16 cm3)
    V = 16 / 10**6

    # ℜ = 8.314 J mol-1K-1, the universal gas constant.
    R = 8.374

    # the number of moles inside optical cell (mol)
    n = (Pa * V) / (R * Ta)

    # the number of moles inside optical cell (mol/m3)
    n = (Pa) / (R * Ta)

    if Xh2o is not None:
        # the number of moles of H2O (mol)
        h2o_mols = Xh2o * n
        
        assert h2o_mols < n, 'Error! More mols of water than of air. Check units, water mole fraction should be in mol/mol.'

        # the number of moles inside optical cell wo H2O (mol)
        n_dry = n - h2o_mols
    else:
        n_dry = n

    # the gas concentration (mol.m-3)
    # ppm * n / V 
    return Xgas * n_dry / V

def dry_air_molar_conc(Ta, Pa, Xh2o=None):
    # Correct temperature (K to °C)
    if np.nanquantile(Ta, 0.1) < 100 and np.nanquantile(Ta, 0.9) < 100:
        Ta = Ta + 274.15
    
    # Correct pressure (KPa to Pa)
    if np.nanquantile(Pa, 0.1) < 10 and np.nanquantile(Pa, 0.9) < 10:
        Pa = Pa * 10**3

    # ℜ = 8.314 J mol-1K-1, the universal gas constant.
    R = 8.374
    
    # Ambient air molar volume
    Va = R * Ta / Pa

    # Molecular weight of water vapour (Mh2o, kgmol-1)
    Mh2o = 0.01802

    # Ambient water vapour mass density (ph2o, kg m-3)
    ph2o = Xh2o * Mh2o / Va

    # water vapor gas constant (Rh2o, JKg-1K-1)
    Rh2o = R / Mh2o

    # Water vapour partial pressure (e, Pa)
    e = ph2o * Rh2o * Ta

    # Dry partial pressure (Pd, Pa)
    Pd = Pa - e
    
    # Dry air molar volume (Vd, m3mol-1)
    Vd = Va * Pa / Pd
    
    return 1 / Vd

def converter(self, Xgas={'co2': 'dryppm'}, Ta='t_cell', Pa='press_cell', Xh2o='h2o', Xh2o_unit=10**-3, drop=True):
    for c, u in Xgas.items():
        if isinstance(u, (list, tuple)):
            #assert(len(u)==2), 'u is too long'
            v, u = u
        else:
            v = c
        if v == c and drop == False:
            self[c+'_bfr_cvt'] = self[c]
            
        self[v] = self[c] * dry_air_molar_conc(Ta=self[Ta], Pa=self[Pa], Xh2o=self[Xh2o]*Xh2o_unit)

    return self