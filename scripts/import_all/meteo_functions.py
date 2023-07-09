'''
METEOROLOGICAL FUNCTIONS
'''

import numpy as np

def wind_direction(u, v, offset=0):
    #wd = np.mod(180 + (180/np.pi) * np.arctan2(v, u), 360)
    '''Calculate raw wind direction from wind vector'''
    wd = 180 - (np.arctan2(v, u) * 180 / np.pi)

    '''accounts for user-supplied anemometer mis-alignment'''
    wd = wd + offset

    '''wrap within 0 - 360'''
    wd = np.mod(wd + 360, 360)

    return wd


def potential_temperature(Ta, Pa):
    return Ta * ((10**5)/Pa)**0.286


def monin_obukhov_length(Ustar, Tp, w=None, ts=None, H=None):
    k = 0.41  # von Karman
    g = 9.81  # [m/s2] gravitational acceleration

    L = - Tp * (Ustar**3) / (k * g * np.cov(w, ts)[0][1]) if np.cov(w, ts)[0][1] != 0 else np.nan

    # FHsfc = H
    #FHsfc = -0.05 # kinematic surface heat flux, FHsfc, in units of degrees Kelvin meter per second (K m/s)
    #L = ((-Ustar**3)*Tv) / (k*g*FHsfc)
    return L


def monin_obukhov_stability(zm, d, L):
    return (zm - d) / L


def ustar(u, v=None, w=None, axis=0):
    '''
    if isinstance(u, np.ndarray) and len(u.shape) == 3 and u.shape[0] == 3:
        u, v, w = u
    covuw = np.cov(u, w)[0]
    covvw = np.cov(v, w)[0]
    return np.array([(covuw[i-u.shape[0]]**2 + covvw[i-u.shape[0]]**2)**(0.25) for i in range(u.shape[0])])
    '''
    if isinstance(u, np.ndarray) and len(u.shape) == 3 and u.shape[0] == 3:
        assert u.shape[0] == 3, 'It needs to be (*3*, times, avg. period)'
        return np.array([ustar(u[0, i, :], u[1, i, :], u[2, i, :]) for i in range(u.shape[1])])
    '''elif isinstance(u, np.ndarray) and len(u.shape) == 2 and u.shape[0] == 3:
        assert u.shape[0] == 3, 'It needs to be (*3*, avg. period)'
        return ustar(u[0, :], u[1, :], u[2, :])'''
    if isinstance(u, np.ndarray) and len(u.shape) == 2:
        return np.array([ustar(u[i, :], v[i, :], w[i, :]) for i in range(u.shape[0])])
    else:
        return (np.cov(u, w)[0][1]**2 + np.cov(v, w)[0][1]**2)**(0.25)


def water_vapour_partial_pressure(Ta, Pa, Xh2o=None):
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

    return e


def vapour_deficit_pressure(T, RH):
    if np.nanquantile(T, 0.1) < 100 and np.nanquantile(T, 0.9) < 100:
        T = T + 274.15

    # Saturation Vapor Pressure (es)
    #es = 0.6108 * np.exp(17.27 * T / (T + 237.3))
    es = (T **(-8.2)) * (2.7182)**(77.345 + 0.0057*T-7235*(T**(-1)))

    # Actual Vapor Pressure (ea)
    ea = es * RH / 100

    # Vapor Pressure Deficit (Pa)
    return (es - ea)# * 10**(-3)


def flag_ustar(data, avg_p):
    if avg_p <= 0:
        avg_p = 20*60

    max_shrink = (avg_p*(len(data)//avg_p))
    u, v, w = data.u[:max_shrink], data.v[:max_shrink], data.w[:max_shrink]

    #print(np.array(data.TIMESTAMP[:max_shrink]).reshape(-1, avg_p)[:, 0])
    #u_ = ustar(np.array([u, v, w]).reshape(3, -1, avg_p))
    u_ = ustar(np.array(u).reshape(-1, avg_p),
               np.array(v).reshape(-1, avg_p),
               np.array(w).reshape(-1, avg_p))
    return (u_ < 0.1)


def lost_ustar(data, avg_p):
    u_ = flag_ustar(data, avg_p)
    return len(u_[u_]) / len(u_)
    return np.count_nonzero(u_ > 0) / len(u_)
