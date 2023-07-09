import pywt
import numpy as np
import matplotlib.pyplot as plt
import itertools

class multilevelflux:
    def __init__(self, **kwargs):
        "fluxes"
        self.__dict__.update(**kwargs)

    def example():
        # run example
        level = 4
        X = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
        coeffs = pywt.wavedec(X, 'db1', level=level)
        A4 = wrcoef(X, 'a', coeffs, 'db1', level)
        D4 = wrcoef(X, 'd', coeffs, 'db1', level)
        D3 = wrcoef(X, 'd', coeffs, 'db1', 3)
        D2 = wrcoef(X, 'd', coeffs, 'db1', 2)
        D1 = wrcoef(X, 'd', coeffs, 'db1', 1)

        fig = plt.figure(**dict(figsize=(12, 4), dpi=72))
        fig_grid = plt.GridSpec(1, 2)

        plt.subplot(fig_grid[0, 0])
        plt.plot(X)
        plt.plot(A4 + D4 + D3 + D2 + D1)
        plt.subplot(fig_grid[0, 1])
        plt.imshow([A4, D4, D3, D2, D1]) 
    
    def get_flux(X1=None, X2=None, Y1=None, Y2=None, level=None, wave="db6"):
        [Y1, Y2], level = decompose(X1, X2, level=level, wave=wave)
        """
        coeffs = pywt.wavedec(X1, wave, level=level)
        level = len(coeffs)-1
        A1 = wrcoef(X1, 'a', coeffs, wave, level)
        D1 = [wrcoef(X1, 'd', coeffs, wave, i) for i in range(1, level+1)]
        Y1 = np.array(D1 + [A1])

        coeffs = pywt.wavedec(X2, wave, level=level)
        level = len(coeffs)-1
        A2 = wrcoef(X2, 'a', coeffs, wave, level)
        D2 = [wrcoef(X2, 'd', coeffs, wave, i) for i in range(1, level+1)]
        Y2 = np.array(D2 + [A2])
        """
        Y12 = Y1 * Y2.conjugate()
        return Y12, level


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
            xy = np.where(xy==0, false, xy)
            Ys[name] = Ys[name] * xy

    """
    # create a mask
    for i, Y in enumerate(YS):
        for s in [1, -1]:
            Y_ = np.ones(Y.shape)
            Y_[s*Y <= 0] = false
            Yi[f"{label[s]}x{i}"] = Y_
    
    Ys = {}
    for combination in set(itertools.combinations([1, -1]*nargs, nargs)):
        cs = ["{}x{}".format(label[c], i) for i, c in enumerate(combination)]
        Ys['_'.join(cs)] = Y12 * sum([Yi[c] for c in cs])
    """
    return Ys
    
def decompose(*args, level=None, wave="db6"):
    Ys = []
    for X in args:
        coeffs = pywt.wavedec(X, wave, level=level)
        level = len(coeffs)-1
        A1 = wrcoef(X, 'a', coeffs, wave, level)
        D1 = [wrcoef(X, 'd', coeffs, wave, i) for i in range(1, level+1)]
        Ys += [np.array(D1 + [A1])]
    return Ys, level


def wrcoef(X, coef_type, coeffs, wavename, level):
    N = np.array(X).size
    a, ds = coeffs[0], list(reversed(coeffs[1:]))

    if coef_type == 'a':
        return pywt.upcoef('a', a, wavename, level=level, take=N)  # [:N]
    elif coef_type == 'd':
        # [:N]
        return pywt.upcoef('d', ds[level-1], wavename, level=level, take=N)
    else:
        raise ValueError("Invalid coefficient type: {}".format(coef_type))
