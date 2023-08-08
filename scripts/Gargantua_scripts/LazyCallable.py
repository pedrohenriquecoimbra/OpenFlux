import sys
import os
import re
import importlib
import pathlib
    
# Julia

class LazyCallable(object):
    def __init__(self, name, fcn=None, pkg=None):#pathlib.Path.cwd()):
        self.modn = name
        self.fc = None
        self.n = fcn
        self.pkg = str(pkg) if pkg else None

    def __get_jl__(self, *a, **k):
        #from julia.api import Julia
        #jl = Julia(compiled_modules=True)
        from julia import Main as jl
        importlib.reload(jl)
        fc = jl.eval('include("{}"); {}'.format(self.modn, self.n))
        del jl
        return fc

    def __get_py__(self, *a, **k):
        if ('/' in self.modn) or ('\\' in self.modn):
            filn = os.path.basename(self.modn).split('.', 1)[0]
            dirn = os.path.dirname(self.modn)
            self.modn = os.path.relpath(dirn, self.pkg)
            self.modn = self.modn.replace(
                '\\', '.') + '.' + filn if self.modn != '.' else filn
            #self.modn = self.modn.replace(
            #    '/', '.').replace('\\', '.')
        
        self.module = importlib.import_module(
            self.modn, self.pkg)

        if self.n is not None:
            fc = getattr(self.module, self.n)
            return fc

        else:
            return self.module
        if self.n is None:
            self.n = self.modn.rsplit('.', 1)[1]

        self.module = importlib.import_module(self.modn)
        fc = getattr(self.module, self.n)
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
        fc = RPackage(
            ''.join(preamble + open(self.modn, 'r').readlines()), 'rpy2_fc')

        if not self.n:
            print("Warning! No function name given, return function with the same name as file. If not found returns first function.")

            fcn = [c for c in fc.__dict__['_exported_names'] if (c == self.modn.rsplit(
                '/', 1)[1].rsplit('.', 1)[0]) and (c in fc.__dict__.keys())]
            fcn = fcn[0] if fcn else list(fc.__dict__['_exported_names'])[0] if list(
                fc.__dict__['_exported_names'])[0] in fc.__dict__.keys() else ''
            if fcn:
                self.n = fcn

        if self.n and self.n != '*':
            fc = fc.__dict__[self.n]

        del RPackage, importr
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


if __name__ == '__main__':
    args = [a for a in sys.argv[3:] if '=' not in a ]
    kwargs = dict([a.split('=') for a in sys.argv[3:] if '=' in a])
    print(LazyCallable(*sys.argv[1:3]).__call__(*args, **kwargs))
