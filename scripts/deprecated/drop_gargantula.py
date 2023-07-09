import ast
import re 
import json
#import codecs
#import ast
#import re
import os
import importlib
import pathlib
import inspect
import warnings

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
        if ('/' in self.modn) or ('\\' in self.modn):
            filn = os.path.basename(self.modn).split('.', 1)[0]
            dirn = os.path.dirname(self.modn)
            self.modn = os.path.relpath(
                dirn, pathlib.Path.cwd())
            self.modn = self.modn.replace('\\', '.') + '.' + filn if self.modn!='.' else filn
            #self.modn = self.modn.replace(
            #    '/', '.').replace('\\', '.')
        
        self.module = importlib.import_module(self.modn)

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

        return fc

    def __get__(self, *a, **k):
        if (self.fc is None):
            try:
                if (self.modn.lower().endswith('.r')) or ((self.n is not None) and (self.n.lower().endswith('.r'))):
                    self.fc = self.__get_R__(*a, **k)
                elif self.modn.lower().endswith('.jl'):
                    self.fc = self.__get_jl__(*a, **k)
                else:
                    self.fc = self.__get_py__(*a, **k)
            except Exception as e:
                warnings.warn(e)
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

class readable_file():
    def __init__(self, __path__, **kwargs):
        self.__path__ = os.path.join(
            __path__, 'readme.txt') if os.path.isdir(__path__) else __path__
        self.__text__ = 'MACHINE READ FILE. PLEASE DO NOT MODIFY SPACES AND LINE JUMPS.'
        self.__iden__ = ''
        self.__main__ = {}
        self.__main__.update(kwargs)
        #self.__dict__.update(kwargs)
        if isinstance(self.__text__, dict):
            self.__text__ = '\n'.join(
                ["'{}': {}".format(k, v) for k, v in self.__text__.items()])
        else:
            self.__text__ = str(self.__text__)

    def dump(self, max_iteration=1000, **kwargs):
        def check(body, text='', title=''):
            for k, v in {k: v for k, v in body.items(
            ) if isinstance(v, dict) == False}.items():
                if text[-2:] == '\n\n' or text == "":
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
                print(
                    "Nothing was done! Can only deal with dictionary and 'readable_file' Class. Try using 'readable_file(...)'.")
                return

        readable = self.__text__ + '\n\n'
        
        readable = readable + check(txt) + '\n\n'

        #readable = readable + '\n\n'.join([k + '::\n' + '\n'.join(['"{}": "{}"'.format(k_, v_) if isinstance(v_, str) else '"{}": {}'.format(k_, v_) for k_, v_ in v.items()]) for k, v in txt.items()])
        readable = readable + str(self.__iden__)

        with open(self.__path__, 'wb+') as f:
            f.write(readable.encode('utf-8'))
        return self

    def load_ipynb(self, **kwargs):
        with open(self.__path__, 'r') as f:
            source = f.read()

        y = json.loads(source)
        pySource = '##Python .py code from .jpynb:\n'
        for i, x in enumerate(y['cells']):
            if x['cell_type'] == 'code':
                pySource = pySource + 'def cell{}():\n'.format(i)

            for x2 in x['source']:
                if x['cell_type'] != 'code':
                    pySource = pySource + \
                        '\n'.join(['#' + l for l in x2.split('\n')])
                else:
                    pySource = pySource + \
                        '\n'.join(['\t' + l for l in x2.split('\n')])
                if x2[-1] != '\n':
                    pySource = pySource + '\n'
            pySource = pySource + '\n'

        with open(self.__path__.rsplit('.', 1)[0]+'_c.py', 'w+') as f:
            f.write(pySource)
        print(pySource)
        return self.load_py(getlocals=True)

    def load_py(self, getlocals=False, **kwargs):
        filename = os.path.basename(self.__path__).split('.', 1)[0]

        with open(self.__path__, 'r') as f:
            fil = f.read()

        function_marker = 'def ([A-z0-9]+)(\(.+\):)\n'
        argument_marker = "(?:\(|(?!\))|(?!\()(?:[, *]+))([A-z0-9]+)(?:\=)?(.*?)?(?=(?:(?:[, *]+|[=].*?)[A-z0-9]*)*\):)"
        fcs = re.findall(function_marker, fil)

        for fc in fcs:
            tit = fc[0]
            try:
                func = LazyCallable(self.__path__, fc[0]).__get__().fc
            except AttributeError:
                continue
            self.__main__.update({tit: {}})
            walking_dic = self.__main__[tit]
            signature = inspect.signature(func)
            walking_dic.update({k: v.default
                                if v.default is not inspect.Parameter.empty
                                else None
                                for k, v in signature.parameters.items()})
            if getlocals:
                walking_dic.update({k: v for k, v in dict(inspect.getmembers(func))[
                                   '__globals__'].items() if (k in signature.parameters.keys()) and (signature.parameters[k].default is inspect.Parameter.empty)})
            '''
            for arg in re.findall(argument_marker, fc[1]):
                if getlocals:
                    print(inspect.getmembers(module).func_defaults)
                    print(inspect.getmembers(module).func_globals)
                    print(fc, arg)
                    val
                else:
                    val = arg[1]
                walking_dic.update({arg[0]: val})
            '''
        return self

    def load(self, **kwargs):
        if self.__path__.endswith('.py'):
            return self.load_py(**kwargs)
        elif self.__path__.endswith('.ipynb'):
            #return self.load_ipynb(**kwargs)
            print('Jupyter notebook (.ipynb) not yet implemented.')
            return self

        with open(self.__path__, 'r') as f:
            fil = f.read()
        pts = fil.split('\n\n')
        self.__text__ = pts[0]
        self.__iden__ = pts[-1] if len(pts) > 2 else None
        for i, txt in enumerate(pts[1:-1]):
            tit = txt.split('\n', 1)[0].split(
                '::')[:-1] if '::' in txt.split('\n', 1)[0] else []
            walking_dic = self.__main__
            for t in tit[:-1]:
                if t not in walking_dic.keys():
                    walking_dic[t] = {}
                walking_dic = walking_dic[t]

            txt = txt.split(
                '::\n', 1)[-1] if '::' in txt.split('\n', 1)[0] else txt
            txt = ast.literal_eval('{' + txt.replace('\n', ', ') + '}')
            walking_dic.update({tit[-1]: txt} if tit else txt)
        return self

    def check_id(self, id, **kwargs):
        check = self.load(**kwargs)
        return str(check.__iden__) == str(id)

    def print(self):
        for k, v in vars(self).items():
            print("{}: {}".format(k, v if k != '__main__' else '\n'.join(
                "   %s: %s" % item for item in v.items())), '\n')
        return

    def to_dict(self):
        #{k_: v_ for k_, v_ in v.items()} if isinstance(v, dict) else
        return {k: v for k, v in self.__main__.items() if (not k.startswith('__') and not k.endswith('__')) or (k == '__init__')}
    
    def to_refdict(self):
        menu = self.to_dict()
        for k, v in menu.items():                            
            def seq_replace(string, dictionary):
                if re.findall('<.+>', str(string)):
                    if isinstance(string, dict):
                        string = {k__: seq_replace(v__, dictionary) for k__, v__ in string.items()}
                    elif isinstance(string, list) or isinstance(string, tuple):
                        string = [seq_replace(v__, dictionary) for v__ in string]
                    else:
                        for sk, sv in dictionary.items():
                            string = string.replace(sk, sv)
                return string
                            
            nest_cond = isinstance(v, dict) and '__init__' in v.keys()
            shortcuts = {k_: str(v_) for k_, v_ in menu['__init__'].items() if (k_.startswith('<')) and (k_.endswith('>'))} if ('__init__' in menu.keys()) else {}
            sec_shortcuts = {k_: seq_replace(v_, shortcuts) if re.findall('<.+>', str(v_)) else v_ for k_, v_ in v['__init__'].items() if (k_.startswith('<')) and (k_.endswith('>'))} if nest_cond else {}
            shortcuts.update({k_: str(v_) for k_, v_ in sec_shortcuts.items() if (k_.startswith('<')) and (k_.endswith('>'))} if nest_cond else {})

            #[print(v_, type(v_), re.findall('<.+>', str(v_))) for k_, v_ in v.items()]
            menu[k] = {k_: seq_replace(v_, shortcuts) if re.findall('<.+>', str(v_)) else v_ for k_, v_ in v.items()}
            #if k_ not in ['__init__']}
            
        return menu