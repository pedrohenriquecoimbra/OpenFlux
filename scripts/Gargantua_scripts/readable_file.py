import sys
import ast
import re
import json
import os
import inspect
import yaml
from collections import OrderedDict
import copy
import queue
from .LazyCallable import LazyCallable
import warnings


def setup_yaml():
  """ https://stackoverflow.com/a/8661021 """
  represent_dict_order = lambda self, data:  self.represent_mapping('tag:yaml.org,2002:map', data.items())
  yaml.add_representer(OrderedDict, represent_dict_order)    

setup_yaml()    


# Custom type
def checkcustomtypes(var):
    if re.match("f\( *[\"']?(.+)[\"']? *, *[\"']?(.+)[\"']? *\)", var):
        return function
    else:
        return type(var)

def function(mod_fcn):
    try:
        modn, fcn = re.findall(
            "f\( *[\"']?(.+)[\"']? *, *[\"']?(.+)[\"']? *\)", mod_fcn)[0]
        fc = queue.Queue()
        fc.put(LazyCallable(modn, fcn).__get__().fc)
        return fc
    except Exception as e:
        #raise TypeError(f"{mod_fcn} not in a function 'f(module, script)'.")
        warnings.warn(f"{mod_fcn} not in a function 'f(module, script)'.")
        return mod_fcn

def casttypedictionary(dic, dt):
    #dic = copy.deepcopy(d)
    if isinstance(dic, dict):
        dic = {k_: casttypedictionary(v_, dt[k_]) if isinstance(dt, dict) and k_ in dt.keys() else v_ for k_, v_ in dic.items()}
    else:
        try:
            if type(dic).__name__ != dt["type"]:
                dic = globals()[dt["type"]](dic)
        except Exception as e:
            warnings.warn(str(e) + f"\ndt: {dt}\ndic: {dic}")
        """
        if callable(dt):
            if type(dic).__name__ != dt.__name__:
                dic = dt(dic)
        else:
            if isinstance(dt, dict) and "type" in dt.keys():
                dt = dt["type"] 
            if callable(dt["type"]):
                if type(dic).__name__ != dt.__name__:
                    dic = dt(dic)
            else:
                if type(dic).__name__ != dt:
                    dic = locals()[dt["type"]](dic)
        """
    return dic

def referencedictionary(d, meta=None, kinit=True):
    dic = copy.deepcopy(d)
    _init = dic.pop("__init__", {})
    shortcuts = {k_: str(v_) for k_, v_ in _init.items() if (
        k_.startswith('<')) and (k_.endswith('>'))}

    for k, v in dic.items():
        if isinstance(v, dict)==False:
            continue
            
        def refine(string, dictionary):
            # add references
            if re.findall('<.+>', str(string)):
                if isinstance(string, dict):
                    string = {k__: refine(
                        v__, dictionary) for k__, v__ in string.items()}
                elif isinstance(string, list) or isinstance(string, tuple):
                    string = [refine(v__, dictionary)
                              for v__ in string]
                else:
                    for sk, sv in dictionary.items():
                        if sv is not None:
                            string = string.replace(sk, sv)
            return string
                
        _init_ = {k_: refine(v_, shortcuts) for k_, v_ in v.pop("__init__", {}).items()}
        soushortc = {k_: v_ for k_, v_ in _init_.items() if (k_.startswith('<')) and (k_.endswith('>'))}
        shortcuts.update(soushortc)
        
        dic[k] = {k_: refine(v_, shortcuts) for k_, v_ in v.items()}
        if kinit and _init_: dic[k].update({"__init__": _init_})
    if kinit and _init: dic.update({"__init__": _init})

    if meta:
        dic = casttypedictionary(dic, meta)
    return dic


class readable_file():
    def __init__(self, __path__, **kwargs):
        self.__path__ = os.path.join(
            __path__, 'readme.yaml') if os.path.isdir(__path__) else __path__
        self.__text__ = 'MACHINE READ FILE. PLEASE DO NOT MODIFY SPACES AND LINE JUMPS.'
        self.__iden__ = ''
        self.__main__ = OrderedDict({})
        self.__main__.update(kwargs)
        #self.__dict__.update(kwargs)
        if isinstance(self.__text__, (dict, OrderedDict)):
            self.__text__ = '\n'.join(
                ["'{}': {}".format(k, v) for k, v in self.__text__.items()])
        else:
            self.__text__ = str(self.__text__)

    def load(self, safe=False):
        with open(self.__path__, 'r') as f:
            self.__main__ = yaml.safe_load(f) if safe else yaml.load(f)
            #self.__main__ = OrderedDict(self.__main__)
        self.__iden__ = self.__main__.pop("gargantula_unique_identifier", None)
        return self
    
    def safe_load(self):
        self.load(safe=True)
        return self
    
    def dump(self, safe=False):
        if self.__iden__: self.__main__["gargantula_unique_identifier"] = self.__iden__
        with open(self.__path__, 'w+') as f:
            #print(type(self.__main__))
            yaml.safe_dump(self.__main__, f) if safe else yaml.dump(self.__main__, f)
        return self

    def safe_dump(self):
        self.dump(safe=True)
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
        #if (not k.startswith('__') and not k.endswith('__')) or (k == '__init__')
        return OrderedDict({k: v for k, v in self.__main__.items()})

    def to_refdict(self):
        menu = self.to_dict()
        menu = referencedictionary(menu, kinit=True)
        """
        for k, v in menu.items():
            def seq_replace(string, dictionary):
                if re.findall('<.+>', str(string)):
                    if isinstance(string, dict):
                        string = {k__: seq_replace(
                            v__, dictionary) for k__, v__ in string.items()}
                    elif isinstance(string, list) or isinstance(string, tuple):
                        string = [seq_replace(v__, dictionary)
                                  for v__ in string]
                    else:
                        for sk, sv in dictionary.items():
                            string = string.replace(sk, sv)
                return string

            nest_cond = isinstance(v, dict) and '__init__' in v.keys()
            shortcuts = {k_: str(v_) for k_, v_ in menu['__init__'].items() if (
                k_.startswith('<')) and (k_.endswith('>'))} if ('__init__' in menu.keys()) else {}
            sec_shortcuts = {k_: seq_replace(v_, shortcuts) if re.findall(
                '<.+>', str(v_)) else v_ for k_, v_ in v['__init__'].items() if (k_.startswith('<')) and (k_.endswith('>'))} if nest_cond else {}
            shortcuts.update({k_: str(v_) for k_, v_ in sec_shortcuts.items() if (
                k_.startswith('<')) and (k_.endswith('>'))} if nest_cond else {})

            #[print(v_, type(v_), re.findall('<.+>', str(v_))) for k_, v_ in v.items()]
            menu[k] = {k_: seq_replace(v_, shortcuts) if re.findall(
                '<.+>', str(v_)) else v_ for k_, v_ in v.items()}
            #if k_ not in ['__init__']}
        """
        return menu

class readable_file_old():
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
                        string = {k__: seq_replace(
                            v__, dictionary) for k__, v__ in string.items()}
                    elif isinstance(string, list) or isinstance(string, tuple):
                        string = [seq_replace(v__, dictionary)
                                  for v__ in string]
                    else:
                        for sk, sv in dictionary.items():
                            string = string.replace(sk, sv)
                return string

            nest_cond = isinstance(v, dict) and '__init__' in v.keys()
            shortcuts = {k_: str(v_) for k_, v_ in menu['__init__'].items() if (
                k_.startswith('<')) and (k_.endswith('>'))} if ('__init__' in menu.keys()) else {}
            sec_shortcuts = {k_: seq_replace(v_, shortcuts) if re.findall(
                '<.+>', str(v_)) else v_ for k_, v_ in v['__init__'].items() if (k_.startswith('<')) and (k_.endswith('>'))} if nest_cond else {}
            shortcuts.update({k_: str(v_) for k_, v_ in sec_shortcuts.items() if (
                k_.startswith('<')) and (k_.endswith('>'))} if nest_cond else {})

            #[print(v_, type(v_), re.findall('<.+>', str(v_))) for k_, v_ in v.items()]
            menu[k] = {k_: seq_replace(v_, shortcuts) if re.findall(
                '<.+>', str(v_)) else v_ for k_, v_ in v.items()}
            #if k_ not in ['__init__']}

        return menu


if __name__ == '__main__':
    print(readable_file(*sys.argv[1:]).load().print())
