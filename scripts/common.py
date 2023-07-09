import os
import importlib
import pathlib

cfp = pathlib.Path(__file__).parent.resolve()

#filepath = os.path.join(cfp, 'import_all')
#filepath = os.path.abspath(os.path.join(pathlib.Path(__file__).parent.resolve(), "..", "..", 'main_scripts'))


def import_from_anywhere(module, package=None, n=None):
    #print(module, package)
    # get a handle on the module
    mdl = importlib.import_module(module, package)

    # is there an __all__?  if so respect it
    if "__all__" in mdl.__dict__:
        names = mdl.__dict__["__all__"]
    else:
        # otherwise we import all names that don't begin with _
        names = [x for x in mdl.__dict__ if not x.startswith("_")]

    if n:
        names = [x for x in names if not x in n]

    # now drag them in
    globals().update({k: getattr(mdl, k) for k in names})


def importlib_to_globals(fpath, rpath=os.getcwd()):
    for root, _, files in os.walk(fpath):
        for name in files:
            if os.path.isfile(os.path.join(root, name)) and name.endswith('.py'):
                #print(name)
                #print(os.path.relpath(root, rpath))
                import_from_anywhere(os.path.relpath(root, rpath).replace(
                    '\\', '.') + '.' + name.split('.', 1)[0], rpath)


importlib_to_globals(os.path.join(cfp, 'import_all'))
