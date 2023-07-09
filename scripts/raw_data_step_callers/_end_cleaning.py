import os
from Lib.open_flux.scripts import common as tt
import pandas as pd
import pathlib
# current file path
cfp = pathlib.Path(__file__).parent.resolve()

def clean(self, date, path, file_pattern):
    # This function deletes files, use with caution

    date = pd.to_datetime(date) - pd.Timedelta('30D')
    found_files = tt.get_files_paths_using_regex(path, startswith=str(date), pattern=file_pattern)
    [os.remove(v) for k, v in found_files.items() if pd.to_datetime(k) < date]
    return self