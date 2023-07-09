MACHINE READ FILE. PLEASE DO NOT MODIFY SPACES AND LINE JUMPS.

"compaign": "COV3ER"
"co_site": "FR-Gri"
"mother_folder": "C:/Users/phherigcoimb/Desktop/INRAE_longfiles/COV3ER/FR-Gri/"
"raw_input_folder": "C:/Users/phherigcoimb/Desktop/INRAE_longfiles/COV3ER/FR-Gri/input/EC/"
"time_range": ('201606010000', '201607312359', '5min')

BM_SETUP::
"compaign": "COV3ER"
"co_site": "FR-Gri"
"mother_folder": "C:/Users/phherigcoimb/Desktop/INRAE_longfiles/COV3ER/FR-Gri/"
"raw_input_folder": "C:/Users/phherigcoimb/Desktop/INRAE_longfiles/COV3ER/FR-Gri/input/EC/"
"time_range": ('201606010000', '201607312359', '5min')
"rawfls_path": "C:/Users/phherigcoimb/Desktop/INRAE_longfiles/COV3ER/FR-Gri/input/BM/"
"output_path": "C:/Users/phherigcoimb/Desktop/INRAE_longfiles/COV3ER/FR-Gri/output/BM/"

EC_SETUP::
"compaign": "COV3ER"
"co_site": "FR-Gri"
"mother_folder": "C:/Users/phherigcoimb/Desktop/INRAE_longfiles/COV3ER/FR-Gri/"
"raw_input_folder": "C:/Users/phherigcoimb/Desktop/INRAE_longfiles/COV3ER/FR-Gri/input/EC/"
"time_range": ('201606010000', '201607312359', '5min')
"prefix": "FR-Gri_EC{}_{}.{}mn.csv"
"avg_period": [60, 120, 300, 600, 1800]
"multiprocess": 1
"overwrite": True
"output_path": "C:/Users/phherigcoimb/Desktop/INRAE_longfiles/COV3ER/FR-Gri/output/EC/doublerotation/"
"rawkwargs": [{'vars': None, 'multiprocess': 1, 'mother_path': 'C:/Users/phherigcoimb/Desktop/INRAE_longfiles/COV3ER/FR-Gri/output/raw_datasets/level_6/', 'file_pattern': '_raw_dataset_([0-9]*).*.csv', 'readkwargs': {'data': 'open_flux', 'date_format': '%Y%m%d%H%M'}, 'jlcsv': {'header': 1, 'delim': ',', 'dateform': 'yyyy-mm-dd HH:MM:SS.s', 'acqrt': 20}}, {'vars': ['u', 'v', 'w'], 'suffix': '_unc', 'multiprocess': 1, 'mother_path': 'C:/Users/phherigcoimb/Desktop/INRAE_longfiles/COV3ER/FR-Gri/output/raw_datasets/level_4/', 'file_pattern': '_raw_dataset_([0-9]*).csv', 'readkwargs': {'data': 'open_flux', 'date_format': '%Y%m%d%H%M'}, 'jlcsv': {'header': 1, 'delim': ',', 'dateform': 'yyyy-mm-dd HH:MM:SS.s', 'acqrt': 20}}]

WV_SETUP::
"compaign": "COV3ER"
"co_site": "FR-Gri"
"mother_folder": "C:/Users/phherigcoimb/Desktop/INRAE_longfiles/COV3ER/FR-Gri/"
"raw_input_folder": "C:/Users/phherigcoimb/Desktop/INRAE_longfiles/COV3ER/FR-Gri/input/EC/"
"time_range": ('201606010000', '201607312359', '5min')
"running_freq": "1D"
"dt": 0.05
"period": 60
"max_period": 43200
"dj": 0.25
"overwrite": True
"filefreq": 5
"ignoreerr": True
"savefig": False
"multiprocess": 1
"fileprefix": "FR-Gri_WV-{}_{}.flux"
"output_path": "C:/Users/phherigcoimb/Desktop/INRAE_longfiles/COV3ER/FR-Gri/output/WV/"
"varstorun": ['co2_w', 'h2o_w', 'ch4_w', 'co2_u', 'h2o_u', 'ch4_u']

WV_SETUP::wv2ec::
"co2_w": "co2_flux"
"ts_w": "H"
"h2o_w": "LE"

WV_SETUP::readkwargs::
"data": "open_flux"
"path": "C:/Users/phherigcoimb/Desktop/INRAE_longfiles/COV3ER/FR-Gri/output/raw_datasets/level_6/"

POST_SETUP::
"compaign": "COV3ER"
"co_site": "FR-Gri"
"mother_folder": "./wavelets_for_flux/data/COV3ER\FR-Gri\doublerotation/"
"raw_input_folder": "C:/Users/phherigcoimb/Desktop/INRAE_longfiles/COV3ER/FR-Gri/input/EC/"
"time_range": ('201606010000', '201607312359', '5min')

