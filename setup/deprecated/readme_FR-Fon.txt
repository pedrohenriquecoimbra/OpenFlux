MACHINE READ FILE. PLEASE DO NOT MODIFY SPACES AND LINE JUMPS.

__init__::
"path": "C:/Users/phherigcoimb/OneDrive/INRAe/thesis-project-1/open_flux/OpenFlux_central.py"
"<PROJECT>": "ICOS"
"<SITE>": "FR-Fon"
"<SCRIPTS>": "C:/Users/phherigcoimb/OneDrive/INRAe/thesis-project-1/open_flux/scripts"
"<MOTHER>": "C:/Users/phherigcoimb/Desktop/INRAE_longfiles/ICOS/FR-Fon"
"<RAWPATH>": "/output/doublerotation/eddypro_raw_datasets"
"<YMD>": ('201901010000', '202112312359', '30min')
"<TIME_BEGIN>": "201901010000"
"<TIME_END>": 202212312359

__init__::OPEN_DEFAULTS::
"TREE": False

run_preparation::
"multiprocess": 1
"ymd": ('<TIME_BEGIN>', '<TIME_END>', '30min')
"path_output": "<MOTHER>/output/raw_datasets/"
"file_name": "<SITE>_raw_dataset_{}.csv"
"verbos": True

run_preparation::__init__::
"<path_steps>": "<SCRIPTS>/pre_processing.py"

run_preparation::steps::despike::
"name": "level_2"
"ds": "despike"
"fc": ('<path_steps>', 'level_2')
"save": True

run_preparation::steps::cross_wind::
"name": "level_3"
"ds": "-"
"fc": ('<path_steps>', 'level_3')
"skip": True

run_preparation::steps::angle_of_attack::
"name": "level_4"
"ds": "-"
"fc": ('<path_steps>', 'level_4')
"skip": True
"save": True
"overwrite": False

run_preparation::steps::tilt_axis::
"name": "level_5"
"ds": "tilt axis"
"fc": ('<path_steps>', 'level_5')

run_preparation::steps::tilt_axis::kwargs::
"method": "2r"
"setup_path": "<MOTHER>/output/log/pre_processing/"
"setup_name": "<SITE>_setup_double_rotation_{}.txt"
"save": False

run_preparation::steps::tilt_axis::kwargs::__text__::
"compaign": "<PROJECT>"
"mother_folder": "<MOTHER>"
"co_site": "<SITE>"

run_preparation::steps::time_lag::
"name": "level_6"
"ds": "time lag"
"fc": ('<path_steps>', 'level_6')
"save": False
"overwrite": False
"default_lag": 1

run_preparation::steps::time_lag::kwargs::
"buffer_period": "30min"

run_preparation::steps::detrending::
"name": "level_7"
"ds": "-"
"fc": ('<path_steps>', 'level_7')
"skip": True

run_preparation::steps::changing units::
"name": "level_6_good_units"
"ds": "changing units"
"fc": ('<SCRIPTS>/corrections/units.py', 'converter')
"save": True
"overwrite": False

run_preparation::steps::changing units::kwargs::Xgas::
"co2": "dryppm"

run_preparation::raw_kwargs::
"path": "<MOTHER>/input/EC/"
"onlynumeric": True
"verbosity": True

run_preparation::fmt_kwargs::
"cut": False
"addkeep": ['t_cell', 'press_cell']

run_preparation::fmt_kwargs::cols::
"co2": "co2_wet"
"h2o": "h2o_wet"
"co2_dry": "co2"
"h2o_dry": "h2o"
"t": "ts"

run_bioclimatology::
"ymd": [2020]
"rawfls_path": "<MOTHER>/input/BM/"
"output_path": "<MOTHER>/output/BM/FR-Gri_testBM_{}.csv"

run_eddycovariance::
"ymd": ('202103160000', '202103162359', '30min')
"output_path": "<MOTHER>/output/EC/tt/FR-Gri_EC{}_{}.{}mn.csv"

run_eddycovariance::raw_kwargs::k::
"multiprocess": 1
"mother_path": "<RAWPATH>/level_4/"
"file_pattern": "_raw_dataset_([0-9]*).*.csv"

run_eddycovariance::raw_kwargs::k::readkwargs::
"data": "eddypro_raw_datasets"
"date_format": "%Y%m%d%H%M"

run_eddycovariance::raw_kwargs::k::jlcsv::
"header": 1
"delim": ","
"dateform": "yyyy-mm-dd HH:MM:SS.s"
"acqrt": 20

run_eddycovariance::raw_kwargs::k2::
"multiprocess": 1
"mother_path": "<MOTHER><RAWPATH>/level_6/"
"file_pattern": "_raw_dataset_([0-9]*).*.csv"

run_eddycovariance::raw_kwargs::k2::readkwargs::
"data": "eddypro_raw_datasets"
"dateform": "yyyy-mm-dd HH:MM:SS.s"

run_eddycovariance::raw_kwargs::k2::jlcsv::
"header": 1
"delim": ","
"date_format": "%Y%m%d%H%M"
"acqrt": 20

consolidate_eddycovariance::
"ymd": None
"path": None
"pattern": None
"output_path": None

consolidate_eddycovariance::__init__::
"name": "EC by year"

run_wavelets::
"ymd": ('202103160000', '202103162359', '30min')
"dt": 0.05
"period": 60
"dj": 0.25
"max_period": 43200
"overwrite": True
"varstorun": ['co2_w', 'h2o_w', 'ts_w']
"mother": "MexicanHat"
"output_path": "<MOTHER>/output/WV/MexicanHat2r/"
"prefix": "FR-Gri_WV-{}_{}.flux"
"ignoreerr": True

run_wavelets::readkwargs::
"path": "<MOTHER><RAWPATH>/level_6/"

consolidate_wavelets::
"ymd": [2021]
"path": "<MOTHER>/output/WV/MexicanHat2r/<SITE>_WV-F{}_{}.flux"
"varstorun": ['co2_w', 'h2o_w', 'ts_w']
"output_path": "<MOTHER>/output/WV/MexicanHat2r/<SITE>_WV_{}.30mn.csv"

