MACHINE READ FILE. PLEASE DO NOT MODIFY SPACES AND LINE JUMPS.

__init__::
"path": "C:/Users/phherigcoimb/OneDrive/INRAe/thesis-project-1/open_flux/OpenFlux_central.py"
"<PROJECT>": "PAUL"
"<SITE>": "PARIS-Jus"
"<SCRIPTS>": "C:/Users/phherigcoimb/OneDrive/INRAe/thesis-project-1/open_flux/scripts"
"<MOTHER>": "C:/Users/phherigcoimb/Desktop/INRAE_longfiles/PAUL/PARIS-Jus"
"<RAWPATH>": "/output/raw_datasets"
"<YMD>": ('201901010000', '202112312359', '30min')
"<TIME_BEGIN>": 202302150000
"<TIME_END>": 202303302359

__init__::__run_selected__::
"ymd": ('202302150000', '202303302359', '30min', '1D')

__init__::__run_selected__::__init__::
"path": "C:/Users/phherigcoimb/OneDrive/INRAe/thesis-project-1/open_flux/OpenFlux_central.py"
"function": "run_openflux"

__init__::OPEN_DEFAULTS::
"TREE": False

__init__::OPEN_DEFAULTS::selected_functions::run_preparation::
"state": 0

__init__::OPEN_DEFAULTS::selected_functions::run_bioclimatology::
"state": 0

__init__::OPEN_DEFAULTS::selected_functions::run_eddycovariance::
"state": 0

__init__::OPEN_DEFAULTS::selected_functions::consolidate_eddycovariance::
"state": 0

__init__::OPEN_DEFAULTS::selected_functions::run_wavelets::
"state": 0

__init__::OPEN_DEFAULTS::selected_functions::consolidate_wavelets::
"state": 0

run_preparation::
"multiprocess": 1
"popup": False
"ymd": ('<TIME_BEGIN>', '<TIME_END>', '30min')
"path_output": "<MOTHER>/output/raw_datasets/"
"file_name": "<SITE>_raw_dataset_{}.csv"
"verbos": True
"result": False

run_preparation::__init__::
"<path_steps>": "<SCRIPTS>/raw_data_step_callers/"

run_preparation::first_step::kwargs::
"onlynumeric": True
"verbosity": True
"path": "<MOTHER>/input/EC/"

run_preparation::first_step::kwargs::fmt::
"cut": False
"addkeep": ['t_cell', 'press_cell']

run_preparation::first_step::kwargs::fmt::cols::
"co2": "co2_wet"
"h2o": "h2o_wet"
"co2_dry": "co2"
"h2o_dry": "h2o"
"t": "ts"

run_preparation::steps::despike::
"name": "level_2"
"ds": "despike"
"fc": ('<path_steps>/_2_despike.py', 'despike')
"save": True

run_preparation::steps::tilt_axis::
"name": "level_5"
"ds": "tilt axis"
"fc": ('<path_steps>/_5_tilt_rotation.py', 'tilt_rotation')

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
"fc": ('<path_steps>/_6_time_lag.py', 'time_lag')
"save": False
"overwrite": False
"skip": False
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

run_preparation::steps::changing units::kwargs::
"drop": False

run_preparation::steps::changing units::kwargs::Xgas::
"co2": "dryppm"

run_bioclimatology::
"ymd": [2020]
"rawfls_path": "<MOTHER>/input/BM/"
"output_path": "<MOTHER>/output/BM/FR-Gri_testBM_{}.csv"

run_eddycovariance::
"ymd": ('<TIME_BEGIN>', '<TIME_END>', '30min')
"output_path": "<MOTHER>/output/EC/openflux/<SITE>_EC{}_{}.{}mn.csv"
"averaging": [30, 5]
"overwrite": True

run_eddycovariance::raw_kwargs::
"file_pattern": ".*_raw_dataset_([0-9]*).csv"
"mother_path": "<MOTHER><RAWPATH>/level_6_good_units/"
"multiprocess": 1

run_eddycovariance::raw_kwargs::jlcsv::
"header": 1
"dateform": "yyyy-mm-ddTHH:MM:SS.ss"
"acqrt": 20
"delim": ","

run_eddycovariance::unc_kwargs::
"multiprocess": 1
"mother_path": "<MOTHER><RAWPATH>/level_2/"
"file_pattern": ".*_raw_dataset_([0-9]*).csv"

run_eddycovariance::unc_kwargs::jlcsv::
"delim": ","
"acqrt": 20
"dateform": "yyyy-mm-ddTHH:MM:SS.ss"
"header": 1

consolidate_eddycovariance::
"ymd": [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]
"path": "<MOTHER>/output/EC/openflux/"
"pattern": "<SITE>_EC_([0-9]*).30mn.csv"
"output_path": "<MOTHER>/output/EC/<SITE>_EC{}_{}.{}mn.csv"

consolidate_eddycovariance::__init__::
"name": "EC consolidate"

run_wavelets::
"ymd": ('<TIME_BEGIN>', '<TIME_END>', '30min')
"dt": 0.05
"period": 60
"dj": 0.25
"max_period": 43200
"overwrite": False
"varstorun": ['co2*w', 'h2o*w', 't_sonic*w']
"mother": "MexicanHat"
"output_path": "<MOTHER>/output/WV/MexicanHat2r/"
"prefix": "<SITE>_WV-{}_{}.flux"
"ignoreerr": True

run_wavelets::readkwargs::
"path": "<MOTHER><RAWPATH>/level_6_good_units/"

consolidate_wavelets::
"ymd": [2023]
"path": "<MOTHER>/output/WV/MexicanHat2r/<SITE>_WV-F{}_{}.flux"
"varstorun": ['co2*w', 'h2o*w', 't_sonic*w']
"output_path": "<MOTHER>/output/WV/<SITE>_WV_{}.30mn.csv"
"verbosity": 1

