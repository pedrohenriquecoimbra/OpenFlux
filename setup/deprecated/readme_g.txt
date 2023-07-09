MACHINE READ FILE. PLEASE DO NOT MODIFY SPACES AND LINE JUMPS.

"compaign": "COV3ER"
"co_site": "FR-Gri"
"mother_folder": "C:/Users/phherigcoimb/Desktop/INRAE_longfiles/COV3ER/FR-Gri/"
"raw_input_folder": "C:/Users/phherigcoimb/Desktop/INRAE_longfiles/COV3ER/FR-Gri/input/EC/"
"time_range": ('201606010000', '201607312359', '5min')
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
"output_path": "C:/Users/phherigcoimb/Desktop/INRAE_longfiles/COV3ER/FR-Gri/output/WV/MexicanHat2r/"
"varstorun": ['co2_w', 'h2o_w', 'ch4_w', 'co2_u', 'h2o_u', 'ch4_u']

wv2ec::
"co2_w": "co2_flux"
"ts_w": "H"
"h2o_w": "LE"

readkwargs::
"data": "open_flux"
"path": "C:/Users/phherigcoimb/Desktop/INRAE_longfiles/COV3ER/FR-Gri/output/raw_datasets/level_6/"

varstorun_loop::co2_w::
"xname": "co2"
"yname": "w"
"prefix": "FR-Gri_WV-Fco2w_{}.flux"

varstorun_loop::h2o_w::
"xname": "h2o"
"yname": "w"
"prefix": "FR-Gri_WV-Fh2ow_{}.flux"

varstorun_loop::ch4_w::
"xname": "ch4"
"yname": "w"
"prefix": "FR-Gri_WV-Fch4w_{}.flux"

varstorun_loop::co2_u::
"xname": "co2"
"yname": "u"
"prefix": "FR-Gri_WV-Fco2u_{}.flux"

varstorun_loop::h2o_u::
"xname": "h2o"
"yname": "u"
"prefix": "FR-Gri_WV-Fh2ou_{}.flux"

varstorun_loop::ch4_u::
"xname": "ch4"
"yname": "u"
"prefix": "FR-Gri_WV-Fch4u_{}.flux"

