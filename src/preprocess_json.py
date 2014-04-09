# Script: preprocess_json
# -----------------------
# script to convert raw json of activities to 
# a large pandas dataframe. call as 'ipython -i preprocess_json.py'
# if you want to pickle or play with the resulting
# dataframe afterwards
import pickle
import json
import time
from Preprocess import Preprocess
from util import *

if __name__ == "__main__":
	print_header ("PREPROCESS JSON - convert raw json to dataframe")
	start_time = time.time ()

	#=====[ Step 1: create preprocess	]=====
	pre = Preprocess ()

	#=====[ Step 2: load json into memory	]=====
	print_status ("preprocess_json", "loading json into memory")
	json_filename = '/Users/jayhack/Dropbox/Data/Spoton/activities_0_till_168694.json'
	a_json = json.load (open(json_filename, 'r'))

	#=====[ Step 3: apply Preprocess to it	]=====
	print_header ('PREPROCESSING JSON')
	a_df = pre.preprocess_a (a_json)

	######[ PRINT ELAPSED TIME	]#####
	end_time = time.time ()
	print_notification ('Elapsed time: ' + str(end_time - start_time))