# Script: preprocess_json
# -----------------------
# script to convert raw json of activities to 
# a large pandas dataframe. call as 'ipython -i preprocess_json.py'
# if you want to pickle or play with the resulting
# dataframe afterwards
import pickle
import json
from Preprocess import Preprocess
from util import *

if __name__ == "__main__":
	print_header ("PREPROCESS JSON - convert raw json to dataframe")

	#=====[ Step 1: create preprocess	]=====
	pre = Preprocess ()

	#=====[ Step 2: load json into memory	]=====
	print_status ("preprocess_json", "loading json into memory")
	json_filename = '/Users/jayhack/Dropbox/Data/Spoton/activities_0_till_168694 2.json'
	a_json = json.load (open(json_filename))

	#=====[ Step 3: apply Preprocess to it	]=====
	a_df = pre.preprocess_a (a_json)