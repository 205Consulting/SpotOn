import os
import json
import pandas as pd
from SpotOn import SpotOnAnalysis 
from util import print_status, print_header


def is_spoton_uid (uid):
	"""
		FUNCTION: is_spoton_uid
		-----------------------
		returns true if the "spoton_uid" is true 
	"""
	return (uid[-10:] == "@spoton.it")


if __name__ == "__main__":

	######[ FILENAMES	]#####
	ce_json_path = '../data/json/calendars_0_till_1000000.json'
	a_json_path = '../data/json/es-activity.json'


	##########################################################################################
	#########################[ Calendar Events ]##############################################
	##########################################################################################
	print_header ("LOADING CALENDAR EVENTS DATAFRAME")

	#=====[ Step 1: load ce json	]=====
	print_status ("Initialization", "Loading calendar events JSON: " + ce_json_path)
	ce_json = json.load(open(ce_json_path, 'r'))

	#=====[ Step 2: put into pandas dataframe	]=====
	print_status ("Initialization", "Converting to Pandas dataframe")
	ce_json = ce_json['hits']['hits']
	ce_json = [c['_source'] for c in ce_json]
	ce_df = pd.DataFrame (ce_json)

	#=====[ Step 3: put into SpotOnAnalysis object	]=====
	so = SpotOnAnalysis ()
	so.calendar_events_df = ce_df;

	#=====[ Step 4: apply reformatting operations	]=====
	print_status ("Reformatting", "Dropping unnecessary columns")
	so.drop_spurious_columns_ce ()

	print_status ("Reformatting", "Filtering for location")
	so.filter_location_ce ();

	print_status ("Reformatting", "tokenizing the name/description, removing stopwords")
	so.clean_name_description_ce ()

	#=====[ Step 5: update ce_df	]=====
	print_status ("Selecting", "Getting only spoton events out")
	ce_df = so.calendar_events_df
	spoton_boolvec = ce_df['uid'].apply(is_spoton_uid)
	spoton_ce_df = ce_df[spoton_boolvec]





	##########################################################################################
	#########################[ Activities ]###################################################
	##########################################################################################
	# print_header ("LOADING ACTIVITIES DATAFRAME")

	# #=====[ Step 1: load a json	]=====
	# print_status ("Initialization", "Loading activities JSON: " + a_json_path)
	# a_json = json.load(open(a_json_path, 'r'))

	# #=====[ Step 2: put into pandas dataframe	]=====
	# print_status ("Initialization", "Converting to Pandas dataframe")
	# a_json = a_json['hits']['hits']
	# for a in a_json:
	# 	a['_source']['_id'] = a['_id']
	# a_json = [a['_source'] for a in a_json]
	# a_df = pd.DataFrame (a_json)
	# print a_df.iloc[0]

	# #=====[ Step 3: put into SpotOnAnalysis object	]=====
	# so = SpotOnAnalysis ()
	# so.activities_df = a_df;

	# #=====[ Step 4: apply reformatting operations	]=====
	# print_status ("Reformatting", "Dropping unnecessary columns")
	# so.drop_spurious_columns_a ()

	# print_status ("Reformatting", "Filtering for location")
	# so.filter_location_a ()

	# print_status ("Reformatting", "tokenizing/cleaning name")
	# so.clean_name_words_a ()

	# #=====[ Step 5: update a_df	]=====
	# a_df = so.activities_df


	# print_header("LOADING COMPLETE")
	# print "ce_df: ALL calendar events dataframe"
	# print "spoton_ce_df: spoton events dataframe"
	# print "a_df: activities dataframe"
	# print "(a_df.iloc[index]['labels'] are the 'tags' for the event)"



