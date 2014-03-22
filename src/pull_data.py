# Script: pull_data
# -----------------
# (optionally) specify a directory and this will pull data 
# into it from SpotOn elasticsearch indices, formatting it 
# as a pandas dataframe as it goes. this will pull 1 million 
# in about 1 hour using Stanford campus' internet connection.
import sys
import os
import json
import requests
import parameters
import argparse
import time
import pandas as pd
from Preprocess import Preprocess
from StorageDelegate import StorageDelegate
from util import print_header, print_error, print_status, print_notification

#==========[ elasticsearch access parameters	]==========
elasticsearch_username = 'ml'
elasticsearch_password = '9g*V(7fB0+mc8Lz(7Z!a'
elasticsearch_activities_endpoint = 'https://spoton.it/api/sources/activities'
elasticsearch_calendars_endpoint = 'https://spoton.it/api/sources/calendars'


#==========[ Data storage parameters	]==========
storage_base_dir = 	{
						'calendars':	'../data/pandas/calendars',
						'activities':	'../data/pandas/activities'
					}
storage_chunk_size = 100000
pre = Preprocess ()



def pull_calendars (pull_start, chunk_size):
	"""
		Function: pull_calendars
		------------------------
		pulls from index 'start' for chunk_size entries,
		returns them in a well-formatted dataframe
	"""
	print_status ("Pulling calendars", "%d to %d" % (pull_start, pull_start + chunk_size))

	#=====[ Step 1: get the data	]=====
	params = {'from':pull_start, 'to':pull_start + chunk_size, 'size':500}
	request = requests.get(elasticsearch_calendars_endpoint, auth=(elasticsearch_username, elasticsearch_password), params=params)

	#=====[ Step 2: preprocess it	]=====
	data_df = pre.preprocess_ce (request.json ())
	return data_df


def pull_activities (pull_start, chunk_size):
	"""
		Function: pull_activities
		-------------------------
		pulls from index 'start' for chunk_size entries,
		returns them in a well-formatted dataframe
	"""
	print_status ("Pulling activities", "%d to %d" % (pull_start, pull_start + chunk_size))

	#=====[ Step 1: get the data	]=====
	params = {'from':pull_start, 'to':pull_start + chunk_size, 'size':500}
	request = requests.get(elasticsearch_activities_endpoint, auth=(elasticsearch_username, elasticsearch_password), params=params)

	#=====[ Step 2: preprocess it	]=====
	data_df = pre.preprocess_a (request.json ())
	return data_df


def parse_arguments (argv):
	"""
		Function: parse_arguments 
		-------------------------
		returns arguments 
	"""
	#=====[ Step 1: construct argparser	]=====
	parser = argparse.ArgumentParser()
	parser.add_argument("endpoint", 	help="which endpoint to access ('calendars' or 'activities')")
	parser.add_argument("start", 		help="start index", type=int)
	parser.add_argument("upper_limit", 	help="number of items to pull",type=int)
	args = parser.parse_args()

	#=====[ Step 2: find the endpoint	]=====
	endpoint = args.endpoint
	if not ((endpoint == 'calendars') or (endpoint == 'activities')):
		print_error ("incorrect endpoint entered", "should be 'calendars' or 'activities'")

	#=====[ Step 3: set pull_data	]=====
	if endpoint == 'calendars':
		pull_data = pull_calendars
	elif endpoint == 'activities':
		pull_data = pull_activities

	#=====[ Step 4: set start/upper limit 	]=====
	start = args.start
	upper_limit = args.upper_limit

	return endpoint, pull_data, start, upper_limit



def get_save_name (endpoint, pull_end):
	"""
		Function: get_save_name
		-----------------------
		given the finish index, returns a string that 
		can serve as a filename for it 
	"""
	return 'up_to_' + str(pull_end) + '.df'


if __name__ == "__main__":

	#==========[ Step 1: parse arguments	]==========
	endpoint, pull_data, start, upper_limit = parse_arguments (sys.argv)
	print_header ("Pulling data from " + endpoint + ": (start, upper_limit) = " + str(start) + ", " + str(upper_limit))


	#==========[ Step 2: set up storage delegate	]==========
	storage_delegate = StorageDelegate ()


	#==========[ PULL DATA]==========
	pull_size = 500
	data_dfs = []
	num_pulled = 0
	while num_pulled < upper_limit:

		#=====[ Step 1: set pull parameters	]=====
		pull_start = start + num_pulled

		#=====[ Step 2: try to pull the data	]=====
		try:
			data_df = pull_data (start + num_pulled, pull_size)
			data_dfs.append (data_df)
			num_pulled += 500


		#=====[ Step 3: handle keyboard interrupts ]=====
		except KeyboardInterrupt:
			print "keyboard interrupt"
			break


		#=====[ Step 4: handle other exceptions (API screws up, etc) ]=====
		except Exception as e:
			print "Non keyboardInterrupt exception detected: %s" % e
			continue


		#=====[ Step 5: save data if the size is right, forget it	]=====
		if len(data_dfs) >= 100:

			#=====[ concatenate	]=====
			df = pd.concat (data_dfs, axis=0)
			if len(df) == 0:
				break

			#=====[ save	]=====
			save_name = get_save_name(endpoint, start + num_pulled)
			print_notification ("Saving dataframe as " + save_name)
			print df
			if endpoint == 'calendars':
				storage_delegate.save_calendar_df (df, save_name)
			elif endpoint == 'activities':
				storage_delegate.save_activity_df (df, save_name)
			print '\n\n'

			#=====[ clear memory	]=====
			del df
			data_dfs = []



