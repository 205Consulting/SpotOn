import os
import math
import numpy as np
import pandas as pd
from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import stopwords
from util import *


class SpotOnAnalysis:


	#==========[ Parameters	]==========
	filenames = {
					'data_dir':				os.path.join(os.getcwd(), '../data'),
					'calendar_events_df':	os.path.join (os.getcwd (), '../data/calendar_events.df'),
					'activities_df':		os.path.join (os.getcwd(), '../data/activities.df')
				}

	valid_timezones = set([	
							u'America/Anchorage',
							u'America/Chicago',
							u'America/Dawson',
							u'America/Dawson_Creek',
							u'America/Denver',
							u'America/Detroit',
							u'America/Edmonton',
							u'America/Halifax',
							u'America/Indiana/Indianapolis',
							u'America/Los_Angeles',
							u'America/Montreal',
							u'America/New_York',
							u'America/Phoenix',
							u'America/Toronto',
							u'America/Vancouver',
							u'America/Whitehorse',
							u'America/Winnipeg',
							u'US/Eastern',
							u'US/Pacific',
							u'UTC'
						])



	def __init__ (self):
		"""
			PUBLIC: Constructor
			-------------------
			loads in the data 
		"""
		print_welcome ()

		#=====[ Step 1: load in dataframes	]=====
		self.load_calendar_events_df ()
		# self.load_activities_df ()

		
	def load_calendar_events_df (self):
		"""
			PRIVATE: load_calendar_events_df
			--------------------------------
			loads in the calendar events dataframe 
		"""
		print_status ("load_calendar_events_df", "loading...")
		self.calendar_events_df = pd.read_pickle (self.filenames['calendar_events_df'])
		print_status ("load_calendar_events_df", "complete")


	def load_activities_df (self):
		"""
			PRIVATE: load_activities_df
			--------------------------------
			loads in the calendar events dataframe 
		"""
		print_status ("load_activities_df", "loading...")
		self.activities_df = pd.read_pickle (self.filenames['activities_df'])
		print_status ("load_activities_df", "complete")


		
	####################################################################################################
	######################[ --- DATAFRAME PREPROCESSING --- ]###########################################
	####################################################################################################

	def filter_location_ce (self):
		"""
			PRIVATE: filter_location_ce
			---------------------------
			removes all rows from self.calendar_events_df that do not have a timezone
			listed in self.valid_timezones
		"""
		def is_valid_timezone (row):
			if 'tz' in row['location']:
				return (row['location']['tz'] in self.valid_timezones)
			return False

		#=====[ Step 1: add column for 'valid timezone'	]=====
		self.calendar_events_df['valid_timezone'] = self.calendar_events_df.apply (is_valid_timezone, 1)

		#=====[ Step 2: select only those with a valid timezone	]=====
		self.calendar_events_df = self.calendar_events_df[self.calendar_events_df['valid_timezone'] == True]


	def tokenize_remove_stopwords (self, s):
		"""
			PRIVATE: tokenize_remove_stopwords
			----------------------------------
			given a string s, returns a tokenized version with stopwords 
			removed 
		"""
		return [w for w in wordpunct_tokenize(s) if not w in stopwords.words('english')]


	def tokenize_name_ce (self):
		"""
			PRIVATE: tokenize_name
			----------------------
			adds a column for tokenized name; also removes stopwords
		"""
		def clean_name (row):
			return self.tokenize_remove_stopwords(row['name'])

		self.calendar_events_df['name'] = self.calendar_events_df.apply (clean_name)




	####################################################################################################
	######################[ --- CONSTRUCTING USERS --- ]################################################
	####################################################################################################

	# ===================
	# USER REPRESENTATION
	# ===================
	# id: user's id
	# event_ids: list of ids of events they have attended
	# event_names: list of all names
	# event_descriptions: list of all descriptions
	# event_times: start times
	# event_locations: currently just timezone
	# event_responses: response (yes/no)


	def init_user_representation (self, user_id):
		"""
			PRIVATE: init_user_representation
			---------------------------------
			given a user id, this returns an empty (initialized) dict
			to represent him or her
		"""
		return {
					'event_ids': [],
					'event_names': [],
					'event_descriptions': [],
					'event_times': [],
					'event_locations': [],
					'event_responses': [],
				}


	def update_user_representation (self, event):
		"""
			PRIVATE: update_user_representation
			-----------------------------------
			given a calendar event (row in self.calendar_events_df), this function 
			will update self.user_representations
		"""
		#=====[ Step 1: retrieve user representation	]=====
		user_id = event['user']
		user_rep = self.user_representations[user_id]

		#=====[ Step 3: add relevant data	]=====
		user_rep['event_ids'].append (event['id'])
		user_rep['event_names'].append (event['name'])					# Note: tokenize this?
		user_rep['event_descriptions'].append (event['description'])	# Note: tokenize this?
		user_rep['event_times'].append (event['dates']['start'])		# Note: only considering start time
		user_rep['event_locations'].append (event['location'])			# Note: most are timezone
		user_rep['event_responses'].append (event['response'])				# Note: only considering start time		


	def construct_users_df (self):
		"""
			PRIVATE: constrctuct_users_df
			-----------------------------
			constructs/returns a pandas df with each row as a user 
			given the dataframe on calendar events 
		"""

		#=====[ Step 1: sort the dataframe by user id	]=====
		self.calendar_events_df = self.calendar_events_df.sort ('user')

		#=====[ Step 2: init user_representations	]=====
		unique_user_ids = [uid for uid in self.calendar_events_df['user'].unique ()]
		self.user_representations = {user_id: self.init_user_representation (user_id) for user_id in unique_user_ids}

		#=====[ Step 3: update user_representations for each row in dataframe  	]=====
		self.calendar_events_df.apply (self.update_user_representation, axis=1)





	####################################################################################################
	######################[ --- TEXT PREPROCESSING --- ]################################################
	####################################################################################################














