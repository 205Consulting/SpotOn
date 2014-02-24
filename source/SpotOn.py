import os
import math
import numpy as np
import pandas as pd
from nltk.tokenize import wordpunct_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from util import *


class SpotOnAnalysis:


	#==========[ Parameters	]==========
	filenames = {
					'data_dir':				os.path.join(os.getcwd(), '../data'),
					'calendar_events_df':	os.path.join (os.getcwd (), '../data/calendar_events.df'),
					'activities_df':		os.path.join (os.getcwd(), '../data/activities.df')
					'users_df':				os.path.join (os.getcwd(), '../data/users.df')
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
			loads in all dataframes
		"""
		print_welcome ()

		#=====[ Step 1: setup tokenizer	]=====
		self.tokenizer = RegexpTokenizer(r'\w+')

		#=====[ Step 1: load in dataframes	]=====
		# self.load_calendar_events_df ()
		# self.load_activities_df ()
		self.load_users_df ()

		
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


	def load_users_df (self):
		"""
			PRIVATE: load_users_df
			----------------------
			loads in the users df 
		"""
		print_status ("load_users_df", "loading...")
		self.users_df = pd.read_pickle (self.filenames['users_df'])
		print_status ("load_users_df", "complete")







		
	####################################################################################################
	######################[ --- DATAFRAME PREPROCESSING --- ]###########################################
	####################################################################################################

	def drop_spurious_columns_ce (self):
		"""
			PRIVATE: drop_spurious_columns_ce
			---------------------------------
			drops columns that we don't need to use 
		"""
		self.calendar_events_df = self.calendar_events_df.drop(['updated', 'website', 'raw'], 1)


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


	def reformat_dates (self):
		"""
			PRIVATE: reformat_dates
			-----------------------
			changes the date storage format
		"""
		def extract_start_date (row):
			return row['dates']['start']

		def extract_end_date (row):
			return row['dates']['end']

		self.calendar_events_df['start_date'] = self.calendar_events_df.apply (extract_start_date, 1)
		self.calendar_events_df['end_date'] = self.calendar_events_df.apply (extract_end_date, 1)		
		self.calendar_events_df.drop ('dates', 1)


	def add_self_created (self):
		"""
			PRIVATE: add_self_created
			-------------------------
			adds a column for 'self_created', instead of 
			keeping the creator 
		"""
		def extract_self_created (row):
			if not type(row['creator']) == type({}):
				return False
			if u'self' in row['creator'].keys ():
				return row['creator']['self']
			return False

		self.calendar_events_df['self_created'] = self.calendar_events_df.apply (extract_self_created, 1)
		self.calendar_events_df.drop ('creator', 1)

	
	def tokenize_remove_stopwords (self, s):
		"""
			PRIVATE: tokenize_remove_stopwords
			----------------------------------
			given a string s,
			- tokenizes via nltk.wordpunct_tokenize
			- removes stopwords 
			- converts all words to lowercase 
		"""
		return [w.lower() for w in self.tokenizer.tokenize(s) if not w in stopwords.words('english')]


	def clean_name_description_ce (self):
		"""
			PRIVATE: clean_name_description
			-------------------------------
			applies 'tokenize_remove_stopwords' to both the name column
			and the description column of self.calendar_events
		"""
		cleaned_names = []
		cleaned_descriptions = []

		def clean_name_description (row):

			#=====[ Step 1: retrieve name/description	]=====
			name = row['name']
			description = row['description']

			#=====[ Step 2: get new name	]=====
			name_type = type(name)
			if (name_type == type(u'string')) or (name_type == type('string')):
				new_name = self.tokenize_remove_stopwords(name)
			else:
				new_name = name

			#=====[ Step 3: get new name	]=====
			desc_type = type(description)			
			if (desc_type == type(u'string')) or (desc_type == type('string')): 
				new_description = self.tokenize_remove_stopwords(description)
			else:
				new_description = description

			#=====[ Step 4: add to lists	]=====
			cleaned_names.append (new_name)
			cleaned_descriptions.append (new_description)


		self.calendar_events_df.apply (clean_name_description, 1)
		self.calendar_events_df['name'] = cleaned_names
		self.calendar_events_df['description'] = cleaned_descriptions


	def clean_description_ce (self):
		"""
			PRIVATE: tokenize_name_ce
			-------------------------
			changes 'name' column
			- tokenizes
			- removes stopwords

		"""
		cleaned_names = []

		def clean_name (row):
			name = row['name']
			if type(name) == type(u'string'):
				new_name = tokenize_remove_stopwords(row['name'])
				cleaned_names.append (new_name)
			else:
				cleaned_names.append (name)

		self.calendar_events_df.apply (clean_name, 1)
		self.calendar_events_df['cleaned_names'] = cleaned_names







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
					'id': None,
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
		user_rep['id'] = user_id
		user_rep['event_ids'].append (event['id'])
		user_rep['event_names'].append (event['name'])					# Note: tokenize this?
		user_rep['event_descriptions'].append (event['description'])	# Note: tokenize this?
		# user_rep['event_times'].append (event['dates']['start'])		# Note: only considering start time
		# user_rep['event_locations'].append (event['location'])			# Note: most are timezone
		# user_rep['event_responses'].append (event['response'])				# Note: only considering start time		


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

		#=====[ Step 4: create dataframe from user_representations	]=====
		self.users_df = pd.DataFrame(self.user_representations.values())
		del self.user_representations




	####################################################################################################
	######################[ --- TEXT PREPROCESSING --- ]################################################
	####################################################################################################














