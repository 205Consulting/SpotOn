# -------------------------------------------------------------------------------- #
# Script: SpotOn
# --------------
# Main script for conducting analysis on the SpotOn data
# -------------------------------------------------------------------------------- #
import numpy as np
import pandas as pd

from parameters import *
from util import *


class SpotOnAnalysis:


	def __init__ (self):
		"""
			PUBLIC: Constructor
			-------------------
			loads in the data 
		"""
		print_welcome ()

		#=====[ Step 1: load in dataframes	]=====
		print_status ("Initialization", "Loading calendar events df")
		self.calendar_events_df = pd.read_pickle (filenames['calendar_events_df'])
		print_status ("Initialization", "Loading activities df")		
		self.activities_df = pd.read_pickle (filenames['activities_df'])

		



	def initialize_user_df (self, user_ids):
		"""
			PRIVATE: initialize_user_df
			---------------------------
			Given: np array of all unique user ids
			Returns: initialized (empty) dataframe to contain user info
		"""
		return pd.DataFrame ({'id':user_ids})


	def extract_users_from_df (self, calendar_df):
		"""
			PRIVATE: extract_users_from_df
			------------------------------
			Given: a dataframe containing calendar events (e.g. self.calendar_df)
			Returns: a dataframe containing a representation of each unique user
		"""
		user_ids = self.get_unique_user_ids (_calendar_df)
		user_df = initialize_user_df (user_ids)


		


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
		user_id = calendar_event['user']
		user_rep = self.user_representations['user_id']

		#=====[ Step 2: add relevant data	]=====
		user_rep['event_ids'].append (event['id'])
		user_rep['event_names'].append (event['name'])					# Note: tokenize this?
		user_rep['event_descriptions'].append (event['description'])	# Note: tokenize this?
		user_rep['event_times'].append (event['dates']['start'])		# Note: only considering start time
		user_rep['event_locations'].append (event['locations']['tz'])	# Note: this is only the timezone
		user_rep['event_responses'].append (event['dates'])		# Note: only considering start time		



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
		unique_user_ids = self.calendar_events_df['user'].unique ()
		self.user_representations = {user_id: self.init_user_representation () for user_id in unique_user_ids}

		#=====[ Step 3: update user_representations for each row in dataframe  	]=====
		self.calendar_events_df.apply (self.update_user_representation, axis=1)



















