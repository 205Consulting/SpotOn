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


	# PUBLIC: constructor
	# -------------------
	# loads in data
	def __init__ (self):
		print_welcome ()

		#==========[ Step 1: load in data ]==========
		print_status ("Initialization", "Loading calendar df")
		self.calendar_df = pd.read_pickle (filenames['calendars_df'])
		print_status ("Initialization", "Loading activity df")		
		self.activity_df = pd.read_pickle (filenames['activity_df'])

		#==========[ Step 2: reformat ]==========
		

	# PRIVATE: get_unique_user_ids
	# ----------------------------
	# Given: a dataframe containing calendar events (e.g. self.calendar_df)
	# Returns: np array of all unique user ids
	def get_unique_user_ids (self, _calendar_df):
		return _calendar_df['user'].unique ()


	# PRIVATE: initialize_user_df
	# ---------------------------
	# Given: np array of all unique user ids
	# Returns: initialized (empty) dataframe to contain user info
	def initialize_user_df (self, user_ids):
		return pd.DataFrame ({'id':user_ids})


	# PRIVATE: extract_users_from_df
	# ------------------------------
	# Given: a dataframe containing calendar events (e.g. self.calendar_df)
	# Returns: a dataframe containing a representation of each unique user
	def extract_users_from_df (self, _calendar_df):
		
		user_ids = self.get_unique_user_ids (_calendar_df)
		user_df = initialize_user_df (user_ids)


		




