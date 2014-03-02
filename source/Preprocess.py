import os
import sys
import json
import pickle
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from StorageDelegate import StorageDelegate
from util import print_header, print_status, is_json, is_dataframe



####################################################################################################
######################[ --- PARAMETERS--- ]#########################################################
####################################################################################################

us_timezones = set([	
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

ce_retain_cols = [	
					'dates', 
					'description', 
					'id', 
					'location', 
					'name', 
					'type', 
					'response', 
					'uid', 
					'user'
				]

a_retain_cols = [	
					'description', 
					'id', 
					'location', 
					'name', 
					'type', 
					'response', 
					'uid', 
					'user'
				]






####################################################################################################
######################[ --- CLASS DEFINITION --- ]##################################################
####################################################################################################

class Preprocess:


	def __init__ (self):
		"""
			PUBLIC: constructor
			-------------------
		"""
		self.tokenizer = RegexpTokenizer(r'\w+')


	def preprocess_ce(self, ce):
		"""
			PUBLIC: preprocess_ce
			---------------------
			given an object representing calendar events (either json or
			pandas dataframe), this will return a correctly-formatted version
		"""
		#=====[ Step 1: ce -> dataframe representation	]=====
		df = self.get_dataframe_rep (ce)

		#=====[ Step 2: apply formatting operations	]=====
		print_status ("preprocess_ce", "dropping unnecessary columns")
		df = self.retain_columns (df, ce_retain_cols)

		print_status ("preprocess_ce", "reformatting location")
		df = self.reformat_location (df)

		print_status ("preprocess_ce", "filtering by location")
		df = self.filter_location (df)

		print_status ("preprocess_ce", "reformatting dates")
		df = self.reformat_date (df)

		print_status ("preprocess_ce", "reformatting name")
		df = self.reformat_name (df)

		print_status ("preprocess_ce", "reformatting description")
		df = self.reformat_description (df)

		return df


	def preprocess_a (self, a):		
		"""
			PUBLIC: preprocess_a
			--------------------
			given an object representing activities (either json or
			pandas dataframe), this will return a correctly-formatted version
		"""
		#=====[ Step 1: a -> dataframe representation	]=====
		df = self.get_dataframe_rep (a)

		#=====[ Step 2: apply formatting operations	]=====
		print_status ("preprocess_a", "dropping unnecessary columns")
		df = self.retain_columns (df, ce_retain_cols)

		print_status ("preprocess_a", "reformatting location")
		df = self.reformat_location (df)

		print_status ("preprocess_a", "filtering by location")
		df = self.filter_location (df)

		print_status ("preprocess_a", "reformatting name")
		df = self.reformat_name (df)

		return df










	####################################################################################################
	######################[ --- PRIVATE --- ]###########################################################
	####################################################################################################

	def get_dataframe_rep (self, obj):
		"""
			PRIVATE: get_dataframe_rep
			--------------------------
			given an object, returns pandas dataframe version of it.
			(raises TypeError if it is neither pandas dataframe nor json)
		"""
		#=====[ Step 1: ce -> dataframe representation	]=====
		if type(obj) == type({}):
			print_status ("Preprocess", "converting to pandas dataframe format")
			df = self.raw_json_to_df (obj)
			del obj
		elif type(obj) == type(pd.DataFrame ({})):
			df = obj
		else:
			raise TypeError ("Format for calendar events not accepted: " + str(type(ce)))
		return df


	def raw_json_to_df (self, json_rep):
		"""
			PRIVATE: raw_json_to_df 
			-----------------------
			given a raw json file (as received from SpotOn elasticseach indices),
			this will return a pandas dataframe 
		"""
		return pd.DataFrame([j['_source'] for j in json_rep['hits']['hits']])


	def retain_columns (self, df, retain_cols):
		"""
			PRIVATE: retain_columns
			-----------------------
			given a list of column names and a dataframe, drops all others
			and returns the modified dataframe
		"""
		current_cols = list(df.columns)
		drop_cols = list(set(current_cols).difference (set(retain_cols)))
		return df.drop (drop_cols, 1)


	def reformat_location(self, df):
		"""
			PRIVATE: reformat_location
			--------------------------
			converts 'location' column to storing just the timezone 
		"""
		def extract_timezone (location):
			if type(location) == type({}):
				if 'tz' in location:
					return location['tz']
			return None

		df['location'] = df['location'].apply(extract_timezone)
		return df


	def filter_location (self, df):
		"""
			PRIVATE: filter_timezone 
			------------------------
			returns a version of the dataframe with only those entries with 
			timezones falling in 'us_timezones'
		"""
		valid_timezones_boolvec = df['location'].apply (lambda x: x in us_timezones)
		return df[valid_timezones_boolvec]


	def reformat_date (self, df):
		"""
			PRIVATE: reformat_time
			----------------------
			converts to a more compact representation of time 
		"""
		df['date_start'] = df['dates'].apply (lambda x: x['start'], 1)
		df['date_end'] = df['dates'].apply (lambda x: x['end'], 1)
		return df.drop ('dates', 1)


	def tokenize_remove_stopwords (self, text):
		"""
			PRIVATE: tokenize_remove_stopwords
			----------------------------------
			given a string s, tokenizes, converts to lower case, removes stopwords
		"""
		if type(text) == type('') or type(text) == type(u''):
			return [w.lower() for w in self.tokenizer.tokenize(text) if not w.lower() in stopwords.words('english')]


	def reformat_name (self, df):
		"""
			PRIVATE: reformat_name 
			----------------------
			converts the given dataframe's 'name' column so that it is tokenized, 
			lowercase, no stopwords 
		"""
		df['name'] = df['name'].apply (self.tokenize_remove_stopwords)
		return df


	def reformat_description (self, df):
		"""
			PRIVATE: reformat_description
			-----------------------------
			converts the given dataframe's 'description' column so that it is tokenized, 
			lowercase, no stopwords 
		"""
		df['description'] = df['description'].apply(self.tokenize_remove_stopwords)
		return df
