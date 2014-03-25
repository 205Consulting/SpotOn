# Class: Preprocess
# -----------------
# class responsible for formatting data in a way such
# that higher-level analysis can take place on it
import os
import sys
import re
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
					'words', 
					'id', 
					'location', 
					'name', 
					'type', 
					'response', 
					'uid', 
					'user',
					'description'
					#need something for date/time in here
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
		# print_status ("preprocess_a", "converting to dataframe representation")
		df = self.get_dataframe_rep (a)

		#=====[ Step 2: apply formatting operations	]=====
		# print_status ("preprocess_a", "dropping unnecessary columns")
		df = self.retain_columns (df, a_retain_cols)

		# print_status ("preprocess_a", "reformatting location")
		df = self.reformat_location (df)

		# print_status ("preprocess_a", "filtering by location")
		df = self.filter_location (df)

		# print_status ("preprocess_a", "reformatting name")
		df = self.reformat_name (df)

		# print_status ("preprocess_a", "reformatting words")
		df = self.reformat_words(df)

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
		#=====[ Case: list of json dicts	]=====
		if type(obj) == list and '_source' in obj[0].keys ():
			return pd.DataFrame([x['_source'] for x in obj])
		elif type(obj) == list and not '_source' in obj[0].keys ():
			return pd.DataFrame(obj)

		#=====[ Step 1: json -> dataframe representation	]=====
		if type(obj) == type({}):
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
		if 'hits' in json_rep.keys ():
			return pd.DataFrame([j['_source'] for j in json_rep['hits']['hits']])
		else:
			return pd.DataFrame (json_rep)


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

		if 'location' in df:
			df['location'] = df['location'].apply(extract_timezone)
		return df


	def filter_location (self, df):
		"""
			PRIVATE: filter_timezone 
			------------------------
			returns a version of the dataframe with only those entries with 
			timezones falling in 'us_timezones'
		"""
		if 'location' in df:
			valid_timezones_boolvec = df['location'].apply (lambda x: x in us_timezones)
			return df[valid_timezones_boolvec]
		return df


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
			given a string s:
			- tokenizes 
			- converts to lower case
			- removes stopwords
			- removes words with digits, 
		"""
		#=====[ Function: indicator for containing digits	]=====
		_digits = re.compile('\d')
		def contains_digits(word):
			return bool(_digits.search(word))
		
		#=====[ Function: indicator for valid words to include	]=====
		def is_valid (word):
			if len(word) > 1:
				if not word in stopwords.words ('english'):
					if not contains_digits(word):
						return True
			return False

		if type(text) == type('') or type(text) == type(u''):
			return [w.lower() for w in self.tokenizer.tokenize(text) if is_valid(w.lower())]
		if type(text) == list:
			return [w for w in text if is_valid(w)]


	def reformat_name (self, df):
		"""
			PRIVATE: reformat_name 
			----------------------
			converts the given dataframe's 'name' column so that it is tokenized, 
			lowercase, no stopwords 
		"""
		if 'name' in df:
			df['name'] = df['name'].apply (self.tokenize_remove_stopwords)
		return df


	def reformat_description (self, df):
		"""
			PRIVATE: reformat_description
			-----------------------------
			converts the given dataframe's 'description' column so that it is tokenized, 
			lowercase, no stopwords 
		"""
		if 'description' in df:
			df['description'] = df['description'].apply(self.tokenize_remove_stopwords)
		return df


	def reformat_words (self, df):
		"""
			PRIVATE: reformat_words 
			-----------------------
			converts the given dataframe's 'words' column so that it is tokenized,
			lowercase, not stopwords 
		"""
		if 'description' in df:
			df['words'] = df['description'].apply (self.tokenize_remove_stopwords)
		else:
			df['words'] = df['words'].apply (self.tokenize_remove_stopwords)
		return df