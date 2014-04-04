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
import string
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
					'source',
					'likedBy',
					'langs',
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
			NOTE: after this call, it will be identical to the product of 
			preprocess_a
		"""
		# print_status ("preprocess_ce", "converting to dataframe representation")
		df = self.get_dataframe_rep (ce)

		# print_status ("preprocess_ce", "dropping unnecessary columns")
		df = self.retain_columns (df, ce_retain_cols)

		# print_status ("preprocess_ce", "reformatting location")
		df = self.reformat_location (df)

		# print_status ("preprocess_ce", "filtering by location")
		df = self.filter_location (df)

		# print_status ("preprocess_ce", "reformatting name")
		df = self.reformat_natural_language_column (df, 'name')

		# print_status ("preprocess_ce", "reformatting description")
		df = self.reformat_natural_language_column (df, 'description')
		df['words'] = df['description']
		df = df.drop ('description', axis=1)

		return df


	def preprocess_a (self, a):		
		"""
			PUBLIC: preprocess_a
			--------------------
			given an object representing activities (either json or
			pandas dataframe), this will return a correctly-formatted version
			NOTE: after this call, it will be identical to the product of 
			preprocess_ce
		"""
		# print_status ("preprocess_a", "converting to dataframe representation")
		df = self.get_dataframe_rep (a)

		# print_status ("preprocess_a", "dropping unnecessary columns")
		df = self.retain_columns (df, a_retain_cols)

		# print_status ("preprocess_a", "reformatting location")
		df = self.reformat_location (df)

		# print_status ("preprocess_a", "filtering by location")
		df = self.filter_location (df)

		# print_status ("preprocess_a", "reformatting name")
		df = self.reformat_natural_language_column (df, 'name')

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
		#=====[ Case: single json dict	]=====
		if type(obj) == dict:
			obj = [obj]

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


	def reformat_location (self, df):
		"""
			PRIVATE: reformat_location
			--------------------------
			takes the 'location' field and expands it
		"""
		#=====[ Step 1: get location dataframe	]=====
		location_df = pd.DataFrame(list(df['location']))
		if 'name' in location_df:
			location_df['location_name'] = location_df['name']
			location_df = location_df.drop ('name', axis=1)
		df = df.drop ('location', axis=1)

		#=====[ Step 2: merge the two	]=====
		return pd.concat ([df, location_df], axis=1)


	def filter_location (self, df):
		"""
			PRIVATE: filter_location
			------------------------
			returns a version of the dataframe with only those entries with 
			timezones falling in 'us_timezones'
		"""
		pattern = r'(America|US|UTC)'
		america_ix = df['tz'].str.contains (pattern, na=False)
		return df.ix[america_ix]


	def reformat_date (self, df):
		"""
			PRIVATE: reformat_time
			----------------------
			converts to a more compact representation of time 
		"""
		df['date_start'] = df['dates'].apply (lambda x: x['start'], 1)
		df['date_end'] = df['dates'].apply (lambda x: x['end'], 1)
		return df.drop ('dates', 1)


	def reformat_natural_language_column (self, df, colname):
		"""
			PRIVATE: reformat_natural_language_column
			-----------------------------------------
			given a column that contains natural language,
			this will reformat it to be a tokenized list 
			of lowercase words, exluding punctation, stopwords
			and those with numbers
		"""
		#=====[ Step 1: convert to lowercase	]=====
		df[colname] = df[colname].str.lower ()

		#=====[ Step 2: remove punctuation (TODO: remove words with numbers too)	]=====
		punctuation = '[^\w\s]'
		df[colname] = df[colname].str.replace(punctuation, '')

		#=====[ Step 3: split on spaces	]=====
		df[colname] = df[colname].str.split ()

		#=====[ Step 4: remove stopwords/words with digits	]=====
		sw = set(stopwords.words('english'))
		_digits = re.compile('\d')
		def contains_digits(word):
			return bool(_digits.search(word))
		def is_valid (word):
			if word in sw:
				return False
			if not contains_digits(word):
				return True
			return False
		def reformat_words (words):
			if type(words) == list:
				return [w for w in words if is_valid(w)]
			else:
				return []
		df[colname] = df[colname].apply (reformat_words)
		return df