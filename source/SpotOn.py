import os
import math
import numpy as np
from scipy.spatial.distance import cosine, euclidean
from random import random
import pandas as pd
from nltk.tokenize import wordpunct_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from gensim.models.word2vec import Word2Vec
from gensim import corpora 
from util import *


class SpotOnAnalysis:


	#==========[ Parameters	]==========
	filenames = {
					'data_dir':				os.path.join(os.getcwd(), '../data'),
					'calendar_events_df':	os.path.join (os.getcwd (), '../data/calendar_events_huge.df'),
					# 'calendar_events_df':	os.path.join (os.getcwd (), '../data/calendar_events.df'),					
					'activities_df':		os.path.join (os.getcwd(), '../data/activities_huge.df'),
					'users_df':				os.path.join (os.getcwd(), '../data/users.df'),
					'word2vec_model':		os.path.join (os.getcwd(), '../data/word2vec_model')
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
			loads in dataframes
		"""
		# print_welcome ()

		#=====[ Step 1: get filenames	]=====
		# self.filenames['json']['activities'] = activities_json
		# self.filenames['json']['calendar_events'] = calendar_events_json

		#=====[ Step 2: setup preprocessing modules	]=====
		self.tokenizer 		= RegexpTokenizer(r'\w+')
		self.load_word2vec_model ()

		#=====[ Step 1: load in dataframes	]=====
		# self.load_calendar_events_df ()
		# self.load_activities_df ()
		# self.load_users_df ()
		# self.load_word2vec_model ()

		
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


	def load_word2vec_model (self):
		
		self.word2vec_model = Word2Vec.load(self.filenames['word2vec_model'])





		
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


	def reformat_dates_ce (self):
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
		return [w.lower() for w in self.tokenizer.tokenize(s) if not w.lower() in stopwords.words('english')]


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


	def drop_spurious_columns_a (self):
		"""
			PRIVATE: drop_spurious_columns_a
			--------------------------------
			drops columns that we don't need to use 
		"""
		drop_labels = [	'added', 'addedBy', 'alsoAddedBy', 'alsoAddedNum', 
						'channels', 'classified', 'cost', 'cropped', 'doneBy', 
						'facebook', 'hash', 'hidden', 'langs', 'likedBy', 
						'numPeople', 'online', 'originalPicture', 'picture', 
						'quality', 'random', 'scrap', 'secondarySites', 
						'secondaryTag', 'site', 'trained', 'updated', 'user', 
						'website', 'wishlistedBy', 'wishlists', 'dates', 'description']	
		
		self.activities_df = self.activities_df.drop(drop_labels, 1)


	def clean_name_words_a (self):
		"""
			PRIVATE: clean_name_description
			-------------------------------
			applies 'tokenize_remove_stopwords' to both the name column
			and the description column of self.calendar_events
		"""
		cleaned_names = []
		cleaned_words = []

		def clean_name_description (row):

			#=====[ Step 1: retrieve name/description	]=====
			name = row['name']
			words = row['words']

			#=====[ Step 2: get new name	]=====
			name_type = type(name)
			if (name_type == type(u'string')) or (name_type == type('string')):
				new_name = self.tokenize_remove_stopwords(name)
			else:
				new_name = name

			#=====[ Step 3: get new words	]=====
			new_words = words
			if type(words) == type([]):
				new_words = [w for w in words if not w.lower() in stopwords.words('english')]

			#=====[ Step 4: add to lists	]=====
			cleaned_names.append (new_name)
			cleaned_words.append (new_words)


		self.activities_df.apply (clean_name_description, 1)
		self.activities_df['name'] = cleaned_names
		self.activities_df['words'] = cleaned_words

 	
 	def filter_location_a (self):
		"""
			PRIVATE: filter_location_a
			---------------------------
			removes all rows from self.activities_df that do not have a timezone
			listed in self.valid_timezones
		"""
		def is_valid_timezone (row):
			if 'tz' in row['location']:
				return (row['location']['tz'] in self.valid_timezones)
			return False

		#=====[ Step 1: add column for 'valid timezone'	]=====
		self.activities_df['valid_timezone'] = self.activities_df.apply (is_valid_timezone, 1)

		#=====[ Step 2: select only those with a valid timezone	]=====
		self.activities_df = self.activities_df[self.activities_df['valid_timezone'] == True]





	####################################################################################################
	######################[ --- TFIDF --- ]#############################################################
	####################################################################################################

	def add_tfidf (self):
		"""
			add_tfidf
			---------
			for each event on hand, adds a list where ith element 
			is tf.idf of ith word 
		"""
		#=====[ Step 1: get the list of all texts ] =====
		words_texts = self.activities_df['words']
		name_texts = self.activities_df['name']





	####################################################################################################
	######################[ --- WORD2VEC --- ]##########################################################
	####################################################################################################

	def tfidf_word2vec_combo (self):
		"""
			PRIVATE: word2vec_tfidf_combo
			-----------------------------
			adds word vectors from the title together based on 
			tfidfs 
		"""
		self.weighted_word2vecs = []
		def get_weighted_word2vec (row):
			name_words = row['name']
			word_vecs = [self.word2vec_model[w] if w in self.word2vec_model else None for w in name_words]
			name_tfidfs = row['name_tfidf']
			total = np.zeros(200,)
			print name_words
			print name_tfidfs
			for i in range (len(name_tfidfs)):
				if type(word_vecs[i]) != type(None):
					total += name_tfidfs[i][1]*word_vecs[i]
			print total.shape
			self.weighted_word2vecs.append (total)

		self.activities_df.apply (get_weighted_word2vec, 1);


	def find_most_similar_activities (self, activity_index):
		"""
			PRIVATE: get_closest_ww2v
			-------------------------
			given an event, finds the closes events via 
			weighted word 2 vectors 
		"""
		print "=====[ Finding most similar activities to the following ]====="
		name = self.activities_df.iloc[activity_index]['name']
		print ' '.join (name)
		w2v = self.activities_df.iloc[activity_index]['weighted_word2vecs']
		def get_distance(w2v2):
			return euclidean (w2v, w2v2)
		self.activities_df['distances'] = self.activities_df['weighted_word2vecs'].apply (get_distance)
		print "###[ results: (euclidean distance, name) ]###"
		self.activities_df_sorted = self.activities_df.sort ('distances')
		for i in range(100):
			row = self.activities_df_sorted.iloc[i]
			print row['distances'], " | ", " ".join(self.activities_df_sorted.iloc[i]['name']) 
		print "\n"


	def w2v_recommend_for_user (self, user_index):
		"""
			PRIVATE: w2v_recommend_for_user 
			-------------------------------
			given a user, recommends activities for them
		"""
		print "=====[ Finding best activities for user ", user_index, " ]====="
		user_row = self.users_df.iloc[user_index]
		user_events = user_row['event_names']
		

	def w2v_similarity_demo (self):
		"""
			PUBLIC: w2v_similarity_demo
			---------------------------
			given an event, comes up with the most semantically
			similar ones 
		"""
		#=====[ Step 1: load in relevant data	]=====
		self.activities_df = pd.read_pickle('activities_small_weighted_word2vec.df')
		self.load_word2vec_model ()

		#=====[ Step 2: print out all names	]=====
		print "==========[ ACTIVITY NAMES ]=========="
		for i in range(len(self.activities_df)):
			print i, " | ", ' '.join(self.activities_df.iloc[i]['name'])

		print "\n>>> Call find_most_similar_activities on the index you want to query"


	def w2v_recommendation_demo (self):
		"""
			PUBLIC: w2v_recommendation_demo
			-------------------------------
			given a user, comes up with the best events to 
			recommend 
		"""
		#=====[ Step 1: load in all the data	]=====
		self.activities_df = pd.read_pickle('activities_small_weighted_word2vec.df')
		self.load_word2vec_model ()
		self.users_df = pd.read_pickle('../data/users_spoton.df')

		#=====[ Step 2: print out 30 random users	]=====
		for i in range(30):
			random_index = int(random()*370)
			print "###[ USER: ", random_index, " ]###"
			user_row = self.users_df.iloc[random_index]
			num_events = len(user_row['event_names'])
			if num_events < 20:
				for event in user_row['event_names']:
					try:
						print '		', ' '.join(event)
					except:
						pass
			else:
				for i in range(20):
					event = user_row['event_names'][i]
					try:
						print '		', ' '.join(event)
					except:
						pass

		#=====[ Step 3: take in a random user	]=====




	def word2vec_sum (self, word_list):
		"""
			PRIVATE: word2vec_sum
			---------------------
			given a list of words, this returns 
			a 300-dimensional vector representation 
		"""
		if type(word_list) == type([]):
			word_vecs = [self.word2vec_model[w] for w in word_list if w in self.word2vec_model]
			s = np.sum(word_vecs, axis=0)
			return s
		return None


	def add_word2vec_ce (self):
		"""
			PRIVATE: add_word2vec_ce
			------------------------
			for each element in self.calendar_events_df, this adds
			a field 'word2vec' that is the sum of the word vectors 
			for each word in its NAME
		"""
		self.calendar_events_df['word2vec'] = self.calendar_events_df['name'].apply (self.word2vec_sum)


	def add_word2vec_a (self):
		"""
			PRIVATE: add_word2vec_a
			-----------------------
			for each element in self.activities_df, this adds
			a field 'word2vec' that is the sum of the word vectors 
			for each word in its NAME
		"""
		self.activities_df['word2vec'] = self.activities_df['name'].apply (self.word2vec_sum)



	####################################################################################################
	######################[ --- RECOMMENDATION --- ]####################################################
	####################################################################################################

	def recommend_word2vec (self, user_row):
		"""
			PRIVATE: recommend_word2vec
			---------------------------
			given a user_row, this returns a sorted list of event names
		"""
		#=====[ Step 1: get word2vecs for user	]=====
		user_w2vs = user_row['event_word2vecs']

		#=====[ Step 2: get list of all activity word2vecs	]=====
		activity_w2v = self.activities_df['word2vec']

		#=====[ Step 3: get a minimum cosine sim for each activity	]=====
		def best_cosine_sim (activity_w2v):
			if len(activity_w2v.shape) == 0:
				return 1
			minimum = np.min([cosine(activity_w2v, u_w2v) for u_w2v in user_w2vs if type(u_w2v) != type(None)])
			return minimum

		self.cosine_sims = self.activities_df['word2vec'].apply (best_cosine_sim)






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
					'event_word2vecs': []
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
		# user_rep['event_locations'].append (event['location'])		# Note: most are timezone
		# user_rep['event_responses'].append (event['response'])		# Note: only considering start time		
		user_rep['event_word2vecs'].append (event['word2vec'])


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
	######################[ --- GETTING DATASET --- ]###################################################
	####################################################################################################

	def add_spoton_column_ce (self):
		"""
			PRIVATE: get_spoton_activities
			------------------------------
			adds a column to calendar_events indicating if spoton created it 
		"""
		def is_spoton_event (description):
			if type(description) == type([]):
				return ('spoton' in description)
			return False

		self.calendar_events_df['spoton_created'] = self.calendar_events_df['description'].apply (is_spoton_event)


	def get_spoton_user_ids (self):
		"""
			PRIVATE: get_spoton_users
			-------------------------
			returns a list of user ids for people who have 
			participated in a spoton event 
		"""
		spoton_events = self.calendar_events_df[(self.calendar_events_df['spoton_created'] == True)]
		spoton_user_ids = spoton_events['user'].unique ()
		return spoton_user_ids


	def update_user_representation_spoton_only (self, event):
		"""
			PRIVATE: update_user_representation
			-----------------------------------
			given a calendar event (row in self.calendar_events_df), this function 
			will update self.user_representations
		"""
		#=====[ Step 1: retrieve user representation	]=====
		user_id = event['user']
		if not user_id in self.spoton_user_ids:
			return
		user_rep = self.user_representations[user_id]

		#=====[ Step 3: add relevant data	]=====
		user_rep['id'] = user_id
		user_rep['event_ids'].append (event['id'])
		user_rep['event_names'].append (event['name'])					# Note: tokenize this?
		user_rep['event_descriptions'].append (event['description'])	# Note: tokenize this?
		# user_rep['event_times'].append (event['dates']['start'])		# Note: only considering start time
		# user_rep['event_locations'].append (event['location'])		# Note: most are timezone
		# user_rep['event_responses'].append (event['response'])		# Note: only considering start time		
		user_rep['event_word2vecs'].append (event['word2vec'])


	def construct_users_df_spoton_only (self):
		"""
			PRIVATE: constrctuct_users_df_spoton_only
			-----------------------------------------
			constructs/returns a pandas df with each row as a user; only includes 
			users that we know are spoton users
		"""
		#=====[ Step 1: get the ids of users that use spoton	]=====
		self.add_spoton_column_ce ()
		self.spoton_user_ids = self.get_spoton_user_ids ()

		#=====[ Step 2: sort the dataframe by user id	]=====
		self.calendar_events_df = self.calendar_events_df.sort ('user')

		#=====[ Step 2: init user_representations	]=====
		self.user_representations = {user_id: self.init_user_representation (user_id) for user_id in self.spoton_user_ids}

		#=====[ Step 3: update user_representations for each row in dataframe  	]=====
		self.calendar_events_df.apply (self.update_user_representation_spoton_only, axis=1)

		#=====[ Step 4: create dataframe from user_representations	]=====
		self.users_df = pd.DataFrame(self.user_representations.values())
		del self.user_representations


	def construct_dataset (self):
		"""
			PRIVATE: construct_dataset
			--------------------------
			gets a list of user rows/events they went to, user rows 
			and events they *didnt* go to. 
		"""
		all_users = so.users_df























