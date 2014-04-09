# Class: SpotOn
# -------------
# Main class for inference and analysis on SpotOn calendar event
# data. Utilizes StorageDelegate, Preprocess, SemanticAnalysis,
# UserAnalysis and Inference objects. 
#=====[ inference ]=====
import numpy as np
import pandas as pd
import gensim

#=====[ our modules	]=====
from StorageDelegate import StorageDelegate
from Preprocess import Preprocess
from SemanticAnalysis import SemanticAnalysis
from UserAnalysis import UserAnalysis
from Inference import Inference
from util import *

class SpotOn:

	def __init__ (self):
		"""
			PUBLIC: Constructor
			-------------------
			constructs member objects
		"""
		#=====[ Step 1: create member objects	]=====
		self.preprocess = Preprocess ()
		self.storage_delegate = StorageDelegate ()
		self.semantic_analysis = SemanticAnalysis ()
		self.user_analysis = UserAnalysis ()
		self.inference = None
		self.activities_corpus = None


	def load (self):
		"""
			PUBLIC: load 
			------------
			loads in all parameters 
		"""
		#=====[ Step 1: load in semantic analysis	]=====
		print_status ("Initialization", "Loading ML parameters (Begin)")
		self.semantic_analysis.load ()
		print_status ("Initialization", "Loading ML parameters (End)")		

		#=====[ Step 2: transfer over models to inference	]=====
		print_status ("Initialization", "Constructing Inference instance (Begin)")
		self.inference = Inference (self.semantic_analysis.lda_model, self.semantic_analysis.lda_model_topics)
		print_status ("Initialization", "Constructing Inference instance (End)")







	####################################################################################################
	######################[ --- Getting Users --- ]#####################################################
	####################################################################################################

	def get_users (self):
		"""
			PUBLIC: get_users
			-----------------
			constructs self.u_df from all available 
			calendar dataframes 
		"""
		self.u_df = self.user_analysis.extract_users (self.storage_delegate.iter_calendar_dfs)
		self.u_df = self.semantic_analysis.analyze (self.u_df, 'all_event_names')


	def load_users (self, filepath='../data/pandas/users/users.df'):
		"""
			PUBLIC: load_users
			------------------
			constructs self.u_df from a saved file
		"""
		self.u_df = pd.read_pickle(filepath)











	####################################################################################################
	######################[ --- Training  --- ]#########################################################
	####################################################################################################

	def extract_text (self, activity_row):
		"""
			PRIVATE: extract_text
			---------------------
			given a row representing an activity, this returns 
			a list of words representing it as a 'text'
		"""
		text = []
		if type(activity_row['name']) == list:
			text += activity_row['name']
		if type(activity_row['words']) == list:
			text += activity_row['words']
		return text


	def get_corpus_dictionary (self):
		"""
			PRIVATE: get_corpus_dictionary
			------------------------------
			Assembles a gensim corpus and dictionary from activities_df,
			where each text is name || words.
		"""
		#=====[ Step 1: iterate through all activity dataframes	]=====
		print_status ("get_corpus", "assembling texts")
		documents = []
		for df in self.storage_delegate.iter_activity_dfs ():
			df['lda_doc'] = df['name'] + df['words']
			documents += list(df['lda_doc'])

		#=====[ Step 2: get dictionary	]=====
		print_status ("get_corpus", "assembling dictionary")
		dictionary = gensim.corpora.Dictionary(documents)

		#=====[ Step 3: get corpus	]=====
		print_status ("get_corpus", "assembling corpus")		
		corpus = [dictionary.doc2bow (d) for d in documents]

		return corpus, dictionary


	def print_lda_topics (self):
		"""
			PUBLIC: print_lda_topics
			------------------------
			prints out a representation of the lda topics found in self.semantic_analysis
		"""
		print_header ("LDA TOPICS: ")
		self.semantic_analysis.print_lda_topics ()


	def train_semantic_analysis (self):
		"""
			PUBLIC: train_semantic_analysis
			-------------------------------
			finds parameters for self.semantic_analysis
		"""
		#=====[ Step 1: get the corpus	]=====
		print_status ("train_semantic_analysis", "getting corpus/dictionary")
		corpus, dictionary = self.get_corpus_dictionary ()

		#=====[ Step 2: train ]=====
		print_status ("train_semantic_analysis", "training semantic analysis")
		self.semantic_analysis.train (corpus, dictionary)

		#####[ DEBUG: print out lda topics	]#####
		self.print_lda_topics ()




	####################################################################################################
	######################[ --- Processing --- ]########################################################
	####################################################################################################

	def activities_json_to_df (self, a_json):
		"""
			PRIVATE: activities_json_to_df
			------------------------------
			given: list of json dicts representing activities 
			returns: dataframe with preprocessing, semantic analysis
		"""
		a_df = self.preprocess.preprocess_a (a_json)
		a_df = self.semantic_analysis.add_lda_vec_column (a_df)
		a_df = self.semantic_analysis.add_w2v_sum_column (a_df)
		return a_df


	def calendar_events_json_to_df (self, ce_json):
		"""
			PRIVATE: calendar_events_json_to_df
			------------------------------
			given: list of json dicts representing calendar events 
			returns: dataframe with preprocessing, semantic analysis
		"""
		ce_df = self.preprocess.preprocess_ce (ce_json)
		ce_df = self.semantic_analysis.add_lda_vec_column (ce_df)
		ce_df = self.semantic_analysis.add_w2v_sum_column (ce_df)
		return ce_df


	def calendar_events_to_user_representation(self, ce_json):
		"""
			PUBLIC: calendar_events_to_user_representation
			----------------------------------------------
			given a list containing json dicts representing calendar events belonging
			to a single user, this will return a representation that can be passed to 
			score_activity_for_user and recommend_for_user
		"""
		user_df 	= self.calendar_events_json_to_df (ce_json)
		lda_vec 	= self.semantic_analysis.get_user_lda_vec (user_df)
		return {'events_df':user_df, 'lda_vec':lda_vec}


	def load_activities_corpus(self, activities):
		'''
			function: load_activities_corpus
			params: activities - list of activities to recommend

			returns: none
			notes: use this function to load a big activities corpus into the SpotOn object, and later when calling
			recommend_for_user we will pull activities to recommend from this corpus.

			Can be called multiple times to update to different activities
		'''
		self.activities_corpus = self.activities_json_to_df (activities)












	####################################################################################################
	######################[ --- Recommending --- ]######################################################
	####################################################################################################

	def score_activity_for_user(self, user_representation, activity):
		"""
			PUBLIC: score_activity_for_user
			-------------------------------
			params: user_representation - representation of the user to score for
								(created by calendar_events_to_user_representation)
					activity - json of the activity to score

			notes: goes from the representation of the user that you use + one activity 
					-> return a score for how much they'd like it
		"""
		#=====[ Step 1: get activity dataframe 	]=====
		activity_df = self.activities_json_to_df ([activity])

		#=====[ Step 2: get scored dataframe	]=====
		activity_df = self.inference.infer_scores (user_representation, activity_df)

		#=====[ Step 3: extract and return score	]=====
		return activity_df.iloc[0]['score']


	def recommend_for_user(self, user_representation, activities=None, topn=10):
		"""
			PUBLIC: recommend_for_user
			--------------------------
			params: user_representation - representation of the user to recommend for
					activities - either a list of json activities, or None if 
									.load_activities_corpus has been called
					topn - number of recommendations to return
		"""
		#=====[ Step 1: get a_df, df of activities to recommend	]=====
		if activities is not None:
			activities_df = self.activities_json_to_df (activities)
		else:
			if not (self.activities_corpus is not None):
				self.load_activities_corpus ()
			activities_df = self.activities_corpus

		#=====[ Step 2: get scores, return sorted	]=====
		activity_ranks = self.inference.rank_activities (user_representation, activities_df)
		return list(activity_ranks)


	def recommend_users_for_activity(self, activity, list_of_users, topn=10):
		"""
			PUBLIC: recommend_users_for_activities
			--------------------------------------
			params: activity - activity to recommend users for
					list_of_users - list of users to filter
					topn - number of users to return

			notes: goes from an activity and a list of users -> topn users for that activity
		"""
		scores = [self.score_activity_for_user(user, activity) for user in list_of_users]
		sorted_ix = np.argsort(scores)[::-1]
		return [list_of_users[sorted_ix[i]] for i in range(topn)]



