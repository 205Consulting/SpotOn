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
from util import print_status, print_inner_status, print_header

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
		texts = []
		for df in self.storage_delegate.iter_activity_dfs ():
			print_inner_status ("assembling texts", "next df")
			texts += list(df.apply(self.extract_text, axis=1))

		#=====[ Step 3: get dictionary	]=====
		print_status ("get_corpus", "assembling dictionary")
		dictionary = gensim.corpora.Dictionary(texts)

		#=====[ Step 4: get corpus	]=====
		print_status ("get_corpus", "assembling corpus")		
		corpus = [dictionary.doc2bow (text) for text in texts]

		return corpus, dictionary



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






	####################################################################################################
	######################[ --- Inference --- ]#########################################################
	####################################################################################################

	def score_activities_old (self, user_activities, recommend_activities):
		"""
			PUBLIC: score_activities
			------------------------
			Given a user and a list of activities, both represented as json, this will return 
			(activities, scores) in a sorted list
		"""
		#=====[ Step 1: preprocess json inputs	]=====
		user_events_df = self.preprocess.preprocess_a (user_activities)
		activities_df = self.preprocess.preprocess_a (recommend_activities)

		#=====[ Step 2: construct a user from user_events_df	]=====
		def f():
			yield user_events_df
		users = self.user_analysis.extract_users (f)
		assert len(users) == 1
		user = users.iloc[0]

		#=====[ Step 3: get scores for each one	]=====
		scores = [inference.score_match (user, activities_df.iloc[i]) for i in range(len(activities_df))]

		#=====[ Step 4: return sorted list of activity, score	]=====
		return sorted(zip(activities_json, scores), key=lambda x: x[1], reverse=True)


	def score_activities (self, user_activities, recommend_activities):
		"""
			PUBLIC: score_activities
			------------------------
			Given a user and a list of activities, both represented as json, this will return 
			(activities, scores) in a sorted list
		"""
		#=====[ Step 1: preprocess user_activities and recommend_activities	]=====
		user_activities = self.preprocess.preprocess_a (user_activities)
		recommend_activities = self.preprocess.preprocess_a (recommend_activities)

		#=====[ Step 2: get scores for each one	]=====
		scores = self.inference.score_activities (user_activities, recommend_activities)
		return scores


	####################################################################################################
	######################[ --- Interface --- ]#########################################################
	####################################################################################################

	def print_lda_topics (self):
		"""
			PUBLIC: print_lda_topics
			------------------------
			prints out a representation of the lda topics found in self.semantic_analysis
		"""
		self.semantic_analysis.print_lda_topics ()






if __name__ == "__main__":

	so = SpotOn ()

	#=====[ Step 1: train semantic analysis	]=====
	# print_header ("Demo Script - Training semantic analysis models")
	# so.train_semantic_analysis ()

	#=====[ Step 2: save semantic analysis models	]=====
	# print_header ("Demo Script - Saving semantic analysis models")
	# so.semantic_analysis.save ()

	#=====[ Step 3: load semantic analysis models	]=====
	print_header ("Demo Script - Loading semantic analysis models")	
	so.semantic_analysis.load ()

	#=====[ Step 4: load users	]=====
	print_header ("Demo Script - Getting users")	
	# so.get_users ()
	so.load_users ()


	#=====[ Step 3: apply to activity dfs	]=====
	print_header ("Demo Script - Performing semantic analysis on activities")
	for adf in so.storage_delegate.iter_activity_dfs ():

		#=====[ Semantic analysis on adf	]=====
		adf = so.semantic_analysis.add_semantic_summary (adf, 'name')

		#=====[ Construct inference	]=====
		inf = Inference (so.semantic_analysis.lda_model, so.semantic_analysis.lda_model_topic_dist)

		#=====[ Recommend	]=====
		print inf.recommend (so.u_df.iloc[3], adf[:100], 'all_event_names_LDA', 'name', 'all_event_names_W2V', 'name_W2V')






