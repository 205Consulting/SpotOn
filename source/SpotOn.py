# Class: SpotOn
# -------------
# container class for all others; goes from raw data 
# to predictions for users
import numpy as np
import gensim
from StorageDelegate import StorageDelegate
from Preprocess import Preprocess
from SemanticAnalysis import SemanticAnalysis
from UserAnalysis import UserAnalysis
from util import print_status, print_inner_status, print_header

class SpotOn:

	def __init__ (self):
		"""
			PUBLIC: Constructor
			-------------------
		"""
		#=====[ Step 1: create member objects	]=====
		self.storage_delegate = StorageDelegate ()
		self.semantic_analysis = SemanticAnalysis ()
		self.user_analysis = UserAnalysis ()



	####################################################################################################
	######################[ --- Getting Users --- ]#########################################################
	####################################################################################################


	def get_users (self):
		"""
			PUBLIC: get_users
			-----------------
			constructs self.u_df from all available 
			calendar dataframes 
		"""
		self.u_df = self.user_analysis.extract_users (self.storage_delegate.iter_calendar_dfs)




	####################################################################################################
	######################[ --- Training  --- ]#########################################################
	####################################################################################################

	def get_corpus (self, df_type, colname):
		"""
			PRIVATE: get_corpus
			-------------------
			given 'activities' or 'calendars', assembles a corpus 
			of all of the text fields and returns it 
		"""
		#=====[ Step 1: assert type is calendar or activitie	]=====
		print_status ("get_corpus", "parsing args")
		assert (df_type == 'calendars' or df_type =='activities')
		if df_type == 'calendars':
			iter_function = self.storage_delegate.iter_calendar_dfs
		elif df_type == 'activities':
			iter_function = self.storage_delegate.iter_activity_dfs


		#=====[ Step 2: iterate through all dataframes	]=====
		print_status ("get_corpus", "assembling texts")
		texts = []
		for df in iter_function ():
			print_inner_status ("assembling texts", "next text")
			texts += [t for t in list (df[colname]) if type(t) == list]


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
			trains self.semantic_analysis 
		"""
		#=====[ Step 1: get the corpus	]=====
		print_status ("train_semantic_analysis", "getting corpus/dictionary")
		corpus, dictionary = self.get_corpus ('activities', 'name')

		#=====[ Step 2: train ]=====
		print_status ("train_semantic_analysis", "training semantic analysis")
		self.semantic_analysis.train (corpus, dictionary)





	####################################################################################################
	######################[ --- Processing --- ]########################################################s
	####################################################################################################

	def perform_semantic_analysis (self, df):
		"""
			PUBLIC: perform_semantic_analysis
			---------------------------------
			given a dataframe, this will mark it up with LDA, TFIDF and W2V,
			then return it
		"""
		df = so.semantic_analysis.apply_lda (df, 'name')
		df = so.semantic_analysis.apply_tfidf (df, 'name')
		df = so.semantic_analysis.apply_w2v (df, 'name')
		return df






	####################################################################################################
	######################[ --- Inference --- ]#########################################################
	####################################################################################################

	def score_activities (user, activities):
		"""
			PUBLIC: score_activities
			------------------------
			given a representatiof a user and a representation of 
			a set of activities, this returns them in sorted order, 
			along with a measure of confidence for each 
		"""
		pass




if __name__ == "__main__":

	so = SpotOn ()

	#=====[ Step 1: train semantic analysis	]=====
	print_status ("Demo Script", "Training semantic analysis")
	so.train_semantic_analysis ()

	#=====[ Step 2: save semantic analysis models	]=====
	print_status ("Demo Script", "Saving semantic analysis models")
	so.semantic_analysis.save ()

	#=====[ Step 3: load semantic analysis models	]=====
	print_status ("Demo Script", "loading semantic analysis models")	
	so.semantic_analysis.load ()


	#=====[ Step 3: apply to activity dfs	]=====
	# print_status ("Demo Script", "Performing semantic analysis on activities")
	# for adf in so.storage_delegate.iter_activity_dfs ():
		# adf = so.perform_semantic_analysis (adf)
		# print adf.iloc[0]
