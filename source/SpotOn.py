# Class: SpotOn
# -------------
# container class for all others; goes from raw data 
# to predictions for users
import gensim
from StorageDelegate import StorageDelegate
from Preprocess import Preprocess
from SemanticAnalysis import SemanticAnalysis
from util import print_status, print_inner_status

class SpotOn:

	def __init__ (self):
		"""
			PUBLIC: Constructor
			-------------------
		"""
		#=====[ Step 1: create member objects	]=====
		self.storage_delegate = StorageDelegate ()
		self.semantic_analysis = SemanticAnalysis ()


	####################################################################################################
	######################[ --- UTILITIES --- ]#########################################################
	####################################################################################################

	def apply_calendars (self, func, row=None):
		"""
			PRIVATE: apply_calendars 
			------------------------
			given a function and optionally a row, this will 
			apply it to all the calendar dataframes available 
		"""
		raise NotImplementedError


	def get_corpus (self, df_type, colname):
		"""
			PRIVATE: get_corpus
			-------------------
			given 'activities' or 'calendars', assembles a corpus 
			of all of the text fields and returns it 
		"""
		#=====[ Step 1: assert type is calendar or activitie	]=====
		print_status ("get_corpus", "parsing args")
		assert (df_type == 'calendar' or df_type =='activity')
		if df_type == 'calendar':
			iter_function = self.storage_delegate.iter_calendar_dfs
		elif df_type == 'activity':
			iter_function = self.storage_delegate.iter_activity_dfs


		#=====[ Step 2: iterate through all dataframes	]=====
		print_status ("get_corpus", "assembling texts")
		texts = []
		for df in iter_function ():
			print_inner_status ("assembling texts", "next text")
			texts += [t for t in list (df[colname]) if type(t) == list]c
		print texts[0]


		#=====[ Step 3: get dictionary	]=====
		print_status ("get_corpus", "assembling dictionary")
		dictionary = gensim.corpora.Dictionary(texts)


		#=====[ Step 4: get corpus	]=====
		print_status ("get_corpus", "assembling corpus")		
		corpus = [dictionary.doc2bow (text) for text in texts]

		return corpus, dictionary









	####################################################################################################
	######################[ --- Training --- ]##########################################################
	####################################################################################################

	def train_semantic_analysis (self):
		"""
			PUBLIC: train_semantic_analysis
			-------------------------------
			trains self.semantic_analysis 
		"""
		#=====[ Step 1: iterate through activities	]=====
		for activity_df in self.storage_delegate.iter_activity_dfs ():
			self.semantic_analysis.update_lda (activity_df)





	####################################################################################################
	######################[ --- Inference --- ]#########################################################
	####################################################################################################

	def score_activities (user, activities):
		"""
			PUBLIC: score_activities
			------------------------
			given a representation of a user and a representation of 
			a set of activities, this returns them in sorted order, 
			along with a measure of confidence for each 
		"""
		pass




if __name__ == "__main__":

	so = SpotOn ()
	corpus, dictionary = so.get_corpus ('activity', 'name')


	# so.semantic_analysis.load_word2vec_model ('../data/models/word2vec_model')
	# for activity in so.storage_delegate.iter_activity_dfs ():
	# 	df = so.semantic_analysis.apply_w2v (activity, 'name')
	# 	print df.iloc[0]
