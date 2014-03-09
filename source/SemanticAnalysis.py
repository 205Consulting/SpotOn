# Class: SemanticAnalysis
# -----------------------
# class responsible for providing all functionality that 
# involved in discerning semantic content from natural 
# language
import numpy as np
from operator import itemgetter
from gensim.models.ldamodel import LdaModel
from gensim.models.tfidfmodel import TfidfModel
from gensim.models.word2vec import Word2Vec
from PD_LDA import PD_LDA


class SemanticAnalysis:

	filenames = 	{
						'lda': '../data/models/lda_model',
						'tfidf': '../data/models/tfidf_model',
						'word2vec': '../data/models/word2vec_model'
					}

	def __init__ (self):
		"""
			PUBLIC: Constructor
			-------------------
		"""
		#=====[ Step 1: load models	]=====
		self.lda_model = None
		self.tfidf_model = None
		self.word2vec_model = None
		self.load_models ()

		#=====[ Step 2: initialize pd_lda	]=====
		self.pd_lda = PD_LDA ()




	####################################################################################################
	######################[ --- LOADING MODELS --- ]####################################################
	####################################################################################################

	def load_models (self, lda_filename=filenames['lda'], tfidf_filename=filenames['tfidf'], word2vec_filename=filenames['word2vec']):
		"""
			PUBLIC: load_models
			-------------------
			loads in the specified models
		"""
		if lda_filename:
			self.load_lda_model (lda_filename)

		if tfidf_filename:
			self.load_tfidf_model (tfidf_filename)

		if word2vec_filename:
			self.load_word2vec_model (word2vec_filename)


	def load_lda_model (self, filename):
		self.lda_model = LdaModel.load (filename)


	def load_tfidf_model (self, filename):
		self.word2vec_model = Word2Vec.load (filename) 


	def load_word2vec_model (self, filename):
		self.word2vec_model = Word2Vec.load (filename) 






	####################################################################################################
	######################[ --- SAVING MODELS --- ]#####################################################
	####################################################################################################

	def save_models (self, 	lda_filename=filenames['lda'], tfidf_filename=filenames['tfidf'], word2vec_filename=filenames['word2vec']):
		"""
			PUBLIC: save_models
			-------------------
			saves all models
		"""
		if lda_filename:
			self.save_lda_model (lda_filename)

		if tfidf_filename:
			self.save_tfidf_model (tfidf_filename)

		if word2vec_filename:
			self.save_word2vec_model (word2vec_filename)


	def save_lda_model (self, filename='../models/lda_model'):
		pickle.dump (self.lda_model, open(filename, 'w'))


	def save_tfidf_model (self, filename='../data/word2vec/word2vec_model'):
		pickle.dump (self.tfidf_model, open(filename, 'w'))		


	def save_word2vec_model (self, filename='../data/word2vec/word2vec_model'):
		self.word2vec_model = Word2Vec.load (filename) 






	####################################################################################################
	######################[ --- LDA --- ]###############################################################
	####################################################################################################

	def update_lda (df, target_columns):
		"""
			PUBLIC: update_lda
			------------------
			updates the lda model with the target columns from 
			the passed dataframe 
		"""
		self.pd_lda.update_lda (df, target_columns)



	def apply_lda (df, target_columns):
		"""
			PUBLIC: apply_lda
			---------------
			given a dataframe and a list of 'target columns', this will run LDA 
			on all of them *combined*, add a column to df, and return it.
		"""
		df = self.pd_lda.apply_lda (df, target_columns)








	####################################################################################################
	######################[ --- TF.IDF --- ]############################################################
	####################################################################################################

	def get_colname_tfidf (self, target_col):
		return target_col + 'tfidf'

	def train_tfidf (self, dictionary, corpus):
		"""
			PUBLIC: train_tfidf
			-------------------
			trains self.tfidf_model given a gensim corpus
		"""
		self.dictionary = dictionary
		self.tfidf_model = TfidfModel (corpus)


	def get_tfidf_vec (word_list):
		"""
			PRIVATE: get_tfidf_vec
			----------------------
			given a list of words, returns a vector 
			where the ithe element is the tfidf of ith word 
			w/r/t the document 
		"""
		bow_rep = dictionary.doc2bow (word_list)
		return [t[1] for t in self.tfidf_model[bow_rep]]


	def apply_tfidf (df, target_col):
		"""
			PUBLIC: apply_tfidf
			-------------------
			given a dataframe and a list of 'target columns', this will run LDA 
			on all of them *concatenated* and return it as a column
		"""

		colname_tfidf = self.get_colname_tfidf (target_col)
		df['colname_tfidf'] = df[target_col].apply (self.get_tfidf_vec)






	####################################################################################################
	######################[ --- Word2Vec --- ]##########################################################
	####################################################################################################

	def get_colname_w2v (self, target_col):
		return target_col + '_W2V'

	def get_colname_w2v_w_sum (self, target_col):
		return target_col + '_W2V_w_s'

	def get_colname_w2v_w_avg (self, target_col):
		return target_col + '_W2V_w_avg'


	def get_w2v (self, word_list):
		"""
			PRIVATE: get_w2v
			----------------
			given a list of words, returns a list where the 
			ith element is the word2vec representation for the 
			ith word
		"""
		if type(word_list) == list:
			return [self.word2vec_model[w] if w in self.word2vec_model else None for w in word_list]
		else:
			return []


	def apply_w2v (self, df, target_col):
		"""
			PUBLIC: apply_word2vec
			----------------------
			given a dataframe and a list of 'target columns', this will return list
			of word vectors for their *concatenated* contents
		"""
		#=====[ Step 1: ensure model is in place, col exist	]=====
		assert (self.word2vec_model)
		assert (target_col in df)

		#=====[ Step 2: get col name, add column	]=====
		col_name = self.get_colname_w2v (target_col)
		df[col_name] = df[target_col].apply (self.get_w2v)
		return df


	def apply_w2v_w_sum (df, target_col):
		"""
			PUBLIC: apply_word2vec_w_sum
			----------------------------
			given a dataframe and a target column, gets a 
			weighted sum of the word2vecs for each element 
			based on its tf.idf 
		"""
		#=====[ Step 1: get column names	]=====
		tfidf_colname 		= self.get_colname_tfidf (target_col)
		w2v_colname 		= self.get_colname_w2v (target_col)
		w2v_w_sum_colname 	= self.get_colname_w2v_w_sum (target_col)

		#=====[ Step 2: ensure requisite columns exist	]=====
		if not tfidf_colname in df:
			df = self.apply_tfidf (df, target_col)
		if not w2v_colname in df:
			df = self.apply_w2v (df, target_col)

		#=====[ Step 3: compute weighted sums	]=====
		def weighted_w2v_sum (row):
			w2vs = row[w2v_colname]
			tfidfs = row[tfidf_colname]
			return np.sum ([])




	def apply_w2v_w_avg (df, target_columns):
		"""
			PUBLIC: apply_word2vec_w_avg
			----------------------------
			given a dataframe and a target column, gets a 
			weighted average of the word2vecs for each element 
			based on its tf.idf 
		"""
		#=====[ Step 1: get column names	]=====
		tfidf_colname 		= self.get_colname_tfidf 	(target_col)
		w2v_colname 		= self.get_colname_w2v (target_col)
		w2v_w_avg_colname 	= self.get_colname_w2v_w_avg (target_col)		

		#=====[ Step 2: ensure requisite columns exist	]=====
		if not tfidf_colname in df:
			df = self.apply_tfidf (df, target_col)
		if not w2v_colname in df:
			df = self.apply_w2v (df, target_col)

		#=====[ Step 3: compute weighted averages	]=====













