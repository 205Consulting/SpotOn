# Class: SemanticAnalysis
# -----------------------
# class responsible for providing all functionality that 
# involved in discerning semantic content from natural 
# language
import numpy as np
import pickle
from operator import itemgetter
from gensim.models.ldamodel import LdaModel
from gensim.models.tfidfmodel import TfidfModel
from gensim.models.word2vec import Word2Vec
from util import print_inner_status



class SemanticAnalysis:

	filenames = 	{
						'gensim_dictionary': '../models/gensim_dictionary',
						'lda': '../data/models/lda_model',
						'tfidf': '../data/models/tfidf_model',
						'word2vec': '../data/models/word2vec_model'
					}
	num_topics_lda = 10

	def __init__ (self):
		"""
			PUBLIC: Constructor
			-------------------
		"""
		self.gensim_dictionary = None
		self.lda_model = None
		self.tfidf_model = None
		self.word2vec_model = None




	####################################################################################################
	######################[ --- LOADING MODELS --- ]####################################################
	####################################################################################################

	def load (self):
		"""
			PUBLIC: load
			------------
			load the state
		"""
		print_inner_status ("SA load", "loading dictionary")
		self.load_dictionary ()

		print_inner_status ("SA load", "loading TF.IDF model")
		self.load_tfidf_model ()

		print_inner_status ("SA load", "loading LDA model ")
		self.load_lda_model ()

		print_inner_status ("SA load", "loading Word2Vec model")
		self.load_word2vec_model ()


	def load_dictionary (self, filename='../data/models/gensim_dictionary'):
		self.dictionary = pickle.load (open(filename, 'r'))

	def load_tfidf_model (self, filename='../data/models/tfidf_model'):
		self.word2vec_model = Word2Vec.load (filename) 

	def load_lda_model (self, filename='../data/models/lda_model'):
		self.lda_model = LdaModel.load (filename)

	def load_word2vec_model (self, filename='../data/models/word2vec_model'):
		self.word2vec_model = Word2Vec.load (filename) 






	####################################################################################################
	######################[ --- SAVING MODELS --- ]#####################################################
	####################################################################################################

	def save (self):
		"""
			PUBLIC: save
			------------
			saves the state
		"""
		print_inner_status ("SA save", "saving dictionary")
		self.save_dictionary ()

		print_inner_status ("SA save", "saving TF.IDF")		
		self.save_tfidf_model ()

		print_inner_status ("SA save", "saving LDA")				
		self.save_lda_model ()


	def save_dictionary (self, filename='../data/models/gensim_dictionary'):
		pickle.dump (self.dictionary, open(filename, 'w'))

	def save_tfidf_model (self, filename='../data/models/tfidf_model'):
		pickle.dump (self.tfidf_model, open(filename, 'w'))		

	def save_lda_model (self, filename='../data/models/lda_model'):
		pickle.dump (self.lda_model, open(filename, 'w'))

	def save_word2vec_model (self, filename='../data/models/word2vec_model'):
		self.word2vec_model = Word2Vec.load (filename) 



	####################################################################################################
	######################[ --- TRAINING MODELS --- ]###################################################
	####################################################################################################

	def train (self, corpus, dictionary):
		"""
			PUBLIC: train
			-------------
			given a corpus and a dictionary, trains all models
		"""
		#=====[ Step 1: save dictionary	]=====
		self.dictionary = dictionary

		#=====[ Step 2: train tf.idf	]=====
		print_inner_status ("SA train", "training TF.IDF model")
		self.train_tfidf (corpus, dictionary)

		#=====[ Step 3: train lda	]=====
		print_inner_status ("SA train", "training LDA model")		
		self.train_lda (corpus, dictionary)






	####################################################################################################
	######################[ --- LDA --- ]###############################################################
	####################################################################################################

	def get_colname_lda (self, target_col):
		return target_col + '_LDA'

	def train_lda (self, corpus, dictionary):
		"""
			PRIVATE: train_lda
			------------------
			given a corpus and a dictionary, this trains self.lda_model
		"""
		self.lda_model = LdaModel(corpus, id2word=dictionary, num_topics=self.num_topics_lda)


	def get_lda_vec (self, word_list):
		"""
			PRIVATE: get_lda_vec
			--------------------
			given a list of words, returns an lda vector characterizing 
			it
		"""
		#=====[ Step 1: convert to gensim bag of words	]=====
		gensim_bow = self.lda_model.id2word.doc2bow(word_list)

		#=====[ Step 2: get and return lda vector	]=====
		gamma, sstats = self.lda_model.inference([gensim_bow])
		normalized_gamma = gamma[0] / sum(gamma[0])
		return normalized_gamma


	def apply_lda (self, df, target_col):
		"""
			PUBLIC: apply_lda
			-----------------
			given a dataframe and a target column, this will run LDA 
			on it, add a column to df, and return it.
		"""
		colname_lda = self.get_colname_lda (target_col)
		df[colname_lda] = df[target_col].apply (self.get_lda_vec)
		return df







	####################################################################################################
	######################[ --- TF.IDF --- ]############################################################
	####################################################################################################

	def get_colname_tfidf (self, target_col):
		return target_col + '_TF.IDF'

	def train_tfidf (self, corpus, dictionary):
		"""
			PUBLIC: train_tfidf
			-------------------
			trains self.tfidf_model given a gensim corpus
		"""
		self.dictionary = dictionary
		self.tfidf_model = TfidfModel (corpus)


	def get_tfidf_vec (self, word_list):
		"""
			PRIVATE: get_tfidf_vec
			----------------------
			given a list of words, returns a vector 
			where the ithe element is the tfidf of ith word 
			w/r/t the document 
		"""
		bow_rep = self.dictionary.doc2bow (word_list)
		return [t[1] for t in self.tfidf_model[bow_rep]]


	def apply_tfidf (self, df, target_col):
		"""
			PUBLIC: apply_tfidf
			-------------------
			given a dataframe and a list of 'target columns', this will run LDA 
			on all of them *concatenated* and return it as a column
		"""

		colname_tfidf = self.get_colname_tfidf (target_col)
		df[colname_tfidf] = df[target_col].apply (self.get_tfidf_vec)






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
		pass












