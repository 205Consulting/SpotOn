# Class: SemanticAnalysis
# -----------------------
# class responsible for providing all functionality that 
# involved in discerning semantic content from natural 
# language
import pickle
from operator import itemgetter
import numpy as np
import pandas as pd
from gensim.models.ldamodel import LdaModel
from gensim.models.tfidfmodel import TfidfModel
from gensim.models.word2vec import Word2Vec
from util import *



class SemanticAnalysis:

	#==========[ PARAMETERS	]==========
	num_topics_lda 	= 50
	filenames 		= {
						'gensim_dictionary': '../models/gensim_dictionary',
						'lda': '../data/models/lda_model',
						'tfidf': '../data/models/tfidf_model',
						'word2vec': '../data/models/word2vec_model'
					}



	def __init__ (self):
		"""
			PUBLIC: Constructor
			-------------------
		"""
		self.gensim_dictionary = None
		self.lda_model = None
		self.lda_model_topics = None
		self.tfidf_model = None
		self.word2vec_model = None



	####################################################################################################
	######################[ --- SAVING MODELS --- ]#####################################################
	####################################################################################################

	def save (self):
		"""
			PUBLIC: save
			------------
			saves the state; Note: doesn't save self.word2vec_model, 
			as this is trained and saved separately.
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
	######################[ --- LOADING MODELS --- ]####################################################
	####################################################################################################

	def load (self):
		"""
			PUBLIC: load
			------------
			load all models into memory
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
		self.tfidf_model = TfidfModel.load (filename) 

	def load_lda_model (self, filename='../data/models/lda_model'):
		self.lda_model = LdaModel.load (filename)
		self.lda_model_topics = self.find_per_topic_word_distributions ()

	def load_word2vec_model (self, filename='../data/models/word2vec_model'):
		self.word2vec_model = Word2Vec.load (filename) 











	####################################################################################################
	######################[ --- TRAINING MODELS --- ]###################################################
	####################################################################################################

	def train (self, corpus, dictionary):
		"""
			PUBLIC: train
			-------------
			given a corpus and a dictionary, fits parameters for all 
			models that require it
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
	######################[ --- ANALYZING --- ]#########################################################
	####################################################################################################

	def add_semantic_summary (self, df, target_col):
		"""
			PUBLIC: add_semantic_summary
			----------------------------
			given a dataframe, applies all types of analysis to it (tf.idf, lda, w2v)
			and returns the modified version
		"""
		assert (target_col in df)

		#=====[ Step 1: apply tf.idf	]=====
		# print_inner_status ("SA analyze", "applying TF.IDF")
		# df = self.apply_tfidf (df, target_col)

		#=====[ Step 2: apply lda	]=====
		print_inner_status ("SA analyze", "applying LDA")
		df = self.add_lda_vec_column (df, target_col)		

		#=====[ Step 3: apply word2vec	]=====
		# print_inner_status ("SA analyze", "applying Word2Vec")
		# df = self.apply_w2v (df, target_col)	

		#=====[ Step 4: apply w2v weighted sum	]=====
		print_inner_status ("SA analyze", "applying Word2Vec")
		df = self.apply_w2v_w_sum (df, target_col)

		#=====[ Step 5: apply w2v weighted average	]=====
		# print_inner_status ("SA analyze", "applying Word2Vec weighted average")
		# df = self.apply_w2v_w_avg (df, target_col)

		return df












	####################################################################################################
	######################[ --- TRAINING LDA --- ]######################################################
	####################################################################################################

	def get_colname_lda (self, target_col):
		return target_col + '_LDA'


	def find_per_topic_word_distributions(self):
		"""
			Function: find_per_topic_word_distributions
			-------------------------------------------
			Operates on self.lda_model to get a list of dicts, the i'th dict mapping 
			words -> probabilities for the i'th topic
		"""
		topic_dists = []

		#=====[ ITERATE THROUGH TOPICS	]=====
		for topic in range(self.lda_model.num_topics): 

			#=====[ Step 1: get/normalize topic distribution	]=====
			topic_dist = self.lda_model.state.get_lambda()[topic] 
			topic_dist = topic_dist / topic_dist.sum()

			#=====[ Step 2: fill topic_dist_dict with strings appropriately	]=====
			topic_dist_dict = {self.lda_model.id2word[i]:topic_dist[i] for i in range(len(topic_dist))}

			#=====[ Step 3: add to list of dicts	]=====
			topic_dists.append(topic_dist_dict) 

		return topic_dists


	def train_lda (self, corpus, dictionary):
		"""
			PRIVATE: train_lda
			------------------
			given a corpus and a dictionary, this fits parameters for self.lda_model, 
			fills self.lda_model_topics with the 
		"""
		self.lda_model = LdaModel(corpus, id2word=dictionary, num_topics=self.num_topics_lda)
		self.lda_model_topics = self.find_per_topic_word_distributions ()


	def print_lda_topics (self, words_per_topic=30):
		"""
			PUBLIC: print_lda_topics
			------------------------
			prints out self.lda_model_topics in an intuitive fashion
		"""
		#=====[ Step 1: ensure necessary conditions	]=====
		if not self.lda_model_topics:
			print_error ("print_lda_topics", "you have not found lda topics yet")

		#=====[ Step 2: iterate through topics, print constituent words	]=====
		for index, topic in enumerate(self.lda_model_topics):			
			print_header ("TOPIC: #" + str(index))

			sorted_words = sorted(topic.items(), reverse=True, key=lambda x: x[1])
			for word, weight in sorted_words[:words_per_topic]:
				print word, ": ", weight
			print "\n\n"





	####################################################################################################
	######################[ --- USING TFIDF --- ]#######################################################
	####################################################################################################

	def add_tfidf_column (self, df):
		"""
			PRIVATE: add_tfidf_column
			-------------------------
			params: df - dataframe containing activities
			returns: df containing 'tfidf_vec' column 
		"""
		def get_tfidf (word_list):
			return self.tfidf_model[self.dictionary.doc2bow(word_list)]
		df['tfidf_col'] = df['lda_doc'].apply (get_tfidf)
		return df


	####################################################################################################
	######################[ --- USING LDA --- ]#########################################################
	####################################################################################################

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


	def add_lda_doc_column (self, df):
		"""
			PRIVATE: add_lda_doc_column
			---------------------------
			adds a column to df, 'lda_doc', that contains 
			the document to be used for the given row
		"""
		df['lda_doc'] = df['name']*5 + df['words']
		return df


	def add_lda_vec_column (self, df):
		"""
			PUBLIC: add_lda_vec_column
			--------------------------
			given a dataframe, this will add an lda column
		"""	
		#=====[ Step 1: get the documents	]=====
		df = self.add_lda_doc_column (df)

		#=====[ Step 2: apply LDA to each	]=====
		df['lda_vec'] = df['lda_doc'].apply (self.get_lda_vec)
		return df


	def get_user_lda_doc (self, user_df):
		"""
			PUBLIC: get_user_doc
			--------------------
			given a dataframe representing a user, this 
			returns a document representing that user 
		"""
		#=====[ Step 1: get lda doc for each row	]=====
		user_df = self.add_lda_doc_column (user_df)

		#=====[ Step 2: sum together to get user doc 	]=====
		return sum(list(user_df['lda_doc']), [])


	def get_user_lda_vec (self, user_df):
		"""
			PUBLIC: get_user_lda_vec
			------------------------
			given a dataframe representing a user, this returns an lda 
			vector representing that user 
		"""
		#=====[ Step 1: get user doc	]=====
		user_doc = self.get_user_lda_doc (user_df)

		#=====[ Step 2: apply lda, return	]=====
		return self.get_lda_vec (user_doc)















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
		return df






	####################################################################################################
	######################[ --- Word2Vec --- ]##########################################################
	####################################################################################################

	def add_w2v_sum_column (self, df):
		"""
			PRIVATE: add_w2v_sum_column
			---------------------------
			adds a column 'w2v_sum' to the dataframe
		"""
		def sum_w2vs (word_list):
			word_list = [w for w in word_list if w in self.word2vec_model]
			if len(word_list) == 0:
				return np.zeros((200,))
			else:
				w2vs = np.array([self.word2vec_model[w] for w in word_list])
				return np.sum (w2vs, axis=0)

		df['w2v_sum'] = df['name'].apply (sum_w2vs)
		return df











