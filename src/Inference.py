# Class: inference
# ----------------
import json
import numpy as np
from scipy.spatial.distance import cosine, cdist
from scipy import stats
from sklearn.cluster import KMeans
from collections import defaultdict
from Preprocess import Preprocess
from util import *

class Inference(object):

	#==========[ PARAMETERS	]==========
	minimum_doc_length = 20


	def __init__(self, lda_model, topic_distributions, feature_weights=[1, 0]):
		"""
			PUBLIC: Constructor
			-------------------
			fills in parameters
		"""
		self.lda_model = lda_model
		self.topic_distributions = topic_distributions
		self.feature_weights = feature_weights


	####################################################################################################
	######################[ --- COMPUTE FEATURES --- ]##################################################
	####################################################################################################

	def get_doc_gen_prob (self, user_lda_vec, word_list):
		"""
			PRIVATE: get_doc_gen_prob
			-------------------------
			params: user_lda_vec - user's LDA topic vector
					word_list - list of words in the document

			returns: probability that user topic vector generates document
					comprised of words in word_list
		"""
		#=====[ Step 1: get valid words	]=====
		valid_words = [w for w in word_list if w in self.topic_distributions[0]]
		if len(valid_words) < self.minimum_doc_length:
			return np.nan

		#=====[ Step 2: get pseudo-p(words | topic)	for each topic ]=====
		#p_w: ith row, jth col = probability that ith topic generated word j
		p_w = np.array([[topic[w] for w in valid_words] for topic in self.topic_distributions])
		#p_W: ith entry = probability that ith topic generated the words list
		p_W = np.exp(np.mean(np.log (p_w), axis=1))
		#normalize p_W: gets rid of priors irrelevant to this user?
		p_W = p_W / np.sum(p_W)

		#=====[ Step 3: 'sparse code' it - remove low values	]=====
		p_W[p_W < 0.05] = 0
		p_W = p_W / np.sum(p_W)

		#=====[ Step 3: multiply in user priors on topics	]=====
		return np.dot (user_lda_vec, p_W)


	def compute_lda_gen_prob_feature (self, user_rep, a_df):
		"""
			PRIVATE: compute_lda_gen_prob_feature
			-------------------------------------
			params: user_rep - user representation 
					a_df - dataframe of activities

			returns: pandas series containing generation probability 
						for user/activity
		"""
		user_lda_vec = user_rep['lda_vec']
		def p_gen (doc):
			return self.get_doc_gen_prob (user_lda_vec, doc)
		return a_df['lda_doc'].apply (p_gen)


	def min_cosine_sims (self, user_col, activities_col):
		"""
			PRIVATE: min_cosine_sims
			------------------------
			params:	user_col - pandas column containing vectors describing user
					activities_col - pandas column containing vectors describing 
									activities

			returns: numpy array where ith element is minumum cosine sim between
						a user and an event
		"""
		#=====[ Step 1: get cosine sims between entities	]=====
		sims = 1 - cdist (activities_col, user_col, 'cosine')
		sims[np.isnan(sims)] = -1

		#=====[ Step 2: return the min of each *row*	]=====
		return np.max (sims, axis=1)


	def compute_lda_sim_feature (self, user_rep, a_df):
		"""
			PRIVATE: add_lda_sim_feature
			----------------------------
			params: user_rep - user representation
					a_df - dataframe of activities 

			returns: pandas series containing cosine similarity 
						between user lda_vec and activity lda_vec
		"""
		u_ldas = np.array(list(user_rep['events_df']['lda_vec']))
		a_ldas = np.array(list(a_df['lda_vec']))
		return self.min_cosine_sims (u_ldas, a_ldas)


	def compute_w2v_sim_feature (self, user_rep, a_df):
		"""
			PRIVATE: add_w2v_sim_feature
			----------------------------
			params: user_rep - user representation
					a_df - dataframe of activities 

			returns: pandas series containing cosine similarity 
						between user w2v and activity w2v

			NOTE: currently implemented as *maximum* cosine similarity
			between the activity_w2v and all of the user w2vs
		"""
		u_w2vs = np.array(list(user_rep['events_df']['w2v_sum']))
		a_w2vs = np.array(list(a_df['w2v_sum']))
		return self.min_cosine_sims (u_w2vs, a_w2vs)


	def normalize_feature (self, array):
		"""
			PRIVATE: normalize_feature
			--------------------------
			given an array representing a feature, this will 'normalize' 
			the feature by calculating its zscore
		"""
		#=====[ Step 1: convert to zscores	]=====
		mean, std = np.mean(array), np.std(array)
		zscores = (array - mean) / std

		#=====[ Step 2: shift so all are positive	]=====
		return (zscores - np.min (zscores))


	def featurize (self, user_rep, a_df):
		"""
			PRIVATE: featurize
			------------------
			params: user_rep - user representation 
					a_df - dataframe containing activities

			returns: a_df with feature columns added appropriately
		"""
		#=====[ Step 1: add raw features	]=====
		# a_df['[FEATURE: lda_gen_prob]']	= self.compute_lda_gen_prob_feature (user_rep, a_df)
		a_df['[FEATURE: lda_sim]'] 		= self.compute_lda_sim_feature (user_rep, a_df)
		a_df['[FEATURE: w2v_sim]'] 		= self.compute_w2v_sim_feature (user_rep, a_df)				

		#=====[ Step 2: normalize features	]=====
		# a_df['[FEATURE: lda_gen_prob] norm']	= self.normalize_feature(a_df['[FEATURE: lda_gen_prob]'])
		a_df['[FEATURE: lda_sim] norm'] 		= self.normalize_feature(a_df['[FEATURE: lda_sim]'])
		a_df['[FEATURE: w2v_sim] norm'] 		= self.normalize_feature(a_df['[FEATURE: w2v_sim]'])				
		return a_df


	def is_featurized (self, a_df):
		"""
			PRIVATE: is_featurized
			----------------------
			params: a_df - dataframe containing activities

			returns: boolean value for wether a_df has had features 
						added to it appropriately
		"""
		if '[FEATURE: lda_sim]' in a_df:
			if '[FEATURE: w2v_sim]' in a_df:
				# if '[FEATURE: lda_gen_prob]' in a_df:
					# return True
				return True

		return False











	####################################################################################################
	######################[ --- COMPUTE SCORES/RANKINGS --- ]###########################################
	####################################################################################################

	def compute_scores (self, a_df):
		"""
			PRIVATE: compute_scores
			-----------------------
			params: a_df - dataframe containing activities *WITH* features
							having been added

			returns: pandas series representing a score for every activity
		"""
		#=====[ Step 1: ensure a_df is featurized	]=====
		assert self.is_featurized (a_df)

		#=====[ Step 2: combine scores	]=====
		# scores = a_df['[FEATURE: w2v_sim] norm']
		# scores = a_df['[FEATURE: lda_gen_prob] norm']
		scores = a_df['[FEATURE: lda_sim]'] * a_df['[FEATURE: w2v_sim]']
		# scores = a_df['[FEATURE: lda_gen_prob] norm'] * a_df['[FEATURE: w2v_sim] norm'] * a_df['[FEATURE: lda_sim] norm']		
		scores = scores.fillna(-100000)
		return scores


	def infer_scores (self, user_rep, a_df):
		"""
			PUBLIC: infer_scores
			--------------------
			params: user_rep - user representation 
					a_df - dataframe containing activities

			returns: dataframe with all features and scores added
		"""
		#=====[ Step 1: featurize	]=====
		a_df = self.featurize (user_rep, a_df)

		#=====[ Step 2: add scores, return	]=====
		a_df['score'] = self.compute_scores (a_df)
		return a_df


	def rank_activities (self, user_rep, a_df):
		"""
			PUBLIC: rank_activities
			-----------------------
			params: user_rep - user representation 
					a_df - dataframe containing activities

			returns: pandas index referencing row names in order of 
						their recommendation scores
			Note: this returns a list of row *names* (pass to .loc), not 
			true .iloc positional indices
		"""
		#=====[ Step 1: add scores	]=====
		a_df = self.infer_scores (user_rep, a_df)

		#=====[ Step 2: sorted index (nans still at front)]=====
		sorted_ix = a_df.sort('score', ascending=False).index
		return sorted_ix


