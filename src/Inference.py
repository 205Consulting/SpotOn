# Class: inference
# ----------------
import json
import numpy as np
from scipy.spatial.distance import cosine
from scipy import stats
from sklearn.cluster import KMeans
from collections import defaultdict
from Preprocess import Preprocess
from util import *

class Inference(object):


	def __init__(self, lda_model, topic_distributions, feature_weights=[1, 0]):
		"""
			PUBLIC: Constructor
			-------------------
			fills in parameters
		"""
		self.lda_model = lda_model
		self.topic_distributions = topic_distributions
		self.feature_weights = feature_weights
		self.very_negative_number = -12930183492234

		

	def apply_weights(self, scores):
		"""
			PRIVATE: apply_weights
			----------------------
			given scores, a list of 
		"""
		#=====[ Step 1: normalize score vectors	]=====
		normalized_scores = [stats.zscore(s) for s in scores]

		#=====[ Step 2: apply weights from self.weights	]=====
		weighted_scores = [np.dot (self.feature_weights, s) for s in normalized_scores]
		return weighted_scores






	def chance_of_generation(self,topic, activity_words):
		'''
			function: chance_of_generation

			params: topic - topic index
					activity_words - words in the activity

			returns: the chance that topic number "topic" generates the list of words "event"
			note: returns the arithmetic average in log space of the generation probabilities so that long events aren't penalized
		'''
		total = 0.0
		seen = 0
		# 1: iterate through every word in activity_words
		for word in activity_words:

			if word not in self.topic_distributions[topic]:
				continue
			else:
				seen +=1 
			try:
				#2:  add the chance to generate this word
				total += np.log(self.topic_distributions[topic][word])
			except:
				print "event:" + str(event)
				print "topic:" + str(topic)
				print "word:" + str(word)
				sys.exit()

		# if seen was 0, then none of the words in activity_words has ever been seen, so we should return
		# the very negative number
		if seen == 0:
			return self.very_negative_number
		# 3: return the arithmetic average in log space (equivalent to the geometric average in probability space)
		total = total/seen
		return total



	def probability_of_generation(self, user_topic_vector, activity_words):
		'''
			function: probability_of_generation

			params: user_topic_vector - user's LDA topic vector
					activity_words - list of words in the activity

			returns: the probability that the topic vector generates the words in activity_words
		'''
		# 1: sum over all topics
		total = 0
		for topic in range(len(user_topic_vector)):
			# 2: chance of picking this topic
			topic_probability = user_topic_vector[topic]

			#3: chance of generating activity_words (NOTE: activity_words a list of words)
			chance_of_generation = self.chance_of_generation(topic, activity_words)

			total += topic_probability*chance_of_generation
		return total


	def cosine_similarity(self, vec_one, vec_two):
		return (1 - cosine(vec_one, vec_two))

	def k_activities_per_cluster(self, index_to_cluster, weighted_scores, k=None):
		'''
			function: k_activities_per_cluster

			params: index_to_cluster - mapping index of weighted scores to the cluster that activity was in
					weighted_scores - scores for each activity
					k - number of activities in each cluster to take
					( if k is none, then don't do any of this k-means stuff )
			returns: list of indices into the activities dataframe listed in order of their scores
		'''
		# if k is none, then just return the sorted indices of weighted_scores
		if k == None:
			return np.argsort(weighted_scores)[::-1]
		else:
			#map cluster index to number of elements of this cluster used
			cluster_mapping = defaultdict(int)
			to_return = []
			for index in np.argsort(weighted_scores)[::-1]:
				# check if we have too many of this cluster yet
				if cluster_mapping[index_to_cluster[index]] < k:
					to_return.append(index)
					# add one to the cluster mapping
					cluster_mapping[index_to_cluster[index]] += 1
				else:
					continue
			return to_return


	def extract_doc_from_activity (self, activity_row):
		"""
			PRIVATE: extract_doc_from_activity
			----------------------------------
			given an activity represented as a series, this returns 
			a list of words representing it. 
		"""
		doc = []
		if activity_row['name']:
			doc += activity_row['name']
		if activity_row['words']:
			doc += activity_row['words']
		return doc


	def get_user_representation_from_activities (self, user_activities_df):
		'''
			function: get_user_representation_from_activities

			params: user_activities - dataframe of activities belonging to the user
					activities_field - field with words in the activities df.

			returns: a dict containing a concatentated list of words including all the words belonging to the user
					 and an LDA vector
		'''
		#=====[ Step 1: get a document representing the user	]=====
		user_document = []
		for i in range(len(user_activities_df)):
			user_document += self.extract_doc_from_activity (user_activities_df.iloc[i])

		#=====[ Step 2: get an LDA vector for it	]=====
		lda_vector = self.get_lda_vec(user_document)

		#=====[ Step 3: construct and return a representation of the user	]=====
		return {'words': user_document, 'lda_vec': lda_vector}


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


	def score_activities (self, user_activities, recommend_activities):
		"""
			PUBLIC: score_activities
			------------------------
			given two dataframes, the first being of a given user's activities,
			the second of activities to recommend, this returns a list of scores 
		"""
		#=====[ Step 1: go from user_activities to a user representation	]=====
		user_rep = self.get_user_representation_from_activities(user_activities)

		#=====[ Step 2: get recommendation scores	]=====
		weighted_scores = self.recommend (user_rep, recommend_activities)

		# 3: TODO: how should we sort scores? currently returns the indices into "recommend_activites" df
		return np.argsort(weighted_scores)[::-1]


	def recommend (self, user_rep, activities_df):
		"""
			function: recommend

			params: user_id - user row to recommend for
					activites_df - activities df to score
					user_lda_field - field in the user row with the lda vector
					activities_field - field in the activities df with the WORDS (not the lda vector!)
					user_w2v_field - field in the user row with the word2vec vector
					activities_w2v_field - field in the activities df with the word2vec vector

			returns: a list of indices into the activities dataframe, in order of score

			notes: assumed LDA vectors in the activities dataframe is in column activities_field + '_lda'
		"""

		#1: get user's LDA vector and user's word2vec vector
		user_lda_vector = user_rep['lda_vec']
		# user_w2v_vector = user_row[user_w2v_field]

		#2: create a list of lists, the i'th list being the words in the the i'th activity
		word_vectors = []
		for i in range(len(activities_df)):
			word_vectors.append (self.extract_doc_from_activity(activities_df.iloc[i]))

		#3: find probability of generation for each activity
		prob_gen = []
		for i in range(len(word_vectors)):
			prob_gen.append(self.probability_of_generation(user_lda_vector, word_vectors[i]))

		#4: create a list of lists, the i'th list being the word2vec vector in the i'th activity
		# word_to_vec_vectors = []
		# for i in range(len(activities_df)):
		# 	word_to_vec_vectors.append(activities_df.iloc[i][activities_w2v_field])

		w2v_cosine_similarity = [0 for i in range(len(activities_df))]
		#5: find the cosine similarity between the user's w2v and the activity's
		# w2v_cosine_similarity = []
		# for i in range(len(word_to_vec_vectors)):
			# w2v_cosine_similarity.append(self.cosine_similarity(user_w2v_vector, word_to_vec_vectors[i]))

		#6: apply weights correctly
		# weighted_scores = self.apply_weights([prob_gen, w2v_cosine_similarity])
		return prob_gen

		#7: run k-means on concatenated LDA + w2v scores
		# concat_LDA_w2v = []
		# for i in range(len(activities_df)):
			# concat_LDA_w2v.append((activities_df.iloc[i][activities_field + '_LDA']) + word_to_vec_vectors[i])

		# kmeans = KMeans(n_clusters=5)
		# fit_predicted = kmeans.fit_predict(concat_LDA_w2v)

		# return self.k_activities_per_cluster(fit_predicted, weighted_scores, k=None)














