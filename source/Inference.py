# Class: inference
# -------------

from Preprocess import Preprocess
from scipy.spatial.distance import cosine
from scipy import stats

class Inference(object):
	def __init__(self, lda_model, topic_distributions, word2vec, feature_weights=[1 0]):
		self.lda_model = lda_model
		self.topic_distributions = topic_distributions
		self.feature_weights = feature_weights
		self.very_negative_number = -12930183492234

		

	def normalize_score_vector(self, vec):
		return stats.zscore(vec)




	def apply_weights(self, scores):
		'''
			function: apply_weights

			params: scores - list of score vectors to apply weights to

			returns: a score vector with normalized, weighted scores
		'''

		#1: normalize each score vector
		normalized_scores = [self.normalize_score_vector(vec) for vec in scores]

		#2: apply weights in self.weights
		weighted_score = [0 for i in range(len(scores[0]))]
		for i in range(len(scores)):
			for j in range(len(scores[i])):
				weighted_score[j] += scores[i][j]*self.feature_weights[i]

		#3: return
		return weighted_score






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




	def recommend(self, user_row, activities_df, user_lda_field, activities_field, user_w2v_field, activities_w2v_field):
		"""
			function: recommend

			params: user_id - user row to recommend for
					activites_df - activities df to score
					user_lda_field - field in the user row with the lda vector
					activities_field - field in the activities df with the WORDS (not the lda vector!)
					user_w2v_field - field in the user row with the word2vec vector
					activities_w2v_field - field in the activities df with the word2vec vector

			returns: a list of scores, the i'th score being the score for the i'th activity
		"""

		#1: get user's LDA vector and user's word2vec vector
		user_lda_vector = user_row[user_lda_field]
		user_w2v_vector = user_row[user_w2v_field]

		#2: create a list of lists, the i'th list being the words in the the i'th activity
		word_vectors = []
		for i in range(len(activities_df)):
			word_vectors.append(activities_df.iloc[i][activities_field])

		#3: find probability of generation for each activity
		prob_gen = []
		for i in range(len(word_vectors)):
			prob_gen.append(self.probability_of_generation(user_lda_vector, word_vectors[i]))

		#4: create a list of lists, the i'th list being the word2vec vector in the i'th activity
		word_to_vec_vectors = []
		for i in range(len(activities_df)):
			word_to_vec_vectors.append(activities_df.iloc[i][activities_w2v_field])

		#5: find the cosine similarity between the user's w2v and the activity's
		w2v_cosine_similarity = []
		for i in range(len(word_to_vec_vectors)):
			w2v_cosine_similarity.append(self.cosine_similarity(user_w2v_vector, word_to_vec_vectors[i]))

		#6: apply weights correctly
		weighted_scores = self.apply_weights([prob_gen, w2v_cosine_similarity])

		#7: return weighted_scores
		return weighted_scores














