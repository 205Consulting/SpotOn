import gensim
import sys
from time import sleep
import math
import numpy as np
from numpy import linalg
from scipy.spatial.distance import cosine
import scipy.sparse as sp
from sklearn.decomposition import PCA
import scipy.sparse.linalg as spln
# from scipy.stats import multivariate_normal

class LDATrainer(object):
	def __init__(self, ce_df, a_df, u_df, num_topics=6):
		self.ce_df = ce_df
		self.a_df = a_df
		self.u_df = u_df
		self.LDA_model, self.dictionary = self.train_lda_model(num_topics)
		self.topic_distributions = self.find_topic_distributions(num_topics)
		# self.mv_gaussian_distribution = self.build_mv_gaussiate_distribution()

	def find_topic_distributions(self, num_topics):
		'''
			PRIVATE: find_topic_distributions
			-------------------
			returns a list of dicts, each dict mapping word -> probability of generation for that topic
		'''
		topic_dists = []
		for topic in range(num_topics):
			topic_dict = {}
			topic_dist = self.LDA_model.state.get_lambda()[topic]
			#normalize topc
			topic_dist = topic_dist/topic_dist.sum()
			for i in range(len(topic_dist)):
				topic_dict[self.LDA_model.id2word[i]] = topic_dist[i]
			topic_dists.append(topic_dict)
		return topic_dists



	def train_lda_model(self, num_topics):
		'''
			PRIVATE: train_lda_model:
			-------------------------
			main function used to train the lda model

		'''

		corpus,dictionary = self.generate_corpus()
		lda = gensim.models.LdaModel(corpus, id2word=dictionary, num_topics=num_topics)
		return lda, dictionary


	def vector_concatenation(self, name, description):
		'''
			PRIVATE: vector_concatenation:

			concatenates list of words, disregarding NaNs
		'''		
		print description
		if type(name) != list:#math.isnan(name):
			if type(description) != list:#math.isnan(description):
				return []
			else:
				return description
		else:
			if type(description) != list:#math.isnan(description):
				return name
			else:
				return name + description




	def find_user_texts(self, user_names, user_descriptions):
		'''
			find_user_texts
			returns a list of lists, each list representing the concatenation of
			all of a single user's names and descriptions
		'''

		texts = []
		for i in range(len(user_names)):
			names = user_names[i]
			descriptions = user_descriptions[i]

			user_text = []
			for name in names:
				if type(name) == list:
					for element in name:
						user_text.append(element)
				else:
					continue
			for description in descriptions:
				if type(description) == list:
					for element in description:
						user_text.append(element)
				else:
					continue
			texts.append(user_text)

		return texts


	def generate_corpus(self):
		'''
			PRIVATE: generate_corupus:

			makes a gensim-style corpus from the data frames given
		'''

		# loop through data frames, storing word lists in texts
		texts = []

		ce_names = list(self.ce_df['name'])			
		ce_descriptions = list(self.ce_df['description'])			
		a_names = list(self.a_df['name'])			
		a_descriptions = list(self.a_df['words'])

		# make gensim text lists
		# ce_texts = [self.vector_concatenation(ce_names[i], ce_descriptions[i]) for i in range(len(ce_names))]
		a_texts = [self.vector_concatenation(a_names[i], a_descriptions[i]) for i in range(len(a_names))]
		self.a_vecs = a_texts
		user_names = list(self.u_df['event_names'])
		user_descriptions = list(self.u_df['event_descriptions'])

		user_texts = self.find_user_texts(user_names, user_descriptions)



		all_texts = a_texts

		print all_texts



		# make gensim dictionary
		dictionary = gensim.corpora.Dictionary(all_texts)
		## could save dictionary here

		#make gensim corpus
		corpus = [dictionary.doc2bow(text) for text in all_texts]
		return corpus, dictionary


	# user, event are both list of words
	# user is assumed to already be the concatenation of all his events


	def chance_of_generation(self,topic, event):
		'''
			chance_of_generation:
			returns the chance that topic number "topic" generates the list of words "event"
			note: returns the arithmetc average in log space of the generation probabilities so that long events aren't penalized
		'''
		total = 0.0
		seen = 0
		for word in event:

			if word not in self.topic_distributions[topic]:
				continue
			else:
				seen +=1 
			try:
				total += np.log(self.topic_distributions[topic][word])
			except:
				print "event:" + str(event)
				print "topic:" + str(topic)
				print "word:" + str(word)
				sys.exit()

		if seen == 0:
			return -19023812
		total = total/seen
		return total



	def probability_of_generation(self, user_topic_vector, event):
		'''
			probability_of_generation:

			returns the probability that the topic vector generates the words in event
		'''
		# sum over all topics
		total = 0
		for topic in range(len(user_topic_vector)):
			#chance of picking this topic
			topic_probability = user_topic_vector[topic]

			#chance of generating event (NOTE: event a list of words)
			chance_of_generation = self.chance_of_generation(topic, event)

			total += topic_probability*chance_of_generation
		return total



	def max_cluster_matching(self,topic_vector_one, topic_vector_two):
		'''
			max_cluster_matching:
			returns if the two topic vectors have the same argmax or not
		'''
		if np.argmax(topic_vector_one) == np.argmax(topic_vector_two):
			return 1
		else:
			return 0



	def featurefactory(self, user, event):
		'''
			featurefactor:
			returns a list of features for the user,event pair
			assumes that user is a concatenated list of all the user's event descriptions and event is a list of words for that event
		'''
		features = []
		user_topic_vector = self.LDA_model.inference([self.LDA_model.id2word.doc2bow(user)])[0][0] ## CHECK THIS [0][0] NONSENSE
		event_topic_vector = self.LDA_model.inference([self.LDA_model.id2word.doc2bow(event)])[0][0]
		# print user_topic_vector
		# print event_topic_vector

		user_topic_vector = user_topic_vector / sum(user_topic_vector)
		event_topic_vector = event_topic_vector / sum(event_topic_vector)
		# cosine similarity
		features.append(1 - cosine(user_topic_vector, event_topic_vector))

		# probability of generation (user->event)
		features.append(self.probability_of_generation(user_topic_vector, event))

		# probability of generation (event->user) (*note: not clear that this should even be used)
		features.append(self.probability_of_generation(event_topic_vector, user))
		
		# max cluster
		features.append(self.max_cluster_matching(user_topic_vector, event_topic_vector))


		# difference vector (consider using PCA on this before adding to features)
		array_features = [feature for feature in features]
		
		array_features += ([1 - abs(x) for x in np.subtract(user_topic_vector, event_topic_vector)])
		
		return array_features



	def find_word_vec_from_user_dataframe(self, user_data_frame):
		word_vec = []
		for sublist in list(user_data_frame['event_descriptions']):
			# for sublist in desc:
			if type(sublist) == list:
				for element in sublist:
					try:
						word_vec += element
					except:
						print "word vec:" + str(word_vec)
						print "element: " + str(element)
						print "sublist: " + str(sublist)
						print "original: " + str(user_data_frame['event_descriptions'])
						sys.exit()
		for sublist in list(user_data_frame['event_names']):
			#for sublist in name:
			if type(sublist) == list:
				for element in sublist:
					word_vec += element

		return word_vec


	def find_word_vec_from_ce_dataframe(self, ce_data_frame):
		word_vec = []
		if type(ce_data_frame['description']) == list:
			word_vec += ce_data_frame['description']
		if type(ce_data_frame['name']) == list:
			word_vec += ce_data_frame['name']
		# print word_vec
		# sleep(2)
		return word_vec

	# def find_feature_vector(self, row):
	# 	# find user id
	# 	user_id = row['user']
	# 	# find that user in the user dataframe
	# 	user_index = (self.u_df['id'] == user_id)
	# 	# get user's word vector
	# 	user = self.u_df[user_index]

	# 	user_word_vec = self.find_word_vec_from_user_dataframe(user)
	# 	return self.featurefactory(user, self.find_word_vec_from_ce_dataframe(row))


	def find_event_vec(self, calendar_event):
		vec = []
		if type(calendar_event['description']) == list:
			vec += calendar_event['description']
		if type(calendar_event['name']) == list:
			vec += calendar_event['name']
		return vec




	def find_user_vec(self, user_dataframe):
		# for descriptions in user_dataframe['event_descriptions']:
			# break
		# for names in user_dataframe['event_names']:
			#break
		descriptions = user_dataframe['event_descriptions']
		names = user_dataframe['event_names']

		word_vec = []
		for sublist in descriptions:
			if type(sublist) == list:
				word_vec += sublist
		for sublist in names:
			if type(sublist) == list:
				word_vec += sublist

		return word_vec


	def build_mv_gaussiate_distribution(self):
		'''
			build_mv_gaussiate_distribution:

			considers each (user,event) pair that we know happened and builds a multivariate gaussian distribution to
			use to estimate the probability that an unknown (user,event) pair would happened
		'''
		features = []
		# user_to_wordvec = {}
		# for i in range(len(self.ce_df)):
		# 	print i
		# 	calendar_event = self.ce_df.iloc[i]
		# 	event_vec = self.find_event_vec(calendar_event)
		# 	if event_vec == []:
		# 		continue

		# 	user = calendar_event['user']
		# 	if user in user_to_wordvec:
		# 		user_vec = user_to_wordvec[user]
		# 	else:
		# 		user_index = (self.u_df['id'] == user)
		# 		user_dataframe = self.u_df[user_index]

		# 		user_vec = self.find_user_vec(user_dataframe)
		# 		user_to_wordvec[user] = user_vec
		# 	features.append(self.featurefactory(user_vec, event_vec))

		# print len(features)
		# print features[2]
		# print "\n\n\n"
		# print features[5]



		user_to_wordvec = {}
		def find_feature_vector(row):
			print len(features)
			if len(features) > 500:
				return
			# find user id
			user_id = row['user']
			if user_id in user_to_wordvec:
				user_word_vec = user_to_wordvec[user_id]
			else:
				# find that user in the user dataframe
				user_index = (self.u_df['id'] == user_id)
				# get user's word vector
				user = self.u_df[user_index]

				user_word_vec = self.find_user_vec(user)
				user_to_wordvec[user_id] = user_word_vec

			feature = self.featurefactory(user_word_vec, self.find_event_vec(row))
			if feature[1] < -19023810:
				print "throwing one away"
				# print "throwing this away"
				# continue
			else:
				print "found one"
				features.append(feature)
		#want to do self.featurefactory() on every user,event pair
		self.ce_df.apply(find_feature_vector, 1)

		means = np.average(features, axis=0)
		cov = np.cov(np.transpose(features))
		#stds = np.std(features, axis=0)
		#return multivariate_normal(mean=means, cov=cov)
		return (means, cov)


	#def find_activity_vec(self, activity):


	def lognormpdf(self,x,mu,S):
		nx = len(S)
		norm_coeff = nx*math.log(2*math.pi)+np.linalg.slogdet(S)[1]
		err = x-mu
		if (sp.issparse(S)):
			numerator = spln.spsolve(S, err).T.dot(err)
		else:
			numerator = np.linalg.solve(S, err).T.dot(err)
		return -0.5*(norm_coeff+numerator)



	def norm_pdf_multivariate(self, x, mu, sigma):
		size = len(x)
		if size == len(mu) and (size, size) == sigma.shape:
			det = linalg.det(sigma)
			if det == 0:
				raise NameError("The covariance matrix can't be singular")
			norm_const = 1.0/ ( math.pow((2*np.pi),float(size)/2) * math.pow(det,1.0/2) )
			x_mu = matrix(x - mu)
			inv = sigma.I        
			result = math.pow(math.e, -0.5 * (x_mu * inv * x_mu.T))
			return norm_const * result
		else:
			raise NameError("The dimensions of the input don't match")

	# user is a u_df row
	def recommend(self, user):
		scores = []
		cosine_sim = []
		prob_gen_one = []
		prob_gen_two = []
		user_word_vec = self.find_user_vec(user)
		mapping = {}
		for i in range(len(self.a_vecs)):
			activity_vec = self.a_vecs[i]
			if len(activity_vec) < 6:
				continue
			feature_vec = self.featurefactory(user_word_vec, activity_vec)
			cosine_sim.append(feature_vec[0])
			prob_gen_one.append(feature_vec[1])
			prob_gen_two.append(feature_vec[2])
			scores.append(feature_vec)
			mapping[len(cosine_sim) -1] = i

		pca = PCA(n_components=1)
		X = np.array(scores)
		transformed = pca.fit_transform(X)
		new_transformed = []
		for x in transformed:
			new_transformed.append(x[0])





		return new_transformed, cosine_sim, prob_gen_one, prob_gen_two, mapping














