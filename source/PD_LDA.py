import pandas as pd
import gensim
import operator
from functools import partial
import sys,os,io



class PD_LDA(object):


	def __init__(self, existing_lda_model = None):
		'''
			function: init

			params: existing_lda_model - a tuple (lda_model, fields) where lda_model is a trained LDA model and fields 
					is a list of fields that was used to train that lda model

			returns: instantiated pd_lda class
		'''
		self.fields_to_lda_models = {}
		if existing_lda_model != None:
			lda_model = existing_lda_model[0]
			fields = existing_lda_model[1]
			self.fields_to_lda_models[self.fields_to_string(fields)] = lda_model

		
		
# ============================[ FUNCTIONS INTENDED TO BE CALLED BY OTHER MODULES ] ======== #

	def get_lda_model(self, fields, distributions=False):
			'''
				function: get_lda_model

				params: fields - list of fields that represent the LDA model you want
						distributions - whether or not you want the distributions of the lda model

				returns: the trained LDA model, as well as the per topics word distributions if requested
			'''
			# 1: make fields string
			fields_string = self.fields_to_string(fields)

			# 2: find lda model
			if fields_string in self.fields_to_lda_models:
				lda_model = self.fields_to_lda_models[fields_string]
			else:
				raise ValueError('There is no LDA model trained using those fields. If you want to train one, use update_lda')

			# 3: find distributions
			if distributions:
				distributions = self.find_per_topic_word_distributions(lda_model)
			

			# 4: return
			if distributions:
				return (lda_model, distributions)
			else:
				return lda_model




	####################################################################################################
	######################[ --- TRAINING --- ]##########################################################
	####################################################################################################

	def update_lda(self, df, fields, lda_model=None):
		'''
			Function: update_lda
			--------------------
			params: df - input df to update the lda model with
					field - the field to update the lda model with. Note that the lda model that will be updated
					is specifically the one trained on the fields given unless otherwise given
					lda_model - a trained LDA model 
			returns: updated lda model

			notes: if there is no current lda model trained, this builds one using the passed in dataframe
		'''
		if lda_model != None:
			return self.update_trained_lda_model(lda_model, df, fields)


		if (self.lda_is_trained(fields)):
			return self.update_trained_lda_model(self.get_lda_model(fields), df, fields)
		else:
			return self.build_lda_model(df, fields)










	####################################################################################################
	######################[ --- APPLICATION --- ]#######################################################
	####################################################################################################


	def gensim_bow_to_lda (self, lda_model, gensim_bow):
		"""
			Function: gensim_bow_to_lda
			---------------------------
			given an lda_model and a gensim bag of words, this returns 
			the *Full* lda vector, including close-to-zero elements 
		"""
		#=====[ Step 2: get lda vector	]=====
		gamma, sstats = lda_model.inference([gensim_bag])
		normalized_gamma = gamma[0] / sum(gamma[0])
		return normalized_gamma



	def add_lda (self, df, field, lda_model=None):
		"""
			Function: apply_lda
			-------------------
			given a dataframe, field name, and optionally an lda model,
			this adds a new lda column and returns the updated df
		"""

		#=====[ Step 1: picking correct LDA model to operate with	]=====
		if lda_model == None:
			if self.lda_is_trained(fields):
				lda_model = self.get_lda_model(fields)
			else:
				lda_model = self.build_lda_model(df, fields)


		#=====[ Step 2: define lda application function 	]=====
		def lda_inference(word_list):

			#=====[ Step 1: convert to gensim bow format	]=====
			gensim_bow = lda_model.id2word.doc2bow(word_list)

			#=====[ Step 2: convert to lda	]=====
			return self.gensim_bow_to_lda (lda_model, gensim_bow)


		df[self.fields_to_string(fields) + '_lda'] = df[field].apply(lda_inference, axis=1)
		return df







# ============================================[ PRIVATE FUNCTIONS ] ============================================ #


	def lda_inference(self, lda_model, fields, row):
		'''
			function: lda_inference

			params: lda_model - lda to run inference with
					fields - fields to run inference on
					row - row in a dataframe with fields to run inference on

			returns: normalized lda vector

			notes: used using a df's .apply method to generate a new column
		'''
		#1 : make bag representation (the value of each field is assumed to already be a list of words)
		bag_rep = []
		for field in fields:
			if type(row[field]) == list:
				bag_rep += row[field]	

		#2 : turn to gensim format
		gensim_bag = lda_model.id2word.doc2bow(bag_rep)

		#3 : get inference vector
		gamma, sstats = lda_model.inference([gensim_bag])
		normalized_gamma = gamma[0] / sum(gamma[0])

		#4 : return
		return normalized_gamma



	def update_trained_lda_model(self, lda_model, df, fields):
		'''
			function: update_trained_lda_model

			params: lda_model - model to update
					df - dataframe with documents to update the lda model with
					fields - fields to use to generate those documents

			returns: updated lda model

			notes: really not clear to me what happens if there is a word in the new corpus that wasn't in the originally
			trained lda corpus... I think that passing in existing dictionary like I did works because in generate_corpus
			i allow it to update. But if there is some wierd behavior we see this might be somewhere to look.
		'''
		#1 : generate corpus
		corpus, dictionary = self.generate_corpus(df, fields, existing_dictionary=lda_model.id2word)

		#2 : update lda model
		lda_model.update(corpus)

		#3 : return
		return lda_model


	def lda_is_trained(self, fields):
		'''
			function: lda_is_trained

			params: fields - the fields to check if an lda model has been trained on

			returns: true or false
		'''
		return self.fields_to_string(fields) in self.fields_to_lda_models



	def fields_to_string(self, fields):
		return "".join(fields)

	def find_per_topic_word_distributions(self, lda_model):
		'''
			function: find_per_topic_word_distributions

			params: lda_model - lda model to find distributions for

			returns: a list of dicts, the i'th dict mapping words -> probabilities for the i'th topic
		'''
		dist = []
		# 1: iterate through topics
		for topic in range(lda_model.num_topics): 
			topic_dist_dict = {}
			# 2: get probability distribution
			topic_dist = lda_model.state.get_lambda()[cluster] 
			# 3: normalize to real probability distribution
			topic_dist = topic_dist / topic_dist.sum()
			for i in range(len(topic_dist)):
				# 4: map the string id of the node to the probability (self.dict goes from gensim's id -> my string id)
				topic_dist_dict[lda_model.id2word[i]] = topic_dist[i] 
			# 5: append to array of dicts
			dist.append(topic_dist_dict) 
		return dist



	# copied above:

	# def get_lda_model(self, fields, distributions=False):
	# 	'''
	# 		function: get_lda_model

	# 		params: fields - list of fields that represent the LDA model you want
	# 				distributions - whether or not you want the distributions of the lda model

	# 		returns: the trained LDA model, as well as the per topics word distributions if requested
	# 	'''
	# 	# 1: make fields string
	# 	fields_string = self.fields_to_string(fields)

	# 	# 2: find lda model
	# 	if fields_string in self.fields_to_lda_models:
	# 		lda_model = self.fields_to_lda_models[fields_string]
	# 	else:
	# 		raise ValueError('There is no LDA model trained using those fields. If you want to train one, use update_lda')

	# 	# 3: find distributions
	# 	if distributions:
	# 		distributions = self.find_per_topic_word_distributions(lda_model)
		

	# 	# 4: return
	# 	if distributions:
	# 		return (lda_model, distributions)
	# 	else:
	# 		return lda_model








	def generate_corpus(self, df, fields, existing_dictionary=None):
		'''
			function: generate_corpus

			params: df - dataframe to generate corpus from
					fields - list of fields to concatenate
					existing_dictionary - existing dictionary for an lda model if one exists

			returns: gensim style corpus given by df and the fields given
		'''


		# 1: make a list of lists, to be filled by each row in df
		texts = []
		# texts = [[] for i in range(len(df))]

		# 2: for each field, concatenate each text with the fields contents (note: probably a better, pd way to do this)
		for i in range(len(df)):
			# get a row
			row = df.iloc[i]
			concat_string = []
			for field in fields:
				# catch nan's
				if type(row[field]) == list:
					concat_string += row[field]
				else:
					continue
				
			texts.append(concat_string)



		# for field in fields:
			# field_values = list(df[field])
			# texts = map(operator.add, texts, field_values)


		# 3: add the resulting texts as a new field in df (deprecated, left in case we need it later)
		# field_string = "".join(fields)
		# df[field_string] = pd.Series(texts)

		# 4: convert to gensim style objects and return
		if existing_dictionary == None:
			dictionary = gensim.corpora.Dictionary(texts)
			update=False
		else:
			print "HERE!!"
			dictionary = existing_dictionary
			update=True

		corpus = [dictionary.doc2bow(text, allow_update=update) for text in texts]
		return corpus, dictionary





	def build_lda_model(self, df, fields, num_topics=6):
		'''
			function: build_lda_model(df, fields)

			params: df - dataframe to train lda model on
					fields - list of fields corresponding to columns in the df to run lda on

			returns: gensim LDAModel class trained on the data given
		'''
		# 1: make corpus and dictionary
		corpus,dictionary = self.generate_corpus(df, fields)

		# 2: run lda and return
		lda = gensim.models.LdaModel(corpus, id2word=dictionary, num_topics=num_topics)
		self.fields_to_lda_models[self.fields_to_string(fields)] = lda
		return lda









