import pandas as pd
from time import strptime, mktime
from Preprocess import Preprocess
from util import print_header, print_status, print_inner_status

class UserAnalysis:

	def __init__ (self):
		"""
			PUBLIC: constructor
			-------------------
		"""
		self.user_representations = {}


	def init_user_representation (self, user_id=None):
		"""
			PRIVATE: init_user_representation
			---------------------------------
			given a user id, returns an empty json dict 
			representing him or her 
		"""
		return {
					'id': user_id,
					'event_ids': [],
					'event_start_dates': [],
					'event_locations': [],
					'all_event_names': [],				# all concatenated words ever
					'all_event_words': []
				}


	def update_user_representation (self, event):
		"""
			PRIVATE: update_user_representation
			-----------------------------------
			given a row representing a calendar event, this function 
			will update self.user_representations
		"""
		#=====[ Step 1: retrieve user representation	]=====
		user_id = event['user']
		user_rep = self.user_representations[user_id]

		#=====[ Step 3: add relevant data	]=====
		# user_rep['id'] = user_id
		# user_rep['event_ids'].append (event['id'])
		user_rep['event_names'].append (event['name'])					# Note: tokenize this?
		user_rep['event_descriptions'].append (event['description'])	# Note: tokenize this?
		user_rep['event_start_dates'].append (event['date_start'])		# Note: only considering start time
		user_rep['event_locations'].append (event['location'])		# Note: most are timezone
		# user_rep['event_responses'].append (event['response'])		# Note: only considering start time		
		# user_rep['event_word2vecs'].append (event['word2vec'])
		if type(event['name']) == list:
			user_rep['all_event_names'] += event['name']
		if type(event['description']) == list:
			user_rep['all_event_words'] += event['description']


	def extract_users (self, calendar_df_iterator):
		"""
			given an iterator over calendar dataframes,
			this constructs and returns a dataframe 
			containing all users
		"""
		print_header ("EXTRACTING USERS")
		#==========[ ITERATE OVER ALL DFS	]==========
		for cdf in calendar_df_iterator ():
			print_status ("Extract users", "next df")

			#=====[ Step 1: sort by user	]=====
			print_inner_status ("extract_users", "sorting by user id")
			cdf = cdf.sort ('user')

			#=====[ Step 2: init user representations	]=====
			print_inner_status ("extract_users", "initializing user representations")
			unique_uids = [uid for uid in cdf['user'].unique ()]
			for uid in unique_uids:
				if not uid in self.user_representations:
					self.user_representations[uid] = self.init_user_representation(uid)

			#=====[ Step 3: update the user representations	]=====
			print_inner_status ("extract_users", "updating user representations")			
			cdf.apply (self.update_user_representation, axis = 1)

		#=====[ Step 4: convert to df, delete irrelevant stuff	]=====
		print_inner_status ("extract_users", "converting to dataframe")		
		self.users_df = pd.DataFrame(self.user_representations.values())
		del self.user_representations
		return self.users_df


	def extract_users_spoton (self, calendar_df_iterator):
		"""
			PUBLIC: extract_users_spoton
			----------------------------
			given an iterator over calendar dataframes,
			this constructs and returns a dataframe 
			containing all users
		"""
		print_header ("EXTRACTING USERS")
		#==========[ ITERATE OVER ALL DFS	]==========
		for cdf in calendar_df_iterator ():
			print_status ("Extract users", "next df")

			idx = (cdf['created'] == 'spoton')

			#=====[ Step 1: sort by user	]=====
			print_inner_status ("extract_users", "sorting by user id")
			cdf = cdf.sort ('user')

			#=====[ Step 2: init user representations	]=====
			print_inner_status ("extract_users", "initializing user representations")
			unique_uids = [uid for uid in cdf['user'].unique ()]
			for uid in unique_uids:
				if not uid in self.user_representations:
					self.user_representations[uid] = self.init_user_representation(uid)

			#=====[ Step 3: update the user representations	]=====
			print_inner_status ("extract_users", "updating user representations")			
			cdf.apply (self.update_user_representation, axis = 1)

		#=====[ Step 4: convert to df, delete irrelevant stuff	]=====
		print_inner_status ("extract_users", "converting to dataframe")		
		self.users_df = pd.DataFrame(self.user_representations.values())
		del self.user_representations

		return self.users_df



	def add_num_events (self, u_df):
		"""
			PUBLIC: add_num_events
			----------------------
			adds a column for the number of events 
		"""
		def get_num_events (event_ids):
			return len(event_ids)
		u_df['num_events'] = u_df['event_ids'].apply (get_num_events)
		return u_df


	def add_timespan (self, u_df):
		"""
			PUBLIC: add_timespan
			--------------------
			given a user dataframe, this will add a field 
			for the total elapsed time they have been using 
			SpotOn
		"""
		def get_timespan (event_start_date):
			
			#=====[ Step 1: get min/max date	]=====
			dmin, dmax = min(event_start_date), max(event_start_date)
			# print "=====[ user	]====="
			# print dmin, dmax

			#=====[ Step 2: parse dates	]=====
			dmin_day = strptime(dmin.split('T')[0], '%Y-%m-%d')
			dmax_day = strptime(dmax.split('T')[0], '%Y-%m-%d')
			# print dmin_day, dmax_day

			#=====[ Step 3: find/return difference	]=====
			elapsed_time =  ((mktime(dmax_day) - mktime(dmin_day)) / (86400))
			elapsed_time += 1
			return elapsed_time

		u_df['elapsed_days_active'] = u_df['event_start_dates'].apply (get_timespan)
		u_df['elapsed_weeks_active'] = u_df['elapsed_days_active'] / float(7.0)

		return u_df


	def add_average_weekly_events (self, u_df):
		"""
			PUBLIC: add_average_weekly_events
			---------------------------------
			given a user dataframe, adds for each user the average 
			number of events they have been to per week.
		"""
		#=====[ Step 1: check dependencies	]=====
		if not 'num_events' in u_df:
			u_df = self.add_num_events (u_df)
		if not 'elapsed_weeks_active' in u_df:
			u_df = self.add_timespan (u_df)

		#=====[ Step 2: calculate average, return ]=====
		u_df['average_weekly_events'] = u_df['num_events'] / u_df['elapsed_weeks_active']
		return u_df


	def get_avg_weekly_events_histogram (self, u_df):
		"""
			PUBLIC: get_avg_weekly_events_histogram
			---------------------------------------
			returns a numpy histograrm of the average number of events 
			a user attends per week 
		"""
		pass


	def user_from_events_dft (self, user_events_df):
		"""
			PUBLIC: user_from_events_json_list
			----------------------------------
			given a list of calendar events represented as json, this 
			returns a pandas series representing a user who has gone 
			to all of them
		"""
		#=====[ Step 1: initialize user representation	]=====
		user = self.init_user_representation ()

		#=====[ Step 2: preprocess events_json_list	]=====
		events_df = Preprocess().Preprocess (events_json_list)

		#=====[ Step 3: get corpus, text	]=====
		user_text = []









