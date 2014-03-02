import os
import sys

class StorageDelegate:

	#==========[ Filestructure Outline	]==========
	filestructure = {'data':'../data'}
	filestructure ['json'] = os.path.join(filestructure['data'], 'json')
	filestructure ['json_ce'] = os.path.join(filestructure['json'], 'calendar_events')
	filestructure ['json_a'] = os.path.join(filestructure['json'], 'activities')	
	filestructure ['pandas'] = os.path.join(filestructure['data'], 'df')	
	filestructure ['pandas_ce'] = os.path.join(filestructure['pandas'], 'calendar_events')	
	filestructure ['pandas_a'] = os.path.join(filestructure['pandas'], 'activities')
	filestructure ['word2vec'] = os.path.join (filestructure['data'], 'word2vec')
	filestructure ['gensim'] = os.path.join (filestructure['data'], 'pandas')
	filestructure_create_order = [
									'data', 
									'json', 'json_ce', 'json_a', 
									'pandas', 'pandas_ce', 'pandas_a', 
									'word2vec',
									'gensim']


	def __init__ (self):
		"""
			PUBLIC: Constructor 
			-------------------
			ensures that the filestructure exists and is well-formatted
		"""
		self.ensure_filestructure_exists ()


	def ensure_dir_exists (self, path):
		"""
			PRIVATE: assert_dir
			-------------------
			given a filepath, ensures that it exists 
		"""
		if not os.path.isdir (path):
			os.mkdir (path)


	def ensure_filestructure_exists (self):
		"""
			PRIVATE: ensure_filestructure_exists
			-----------------------------
			ensures that the filestructure is correct. looks like as follows:
			../data
				/json
					/calendar_events
					/activities 
				/pandas
					/calendar_events
					/activities
				/word2vec
				/gensim
		"""
		for dir_name in filestructure_create_order:
			ensure_dir_exists(filestructure['dir_name'])

