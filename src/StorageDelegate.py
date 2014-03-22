import os
import sys
import pandas as pd

class StorageDelegate:

	#==========[ Filestructure Outline	]==========
	filestructure = {'data':'../data'}
	filestructure ['pandas'] = os.path.join (filestructure['data'], 'pandas')
	filestructure ['pandas_c'] = os.path.join(filestructure['pandas'], 'calendars')	
	filestructure ['pandas_a'] = os.path.join(filestructure['pandas'], 'activities')
	filestructure ['models'] = os.path.join (filestructure['data'], 'word2vec')
	filestructure_create_order = 	[
										'data', 
										'pandas', 'pandas_c', 'pandas_a', 
										'models'
									]

	def __init__ (self):
		"""
			PUBLIC: Constructor 
			-------------------
			ensures that the filestructure exists and is well-formatted
		"""
		self.ensure_filestructure_exists ()


	####################################################################################################
	######################[ --- FILESYSTEM STRUCTURE --- ]##############################################
	####################################################################################################

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
				/pandas
					/calendar_events
					/activities
				/models
		"""
		for dir_name in self.filestructure_create_order:
			self.ensure_dir_exists(self.filestructure[dir_name])






	####################################################################################################
	######################[ --- LOADING/SAVING --- ]####################################################
	####################################################################################################

	def save_calendar_df (self, df, name):
		"""
			PRIVATE: save_calendar_df
			-------------------------
			saves the calendar df according to 'name'
		"""
		path = os.path.join (self.filestructure['pandas_c'], name)
		df.to_pickle (path)


	def get_calendar_df (self, name):
		"""
			PRIVATE: get_calendar_df
			------------------------
			gets the calendar df that goes by 'name'
		"""
		path = os.path.join (self.filestructure['pandas_c'], name)
		return pd.read_pickle (path)


	def save_activity_df (self, df, name):
		"""
			PRIVATE: save_activity_df
			-------------------------
			saves the activity df according to 'name'
		"""
		path = os.path.join (self.filestructure['pandas_a'], name)
		df.to_pickle (path)


	def get_activity_df (self, name):
		"""
			PRIVATE: get_activity_df
			------------------------
			gets the calendar df that goes by 'name'
		"""
		path = os.path.join (self.filestructure['pandas_a'], name)
		return pd.read_pickle (path)







	####################################################################################################
	######################[ --- GETTING CALENDARS/ACTIVITIES --- ]######################################
	####################################################################################################

	def get_calendar_df_names (self):
		"""
			PRIVATE: get_calendar_filepaths
			-------------------------------
			returns a list of the filepaths of all calendar dataframes
		"""
		return os.listdir (self.filestructure['pandas_c'])


	def iter_calendar_dfs (self):
		"""
			PRIVATE: iter_calendars
			-----------------------
			loads calendars into memory one by one, iterates 
			over them
		"""
		for name in self.get_calendar_df_names():
			if not name[0] == '.':
				yield self.get_calendar_df (name)


	def get_activity_df_names (self):
		"""
			PRIVATE: get_activity_filepaths
			-------------------------------
			returns a list of the filepaths of all activity dataframes
		"""
		return os.listdir(self.filestructure['pandas_a'])


	def iter_activity_dfs (self):
		"""
			PRIVATE: iter_activities
			------------------------
			loads activities into memory one by one, iterates 
			over them
		"""
		for name in self.get_activity_df_names ():
			if not name[0] == '.':
				yield self.get_activity_df (name)







if __name__ == '__main__':

	sd = StorageDelegate ()

	#=====[ Example: iter over calendars	]=====
	# for cdf in sd.iter_calendar_dfs ():
		# print cdf.iloc[0]

	#=====[ Example: iter over activities	]=====
	# for adf in sd.iter_activity_dfs ():
		# print adf.iloc[0]











