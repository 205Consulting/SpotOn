from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import stopwords

####################################################################################################
######################[ --- INTERFACE --- ]#########################################################
####################################################################################################

def print_welcome ():
	"""
		UTIL: print_welcome
		-------------------
		prints out a welcome message 
	"""
	print "###########################################################"
	print "#####[ SpotOn Calendar Event Recommendation System ]#####"
	print "#####[ =========================================== ]#####"
	print "#####[ by Jay Hack, Sam Beder, Ankit Kumar	]#####"
	print "#####[ Winter 2014					]#####"
	print "###########################################################"
	print "\n"


def print_header (header_text):	
	"""
		UTIL: print_header
		------------------
		prints the specified header text in a salient way
	"""
	print "-" * (len(header_text) + 12)
	print ('#' * 5) + ' ' +  header_text + ' ' + ('#' * 5)
	print "-" * (len(header_text) + 12)


def get_object_header (obj_name):
	"""
		UTIL: get_(sub)object_header
		----------------------------
		returns a header string for the specified (sub)object
		used in printing out objects
	"""
	return "###[ " + str(obj_name) + " ]###\n"

def get_subobject_header (obj_name):
	return "	> " + str(obj_name) + ": "


def print_error (top_string, bottom_string):
	"""
		UTIL: print_error
		-----------------
		prints an error and exits 
	"""
	print "Error: " + top_string
	print "-" * len ("Error: " + top_string)
	print bottom_string
	print "\n\n"
	exit ()


def print_status (stage, status):	
	"""
		UTIL: print_status
		------------------
		prints out a status message
	"""
	print "-----> " + stage + ": " + status


def print_inner_status (stage, status):
	"""
		UTIL: print_inner_status
		------------------------
		prints out a status message for inner programs
	"""
	print "	-----> " + stage + ": " + status


def print_notification (notification_text):
	"""
		UTIL: print_notification
		------------------------
		prints a notification for the user
		using the format '	>>> [notification text] <<<'
	"""
	print "	>>> " + notification_text + " <<<"



####################################################################################################
######################[ --- FILES --- ]#############################################################
####################################################################################################

def is_json (filename):
	return (filename[-5:] == '.json')


def is_dataframe (filename):
	return (filename[-3:] == '.df')

