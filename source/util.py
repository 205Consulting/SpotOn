from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import stopwords


####################################################################################################
######################[ --- NLP --- ]###############################################################
####################################################################################################

def tokenize_remove_stopwords (s):
	"""
		PRIVATE: tokenize_remove_stopwords
		----------------------------------
		given a string s,
		- tokenizes via nltk.wordpunct_tokenize
		- removes stopwords 
		- converts all words to lowercase 
	"""
	return [w.lower() for w in wordpunct_tokenize(s) if not w in stopwords.words('english')]











####################################################################################################
######################[ --- INTERFACE --- ]#########################################################
####################################################################################################

# UTIL: print_welcome
# -------------------
# prints out a welcome message
def print_welcome ():
	print "###########################################################"
	print "#####[ SpotOn Calendar Event Recommendation System ]#####"
	print "#####[ =========================================== ]#####"
	print "#####[ by Jay Hack, Sam Beder, Ankit Kumar	]#####"
	print "#####[ Winter 2014					]#####"
	print "###########################################################"
	print "\n"


# UTIL: print_header
# ------------------
# prints the specified header text in a salient way
def print_header (header_text):	
	print "-" * (len(header_text) + 12)
	print ('#' * 5) + ' ' +  header_text + ' ' + ('#' * 5)
	print "-" * (len(header_text) + 12)


# UTIL: get_(sub)object_header
# ----------------------------
# returns a header string for the specified (sub)object
# used in printing out objects
def get_object_header (obj_name):
	return "###[ " + str(obj_name) + " ]###\n"

def get_subobject_header (obj_name):
	return "	> " + str(obj_name) + ": "


# UTIL: print_error
# -----------------
# prints an error and exits 
def print_error (top_string, bottom_string):
	print "Error: " + top_string
	print "---------------------"
	print bottom_string
	exit ()


# UTIL: print_status
# ------------------
# prints out a status message 
def print_status (stage, status):	
	print "-----> " + stage + ": " + status


# UTIL: print_inner_status
# ------------------------
# prints out a status message for inner programs
def print_inner_status (stage, status):
	print "	-----> " + stage + ": " + status


# UTIL: print_notification
# ------------------------
# prints a notification for the user
# using the format '	>>> [notification text] <<<'
def print_notification (notification_text):
	print "	>>> " + notification_text + " <<<"



