# Script: demo_recommendation
# ---------------------------
# script to demonstrate the recommendation capabilities
# of SpotOn class
import json
from SpotOn import SpotOn
import numpy as np
from util import *

if __name__ == "__main__":
	print_header ("RECOMMENDATION DEMO")

	#=====[ Step 1: construct/load SpotOn object	]=====
	so = SpotOn ()
	so.load ()

	#=====[ Step 2: get user representation(s) ]=====
	ce_json = json.load (open('demo_calendar_events.json', 'r'))
	user_mother = [		
						ce_json[1], 	# zumba class
						ce_json[28],	# yoga class
						ce_json[9], 	# family shenanigans
						ce_json[8] 		# women in business
					]
	user_mother = so.calendar_events_to_user_representation (user_mother)

	user_graduate = [	
						ce_json[36],	# wine/drink night
						ce_json[55], 	# running and walking in austin
					]
	user_graduate = so.calendar_events_to_user_representation (user_graduate)

	user_rep = user_mother
	#####[ DISPLAY USER ]#####
	print_header ("USER REP:")
	for i in range(len(user_rep['events_df'])):
		print 'Event: ', ' '.join(user_rep['events_df'].iloc[i]['name'])
	print '\n\n'
	#####[ END DISPLAY USER	]#####
	

	#=====[ Step 3: get activities to recommend	]=====
	activities_json = json.load (open('demo_activities.json', 'r'))






	####################################################################################################
	######################[ --- Exposed API TEST --- ]##################################################
	####################################################################################################

	#####[ TEST: score activity for user	]#####
	print_header("Test 1: score_activity_for_user")
	score = so.score_activity_for_user(user_rep, activities_json[6])
	print activities_json[6]['_source']['name'], ': ', score
	print '\n\n'

	#####[ TEST: load_activities_corpus and recommend_for_user	]#####
	print_header("Test 2: recommend_for_user")
	so.load_activities_corpus(activities_json)
	activity_ranks = so.recommend_for_user(user_rep)
	for rank, index in enumerate(activity_ranks[:50]):
		print rank, ': ', activities_json[index]['_source']['name']
	print '\n\n'

	#####[ TEST: recommend_users_for_activity	]#####
	print_header("Test 3: recommend_users_for_activity")
	top_users = so.recommend_users_for_activity(activities_json[6], [user_mother, user_graduate], topn=2)
	print '#####[ Activity: ]#####'
	print activities_json[6]['_source']['name']
	print '\n'
	print '#####[ Top Users ]#####'
	for user in top_users:
		print '===[ User: ]==='
		print user['events_df']['name']
		print "\n"

