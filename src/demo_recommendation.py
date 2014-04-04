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
	#=====[ Step 2: get user representation ]=====
	ce_json = json.load (open('demo_calendar_events.json', 'r'))
	user_events = [ce_json[1], ce_json[9], ce_json[8], ce_json[28]]
	user_rep = so.calendar_events_to_user_representation (user_events)
	######[ DISPLAY USER ]#####
	print_header ("USER REP:")
	for event in user_events:
		print event['_source']['name'], " | ", event['_source']['description']
		print '\n', '=' * 40, '\n'
	print "\n\n\n"
	#=====[ Step 3: get activities to recommend	]=====
	a_json = json.load (open('demo_activities.json', 'r'))


	#==========[ TESTS	]==========

	#=====[ TEST 1: score activity	]=====
	print_header ("SCORE ACTIVITIES:")
	for activity in a_json[-100:]:
		score = so.score_activity_for_user (user_rep, activity)
		print score, ': ', activity['_source']['name']

	#=====[ TEST 2: score activities	]=====
	print_header ("USER RECOMMENDATIONS")
	top_scores = so.recommend_for_user (user_rep, a_json)
	print top_scores


	# #=====[ Step 3: get/display scores for activities	]=====
	# scored_activities, returned_activities = so.score_activities (user, activities)
	# sorted_activities = np.argsort(scored_activities)[::-1]
	# print_header ("ACTIVITY RECOMMENDATIONS: ")
	# for i in range(5): #change 5 to see more results
	# 	activity_index = sorted_activities[i]
	# 	print i, ": ", returned_activities.iloc[activity_index]
	# 	print '\n', '='*40, '\n'



	