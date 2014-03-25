# Script: demo_recommendation
# ---------------------------
# script to demonstrate the recommendation capabilities
# of SpotOn class
import json
from SpotOn import SpotOn
from util import *

if __name__ == "__main__":
	print_header ("RECOMMENDATION DEMO")

	#=====[ Step 1: construct/load SpotOn object	]=====
	so = SpotOn ()
	so.load ()

	#=====[ Step 2: get user, activities ]=====
	all_activities = json.load (open('demo_activities.json', 'r'))
	user = [	
				all_activities[9], 
				all_activities[16], 
				all_activities[21], 
				all_activities[37]
			]
	activities = all_activities


	#=====[ DISPLAY USER ]=====
	print_header ("USER REP:")
	for activity in user:
		print activity['_source']['name'], " | ", activity['_source']['description']
		print '\n', '=' * 40, '\n'
	print "\n\n\n"

	#=====[ Step 3: get/display scores for activities	]=====
	scored_activities = so.score_activities (user, activities)
	sorted_activities = sorted(zip(activities, scored_activities), key=lambda x: x[1])
	print_header ("ACTIVITY RECOMMENDATIONS: ")
	for activity, score in sorted_activities:
		print score, ": ", activity['_source']['name'], " | ", activity['_source']['description']
		print '\n', '=' * 40, '\n'



	