# Script: demo_recommendation
# ---------------------------
# script to demonstrate the recommendation capabilities
# of SpotOn class
import json
from SpotOn import SpotOn
from util import *

if __name__ == "__main__":
	print_header ("RECOMMENDATION DEMO")

	#=====[ Step 1: construct SpotOn object	]=====
	so = SpotOn ()
	so.semantic_analysis.load ()

	#=====[ Step 2: get user, activities ]=====
	all_activities = json.load (open('demo_activities.json', 'r'))
	activities = all_activities[:95]
	user = all_activities[95:]

	#=====[ Step 3: get scores for activities	]=====
	scored_activities = so.score_activities (user, activities)

	#=====[ DISPLAY USER ]=====
	print_header ("USER SUMMARY: EVENTS")
	for a in user:
		print "- ", a['_source']['name']

	#=====[ DISPLAY EVENTS	]=====
	# print_header ("RECOMMENDATIONS")
	# for a in activities 


	