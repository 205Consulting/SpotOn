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

	# Ankit's Step 2: Get users
	print_header("Test one: pass a list of calendar events for one user -> create a representation of the user that you use")
	activities_json = json.load(open('activities_new.json'))
	user_names = ['shopping', 'music', 'community']
	user_events = [[activities_json[2], activities_json[4], activities_json[7], activities_json[14]], [activities_json[21], activities_json[51], activities_json[57], activities_json[220]], [activities_json[94], activities_json[196], activities_json[101], activities_json[365]]]
	user_reps = [so.calendar_events_to_user_representation(user) for user in user_events]

	# display users:
	print_header("USER REPS:")
	for user in user_reps:
		print user['events_df']['words']
		print "\n\n"

	# TESTS --------------

	# Test 2: score_activity_for_user
	print_header("Test two: pass the representation of the user that you use + one activity -> return a score for how much they'd like it (or a yes/no)")
	score = so.score_activity_for_user(user_reps[0], activities_json[2])
	print score

	# Test 3: load_activities_corpus and recommend_for_user
	print_header("Test three: pass the representation of the user that you use -> return recommendations")
	# load activities to recommend
	so.load_activities_corpus(activities_json)
	top_scores = so.recommend_for_user(user_reps[1])
	print "Top scores: "
	for score in top_scores:
		print score
		print "\n"

	# Test 4: recommend_users_for_activity
	print_header("Test four: pass one activity and a list of users (represented your way) -> filter the list of users to get the ones who would like that activity")
	top_users = so.recommend_users_for_activity(activities_json[2], user_reps, topn=3)
	print "Top users (shopping): "
	for user in top_users:
		print user['events_df']['words']
		print "\n"

	top_users = so.recommend_users_for_activity(activities_json[21], user_reps, topn=3)
	print "Top users (music): "
	for user in top_users:
		print user['events_df']['words']
		print "\n"





	# #=====[ Step 2: get user representation ]=====
	# ce_json = json.load (open('demo_calendar_events.json', 'r'))
	# user_events = [ce_json[1], ce_json[9], ce_json[8], ce_json[28]]
	# user_rep = so.calendar_events_to_user_representation (user_events)
	# ######[ DISPLAY USER ]#####
	# print_header ("USER REP:")
	# for event in user_events:
	# 	print event['_source']['name'], " | ", event['_source']['description']
	# 	print '\n', '=' * 40, '\n'
	# print "\n\n\n"
	# #=====[ Step 3: get activities to recommend	]=====
	# a_json = json.load (open('demo_activities.json', 'r'))


	# #==========[ TESTS	]==========

	# #=====[ TEST 1: score activity	]=====
	# print_header ("SCORE ACTIVITIES:")
	# for activity in a_json[-100:]:
	# 	score = so.score_activity_for_user (user_rep, activity)
	# 	print score, ': ', activity['_source']['name']

	# #=====[ TEST 2: score activities	]=====
	# print_header ("USER RECOMMENDATIONS")
	# top_scores = so.recommend_for_user (user_rep, a_json)
	# print top_scores


	# # #=====[ Step 3: get/display scores for activities	]=====
	# # scored_activities, returned_activities = so.score_activities (user, activities)
	# # sorted_activities = np.argsort(scored_activities)[::-1]
	# print_header ("ACTIVITY RECOMMENDATIONS: ")
	# for i in range(5): #change 5 to see more results
	# 	activity_index = sorted_activities[i]
	# 	print i, ": ", returned_activities.iloc[activity_index]
	# 	print '\n', '='*40, '\n'



	