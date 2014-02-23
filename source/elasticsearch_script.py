import json
import requests
import parameters
import sys
import argparse
import time

user = parameters.elasticsearch_username
pw = parameters.elasticsearch_password



def main(argv):
	# get arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("endpoint", help="which endpoint to access ('calendars' or 'activities'")
	parser.add_argument("start", help="start index", type=int)
	parser.add_argument("upper_limit", help="how many items we want to get",
                    type=int)
	
	args = parser.parse_args()
	# find endpoint
	endpoint = None
	if args.endpoint == 'activities':
		endpoint = parameters.elasticsearch_activities_endpoint
	elif args.endpoint == 'calendars':
		endpoint = parameters.elasticsearch_calendars_endpoint
	else:
		print "endpoint should be 'activities' or 'calendars'. try again."
		sys.exit()


	upper_limit = args.upper_limit
	start = args.start



	# page through endpoint, appending data to a master list
	master_data = []
	last = 0
	while (len(master_data) < upper_limit):
		try:
			print "CURRENTLY PULLING: %d to %d" % (len(master_data) + start, len(master_data) + start + 500)
			params = {'from': len(master_data) + start, 'to':len(master_data) + start + 500, 'size':500}
			request = requests.get(endpoint, auth=(user, pw), params=params)
			data = request.json()['hits']['hits']
			master_data += data
			last = len(master_data) + start
		# keyboardinterrupt breaks the while loop and dumps the data we got so far
		except KeyboardInterrupt:
			print "keyboard interrupt"
			break
		# other exceptions are assumed to be, for example, accessing api too much or something and are printed but ignored.
		except Exception as e:
			print "Non keyboardInterrupt exception detected: %s" % e
			continue

	json.dump(master_data, open('%s_%s_till_%s.json' % (args.endpoint, start, last),'w'))





	



	# current_data = []
	# while (len(current_data) < )



if __name__ == '__main__':
	main(sys.argv)