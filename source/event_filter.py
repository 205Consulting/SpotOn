import json
import pickle

#load events
events = json.load(open('calendars_0_till_1000000.json'))

mapping = {}
curr_index = 0
for event in events:
	try:


		if event['_source']['type'] != 'spoton':
			print "not spoton; %s" % event['_source']['type']
			continue
		if event['_source']['description'] in mapping:
			print "in mapping"
			continue
		else:
			print "adding to map"
			mapping[event['source']['description']] = curr_index
			curr_index += 1
	except:
		print "exception"
		continue


pickle.dump(mapping, open('mapping.pkl', 'wb'))