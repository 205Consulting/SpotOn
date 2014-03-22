import os
elasticsearch_username = 'ml'
elasticsearch_password = '9g*V(7fB0+mc8Lz(7Z!a'
elasticsearch_activities_endpoint = 'https://spoton.it/api/sources/activities'
elasticsearch_calendars_endpoint = 'https://spoton.it/api/sources/calendars'



filenames = {
				'data_dir':os.path.join(os.getcwd(), '../data'),
				'calendar_df':os.path.join (os.getcwd (), '../data/calendar.df'),
				'activity_df':os.path.join (os.getcwd(), '../data/activity_df')
			}

timezones = set([	u'America/Anchorage',
					u'America/Chicago',
					u'America/Dawson',
					u'America/Dawson_Creek',
					u'America/Denver',
					u'America/Detroit',
					u'America/Edmonton',
					u'America/Halifax',
					u'America/Indiana/Indianapolis',
					u'America/Los_Angeles',
					u'America/Montreal',
					u'America/New_York',
					u'America/Phoenix',
					u'America/Toronto',
					u'America/Vancouver',
					u'America/Whitehorse',
					u'America/Winnipeg',
					u'US/Eastern',
					u'US/Pacific',
					u'UTC'
				])
