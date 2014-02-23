import os
elasticsearch_username = 'ml'
elasticsearch_password = '9g*V(7fB0+mc8Lz(7Z!a'
elasticsearch_activities_endpoint = 'https://spoton.it/api/sources/activities'
elasticsearch_calendar_endpoint = 'https://spoton.it/api/sources/calendars'



filenames = {
				'data_dir':os.path.join(os.getcwd(), '../data'),
				'calendar_df':os.path.join (os.getcwd (), '../data/calendar.df'),
				'activity_df':os.path.join (os.getcwd(), '../data/activity_df')
			}