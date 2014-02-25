# -------------------------------------------------------------------------------- #
# Activity Recommender for SpotOn
# -------------------------------
# Jay Hack, Ankit Kumar and Sam Beder
# Winter 2014
# -------------------------------------------------------------------------------- #

0. To Do:
---------
• Remove numbers from calendar event titles
• 
• find correspondences between activities_df and calendar_events_df





============================
I. Installation/Dependencies
============================
• This entire project was developed in ipython; we recommend you run it in ipython as well
• It relies on the following packages:
	- pandas
	- numpy
	- sklearn


=============
I. Data Specs
=============
• All data is located in ./data, including both the original (raw, as received) json files and our curated
pandas dataframes
• Relevant information contained in json_obj['hits']['hits']['_source'], which will be a list of dicts representing calendar events, etc.

A. Calendar:
------------
• original: es-calendar.json, curated pandas dataframe in calendar.df


B. Recommended Activities:
--------------------------
• original: es-activity.json, curated pandas dataframe in activity.df
• 
