import pandas as pd
from LDATrainer import LDATrainer
ce_df = pd.read_pickle('calendar_events_old.df')
a_df = pd.read_pickle('activities.df')
u_df = pd.read_pickle('users.df')


trainer = LDATrainer(ce_df, a_df, u_df)
