import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# import statsmodels.api as sm
# from statsmodels.formula.api import ols
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

input_data = '/Users/saiyingge/Resume_Study_DATA/group_discussion/discussion_data_add_speakers.csv'
playerList = ['Alpha', 'Bravo', 'Charlie', 'Delta', 'Echo']
deceiversList = ['Alpha', 'Delta']
variable_List = ['smile'] + ['pitch_angle', 'roll_angle', 'yaw_angle'] + ['anger', 'disgust', 'fear',
       'happiness', 'neutral', 'sadness', 'surprise']
info_List = ['game','speaker','stage','imgindex','player']

# read the group discussion data, where the idx_timestamp is int

df = pd.read_csv(input_data, dtype={19: 'object', 20: 'object'})
df.info()

# only keep ['pitch_angle', 'roll_angle', 'yaw_angle'] and change from wide to long
df = df[info_List + ['pitch_angle', 'roll_angle', 'yaw_angle'] + ['interaction']+['idx_timestamp']]
df = df.melt(id_vars=info_List + ['interaction']+['idx_timestamp'], value_vars=['pitch_angle', 'roll_angle', 'yaw_angle'], var_name='head_p', value_name='head_angle')
# change the head_p to pitch, roll, yaw
df['head_p'] = df['head_p'].str.replace('_angle', '')

#group by game, speaker, stage, head_p, interaction, idx_timestamp
#  count average numbers of frames in each group
temp = df.groupby(info_List + ['head_p',  'idx_timestamp']).count().reset_index()
# add to do list
# 1. add the average numbers of frames in each group/ the terms
#  get the variance
# variance_all along the time