import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

input_data = '/Users/saiyingge/Resume_Study_DATA/group_discussion/discussion_data_add_speakers.csv'
playerList = ['Alpha', 'Bravo', 'Charlie', 'Delta', 'Echo']
deceiversList = ['Alpha', 'Delta']
variable_List = ['smile'] + ['pitch_angle', 'roll_angle', 'yaw_angle'] + ['anger', 'disgust', 'fear',
       'happiness', 'neutral', 'sadness', 'surprise']
info_List = ['game','speaker','stage','imgindex','player']

df = pd.read_csv(input_data, dtype={19: 'object', 20: 'object'})


df.info()





# add a column to indicate whether the interaction status of the player
#  speaking: player is also the speaker_1
#  interacting: player in speaker2,3,4,5
#  listening: player is not in the speaker list
#  Silence: all nan
df['interaction'] = df.apply(
       lambda x: 'Speaking' if x['player'] == x['speaker_1']
       else 'Interacting' if x['player'] in x[['speaker_2','speaker_3','speaker_4','speaker_5']].values
       else 'Listening' if not pd.isnull(x['speaker_1']) else 'Silence', axis=1)