import pandas as pd
import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# import statsmodels.api as sm
# from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

input_data = '/Users/saiyingge/Resume_Study_DATA/group_discussion/discussion_data_add_speakers.csv'
playerList = ['Alpha', 'Bravo', 'Charlie', 'Delta', 'Echo']
deceiversList = ['Alpha', 'Delta']
variable_List = ['smile'] + ['pitch_angle', 'roll_angle', 'yaw_angle'] + ['anger', 'disgust', 'fear',
       'happiness', 'neutral', 'sadness', 'surprise']
info_List = ['game','speaker','stage','imgindex','player']

frame_info = pd.read_csv('/Users/saiyingge/Resume_Study_DATA/group_discussion/compiled_data.csv')
game_info=pd.read_csv('/Users/saiyingge/**Research Projects/Resume/data/qualtrics_manual.csv')

result_save_path = '/Users/saiyingge/Resume_Study_DATA/group_discussion/analysis_result/'



# descriptive statistics
# TODO: 1.  include some statistic about the interaction:
#  add the average numbers of frames in each group/ the terms
# read all frames info from complied file

# group by game level count, average, std, min, max
frame_info.groupby(['game'])['idx_timestamp'].count().reset_index().describe()

#calculate the diff between start_y and end_y
frame_info['seconds'] = (frame_info['end_y'] - frame_info['start_y'])/25
#group by game and idx_timestamp, get the second， pick the first one ( should be the same for each group)
timestamp_info=frame_info.groupby(['game','idx_timestamp'])['seconds'].first().reset_index()

#for each game, get the average length of speech and the count of speech
game_level = timestamp_info.groupby('game')['seconds'].agg(['mean', 'count', 'sum']).reset_index()
# calculate the Silence time for each game silence time = total time(the largest frame- smallest frame)/25 - active time
total_time_series = (frame_info.groupby('game')['end_y'].max() - frame_info.groupby('game')['start_y'].min()) / 25
game_level['total_time'] = total_time_series.reset_index(drop=True)
game_level['silence_time'] = game_level['total_time'] - game_level['sum']
game_level['silence_time_percent'] = game_level['silence_time']/game_level['total_time']

#merge the frame info with game info, on game name and player name from frame info, game and spaker name from game info
game_level = pd.merge(game_level, game_info.groupby(['group_id'])['winner'].first().reset_index(), left_on=['game'], right_on=['group_id'], how='left')

#plot the number of interaction in each game, sort by the number of interaction(count),  color the game by winner
game_level.sort_values(by=['count']).plot.bar(x='game', y='count', color=game_level['winner'].map({0: 'blue',1: 'red'}))
colors = np.where(game_level['winner']==0, '#FFD700', '#20B2AA')
# add legend for the colors
plt.legend(handles=[Patch(facecolor='blue', label='Winner: Truth Teller'),
                   Patch(facecolor='red', label='Winner: Deceiver')])

#add a line in the plot, show the average number of frames, include the number in the plot
plt.axhline(y=game_level['count'].mean(), color='r', linestyle='-')
plt.show()
# save the plot
plt.savefig(result_save_path+'Game_interaction.png')


# Sort the game_level by total_time
game_level = game_level.sort_values(by='total_time', ascending=False)

plt.figure(figsize=(12, 6))

# Plot the active time with a specific pattern
plt.bar(game_level['game'], game_level['sum'], label='Active Time', color=colors, hatch="/")

# Plot the silence time on top of active time with another pattern
plt.bar(game_level['game'], game_level['silence_time'], bottom=game_level['sum'], label='Silence Time', color=colors, hatch="x")

# Annotate bars with the silence time percentage
for idx, game in enumerate(game_level['game']):
    plt.text(idx, game_level['sum'].iloc[idx] + (game_level['silence_time'].iloc[idx] / 2), f"{game_level['silence_time_percent'].iloc[idx]:.2f}%", ha='center', va='center', fontsize=9)

plt.xlabel('Game')
plt.ylabel('Total Time')
plt.title('Active Time vs Silence Time in Each Game')
plt.xticks(rotation=45)
plt.legend()

plt.tight_layout()
plt.show()





# 1. add the average numbers of frames in each group/ the terms
#  get the variance
# TODO: 2.  calculate the variance of the head_angle in each group


# read the group discussion data, where the idx_timestamp is int

df = pd.read_csv(input_data, dtype={19: 'object', 20: 'object'})
df.info()

# only keep ['pitch_angle', 'roll_angle', 'yaw_angle'] and change from wide to long
df = df[info_List + ['pitch_angle', 'roll_angle', 'yaw_angle'] + ['interaction']+['idx_timestamp']]
df = df.melt(id_vars=info_List + ['interaction']+['idx_timestamp'], value_vars=['pitch_angle', 'roll_angle', 'yaw_angle'], var_name='head_p', value_name='head_angle')
# change the head_p to pitch, roll, yaw
df['head_p'] = df['head_p'].str.replace('_angle', '')

