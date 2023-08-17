import pandas as pd
import numpy as np
import datetime
import seaborn as sns
# import matplotlib.pyplot as plt
# import statsmodels.api as sm
# from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


frame_info = pd.read_csv('/Users/saiyingge/Resume_Study_DATA/group_discussion/data/compiled_data.csv')
frame_spk = pd.read_csv('/Users/saiyingge/Resume_Study_DATA/group_discussion/data/frame_speakers.csv')
game_info=pd.read_csv('/Users/saiyingge/**Research Projects/Resume/data/qualtrics_manual.csv')
discussion_info = pd.read_csv('/Users/saiyingge/Resume_Study_DATA/group_discussion/data/discussion_data_add_speakers.csv', dtype={19: 'object', 20: 'object'})
result_save_path = '/Users/saiyingge/Resume_Study_DATA/group_discussion/analysis_result/'


playerList = ['Alpha', 'Bravo', 'Charlie', 'Delta', 'Echo']
deceiversList = ['Alpha', 'Delta']
variable_List = ['smile'] + ['pitch_angle', 'roll_angle', 'yaw_angle'] + ['anger', 'disgust', 'fear',
       'happiness', 'neutral', 'sadness', 'surprise']
info_List = ['game','speaker','stage','imgindex','player']
# game info, get column include with 'rated_by'
rated_List= game_info.columns[game_info.columns.str.contains('rated_by')].tolist()





# descriptive statistics
# TODO: 1.  include some statistic about the interaction:
#  add the average numbers of frames in each group/ the terms
# read all frames info from complied file

#calculate the sec between start_y and end_y
frame_spk['seconds'] = (frame_spk['end_y'] - frame_spk['start_y'])/25
# group by for game and player, count the number of idx when the rank =1, and idx when the rank is others
frame_spk.groupby(['game','speaker'])['idx_timestamp'].count().reset_index().describe()
player_level = frame_spk.groupby(['game', 'speaker']).apply(
    lambda x: pd.Series({
        'count_Speaking': int(sum(x['rank'] == 1)),
        'count_Interacting': int(sum(x['rank'] != 1)),
        'sum_sec_Speaking': x.loc[x['rank'] == 1, 'seconds'].sum(),
        'sum_sec_Interacting': x.loc[x['rank'] != 1, 'seconds'].sum(),
        'sum_length_Speaking': x.loc[x['rank'] == 1, 'length_speech'].sum(),
        'sum_length_Interacting': x.loc[x['rank'] != 1, 'length_speech'].sum()
    })
).reset_index()

player_level['speaking_speed'] = player_level['sum_length_Speaking'] / player_level['sum_sec_Speaking']
#merge the player_level frame info with game info, on game name and player name from frame info, game and spaker name from game info
player_level = pd.merge(player_level, game_info.groupby(['group_id','codename'])[rated_List+ ['winner', 'role', 'sex']].first().reset_index(), left_on=['game','speaker'], right_on=['group_id','codename'], how='right')

# find player in player_level not in the game_info
# player_level[player_level['speaker'].isnull()]
#SG419A,Delta never talked

# create histogram for ['count_Speaking', 'count_Interacting', 'sum_sec_Speaking' , 'sum_length_Speaking', 'sum_length_Interacting', 'speaking_speed'], include the mean, std, min, max, color by role,in one plot

# Columns to plot
columns_to_plot = ['count_Speaking', 'count_Interacting', 'sum_sec_Speaking', 'sum_length_Speaking',
                   'sum_length_Interacting', 'speaking_speed']


# Function to display statistics on the histogram
def display_statistics(ax, data):
    stats = {
        'Mean': np.mean(data),
        'Std': np.std(data),
        'Min': np.min(data),
        'Max': np.max(data)
    }
    stats_text = "\n".join([f"{k}: {v:.2f}" for k, v in stats.items()])
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.5))


# Create a 2x3 grid of subplots
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

for idx, var in enumerate(columns_to_plot):
    row = idx // 3  # Determine the row index for the subplot
    col = idx % 3  # Determine the column index for the subplot
    ax = axs[row, col]

    # Plot histogram on the determined subplot
    ax.hist(player_level[var], bins=20, label=var, alpha=0.7, color='blue')

    # Display statistics on the subplot
    display_statistics(ax, player_level[var])
    ax.set_title(var)

# Adjust layout
plt.tight_layout()
# Save figure
plt.savefig(result_save_path + 'player_level_histogram_'+datetime.datetime.today().strftime('%Y-%m-%d')+'.png')
plt.show()

## boxplot  color by role,in one plot
# Create a 2x3 grid of subplots
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

for idx, var in enumerate(columns_to_plot):
    row = idx // 3  # Determine the row index for the subplot
    col = idx % 3  # Determine the column index for the subplot
    ax = axs[row, col]

    # Create boxplot on the determined subplot
    sns.boxplot(x='role', y=var, data=player_level, ax=ax, palette="Set3")
    ax.set_title(var)
    ax.set_ylabel('')  # To avoid repetitive y-labels
    ax.set_xlabel('Role')  # Set xlabel for each subplot
# Adjust layout
plt.tight_layout()
plt.show()


### game level

# group by game level count, average, std, min, max
frame_info.groupby(['game'])['idx_timestamp'].count().reset_index().describe()

#group by game and idx_timestamp, get the second， pick the first one ( should be the same for each group)
timestamp_info=frame_spk.groupby(['game','idx_timestamp'])['seconds'].first().reset_index()

#for each game, get the average length of speech and the count of speech
game_level = timestamp_info.groupby('game')['seconds'].agg(['mean', 'count', 'sum']).reset_index()
# calculate the Silence time for each game silence time = total time(the largest frame- smallest frame)/25 - active time
total_time_series = (frame_info.groupby('game')['end_y'].max() - frame_info.groupby('game')['start_y'].min()) / 25
game_level['total_time'] = total_time_series.reset_index(drop=True)
game_level['silence_time'] = game_level['total_time'] - game_level['sum']
game_level['silence_time_percent'] = game_level['silence_time']/game_level['total_time']

#merge the frame info with game info, on game name and player name from frame info, game and spaker name from game info
game_level = pd.merge(game_level, game_info.groupby(['group_id'])['winner'].first().reset_index(), left_on=['game'], right_on=['group_id'], how='left')

# assigned color to each game, winner is truth teller, color is yellow, winner is deceiver, color is blue
colors = np.where(game_level['winner']==1, 'maroon', 'darkgrey')

### turns-of-talk
game_level['count'].describe()

#plot the number of interaction in each game, sort by the number of interaction(count),  color the game by winner
plt.figure(figsize=(12, 6))
game_level = game_level.sort_values(by='count', ascending=False)
plt.bar(game_level['game'], game_level['count'], label='interaction counts', color=colors)
#add a line in the plot, show the average number of frames, include the ave number in the plot
plt.axhline(y=game_level['count'].mean(), color='r', linestyle='-', label='Average')
plt.text(20, game_level['count'].mean(), f"average:{game_level['count'].mean():.2f}", ha='right', va='bottom', fontsize=12)
plt.xlabel('Game')
plt.ylabel('Turns-of-talk')
plt.title('Number of Turns-of-talk in Each Game')
plt.xticks(rotation=45)
# show legend with color meaning
plt.legend(handles=[Patch(facecolor='darkgrey', label='Winner: Truth Teller'), Patch(facecolor='maroon', label='Winner: Deceiver')])
plt.tight_layout()

# save the plot,add date to the file name
# plt.savefig(result_save_path+'Game_interaction.png')
#include today's date in the file name
plt.savefig(result_save_path+'Game_interaction_'+datetime.datetime.today().strftime('%Y-%m-%d')+'.png')
plt.show()


###  length of speech
game_level['sum'].describe()
game_level['silence_time_percent'].describe()

# Sort the game_level by total_time
game_level = game_level.sort_values(by='sum', ascending=False)

plt.figure(figsize=(12, 6))
# Plot the active time with a specific pattern
plt.bar(game_level['game'], game_level['sum'], label='Active Time', color=colors)

# Plot the silence time on top of active time with another pattern
plt.bar(game_level['game'], game_level['silence_time'], bottom=game_level['sum'], label='Silence Time', color='white', edgecolor='lightgray')

# Annotate bars with the silence time percentage ABOVE the stacked bars
for idx, game in enumerate(game_level['game']):
    total_height = game_level['sum'].iloc[idx] + game_level['silence_time'].iloc[idx]
    plt.text(idx, total_height + 0.05 * total_height,  # Adding a small offset above the bar
             f"{int(game_level['silence_time_percent'].iloc[idx]*100)}%",
             ha='center', va='bottom', fontsize=9)

plt.xlabel('Game')
plt.ylabel('Total Time(seconds)')
plt.title('Active Time vs Silence Time in Each Game')
plt.xticks(rotation=45)
plt.legend(handles=[Patch(facecolor='darkgrey', label='Active time: Truth Teller'), Patch(facecolor='maroon', label='Active time: Deceiver'), Patch(facecolor='white', label='Silence Time')])
plt.tight_layout()

#include today's date in the file name
plt.savefig(result_save_path+'length_of_speech_'+datetime.datetime.today().strftime('%Y-%m-%d')+'.png')
plt.show()


###
# only keep ['pitch_angle', 'roll_angle', 'yaw_angle'] and change from wide to long
discussion_info = discussion_info[info_List + ['pitch_angle', 'roll_angle', 'yaw_angle'] + ['interaction']+['idx_timestamp']]
## check how many data points for each [game idx_timestamp]
discussion_info.groupby(['game','idx_timestamp'])['pitch_angle'].count().reset_index().describe()
#mean       19.287582   148.462418
# std        14.585118

# group by [game,player, idx_timestamp], for each group, get the average,variance and count of pitch, roll, yaw
temp = discussion_info.groupby(['game','player', 'idx_timestamp','interaction'])['pitch_angle', 'roll_angle', 'yaw_angle'].agg(['mean', 'var', 'count']).reset_index()

#TODO: check the pooling mean. variance, then run analysis

# discussion_info = discussion_info.melt(id_vars=info_List + ['interaction']+['idx_timestamp'], value_vars=['pitch_angle', 'roll_angle', 'yaw_angle'], var_name='head_p', value_name='head_angle')
# # change the head_p to pitch, roll, yaw
# discussion_info['head_p'] = discussion_info['head_p'].str.replace('_angle', '')


