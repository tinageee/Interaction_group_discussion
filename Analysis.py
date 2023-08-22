import pandas as pd
import numpy as np
import datetime
import seaborn as sns
# import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
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


# # Create a 2x3 grid of subplots
# fig, axs = plt.subplots(2, 3, figsize=(15, 10))
#
# for idx, var in enumerate(columns_to_plot):
#     row = idx // 3  # Determine the row index for the subplot
#     col = idx % 3  # Determine the column index for the subplot
#     ax = axs[row, col]
#
#     # Plot histogram on the determined subplot
#     ax.hist(player_level[var], bins=20, label=var, alpha=0.7, color='blue')
#
#     # Display statistics on the subplot
#     display_statistics(ax, player_level[var])
#     ax.set_title(var)
#
# # Adjust layout
# plt.tight_layout()
# # Save figure
# plt.savefig(result_save_path + 'player_level_histogram_'+datetime.datetime.today().strftime('%Y-%m-%d')+'.png')
# plt.show()
#
# ## boxplot  color by role,in one plot
# # Create a 2x3 grid of subplots
# fig, axs = plt.subplots(2, 3, figsize=(15, 10))
#
# for idx, var in enumerate(columns_to_plot):
#     row = idx // 3  # Determine the row index for the subplot
#     col = idx % 3  # Determine the column index for the subplot
#     ax = axs[row, col]
#
#     # Create boxplot on the determined subplot
#     sns.boxplot(x='role', y=var, data=player_level, ax=ax, palette="Set3")
#     ax.set_title(var)
#     ax.set_ylabel('')  # To avoid repetitive y-labels
#     ax.set_xlabel('Role')  # Set xlabel for each subplot
# # Adjust layout
# plt.tight_layout()
# plt.show()


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
# colors = np.where(game_level['winner']==1, 'maroon', 'darkgrey')
#
# ### turns-of-talk
# game_level['count'].describe()
#
# #plot the number of interaction in each game, sort by the number of interaction(count),  color the game by winner
# plt.figure(figsize=(12, 6))
# game_level = game_level.sort_values(by='count', ascending=False)
# plt.bar(game_level['game'], game_level['count'], label='interaction counts', color=colors)
# #add a line in the plot, show the average number of frames, include the ave number in the plot
# plt.axhline(y=game_level['count'].mean(), color='r', linestyle='-', label='Average')
# plt.text(20, game_level['count'].mean(), f"average:{game_level['count'].mean():.2f}", ha='right', va='bottom', fontsize=12)
# plt.xlabel('Game')
# plt.ylabel('Turns-of-talk')
# plt.title('Number of Turns-of-talk in Each Game')
# plt.xticks(rotation=45)
# # show legend with color meaning
# plt.legend(handles=[Patch(facecolor='darkgrey', label='Winner: Truth Teller'), Patch(facecolor='maroon', label='Winner: Deceiver')])
# plt.tight_layout()
#
# # save the plot,add date to the file name
# # plt.savefig(result_save_path+'Game_interaction.png')
# #include today's date in the file name
# plt.savefig(result_save_path+'Game_interaction_'+datetime.datetime.today().strftime('%Y-%m-%d')+'.png')
# plt.show()
#
#
# ###  length of speech
# game_level['sum'].describe()
# game_level['silence_time_percent'].describe()
#
# # Sort the game_level by total_time
# game_level = game_level.sort_values(by='sum', ascending=False)
#
# plt.figure(figsize=(12, 6))
# # Plot the active time with a specific pattern
# plt.bar(game_level['game'], game_level['sum'], label='Active Time', color=colors)
#
# # Plot the silence time on top of active time with another pattern
# plt.bar(game_level['game'], game_level['silence_time'], bottom=game_level['sum'], label='Silence Time', color='white', edgecolor='lightgray')
#
# # Annotate bars with the silence time percentage ABOVE the stacked bars
# for idx, game in enumerate(game_level['game']):
#     total_height = game_level['sum'].iloc[idx] + game_level['silence_time'].iloc[idx]
#     plt.text(idx, total_height + 0.05 * total_height,  # Adding a small offset above the bar
#              f"{int(game_level['silence_time_percent'].iloc[idx]*100)}%",
#              ha='center', va='bottom', fontsize=9)
#
# plt.xlabel('Game')
# plt.ylabel('Total Time(seconds)')
# plt.title('Active Time vs Silence Time in Each Game')
# plt.xticks(rotation=45)
# plt.legend(handles=[Patch(facecolor='darkgrey', label='Active time: Truth Teller'), Patch(facecolor='maroon', label='Active time: Deceiver'), Patch(facecolor='white', label='Silence Time')])
# plt.tight_layout()
#
# #include today's date in the file name
# plt.savefig(result_save_path+'length_of_speech_'+datetime.datetime.today().strftime('%Y-%m-%d')+'.png')
# plt.show()

def same_group(speaker, player):
    # if speaker is the player
    if speaker == player:
        return 'self'
    # if speaker and player are in the same group: both deceivers or both truth tellers
    elif (speaker in deceiversList and player in deceiversList) \
            or (speaker not in deceiversList and player not in deceiversList):
        return 'same'
    else:
        return 'diff'

def pooled_std(df, headPose):
    # ignore the nan value in std

    counts = df[(headPose, 'count')].values
    stds = df[(headPose, 'std')].values

    # Filter out NaN values in stds
    valid_indices = ~np.isnan(stds)
    counts = counts[valid_indices]
    stds = stds[valid_indices]

    numerator = np.sum((counts - 1) * stds ** 2)
    denominator = np.sum(counts) - len(counts)

    return np.sqrt(numerator / denominator)

def pool_group_by(df,grouByColumns):

    pooled_std_results = pd.DataFrame()

    for x in ['pitch_angle', 'roll_angle', 'yaw_angle']:
        result = df.groupby(grouByColumns).apply(pooled_std, x).reset_index(name=f"{x}_pooled_std")
        if pooled_std_results.empty:
            pooled_std_results = result
        else:
            pooled_std_results = pd.merge(pooled_std_results, result,  how='left')

    return pooled_std_results
###
# only keep ['pitch_angle', 'roll_angle', 'yaw_angle'] and change from wide to long
discussion_info = discussion_info[info_List + ['pitch_angle', 'roll_angle', 'yaw_angle'] + ['interaction']+['idx_timestamp']]
# add value for idx_timestamp, when the interaction is silence, the idx_timestamp is the same as (the previous idx+0.5)
discussion_info['idx_timestamp'] = discussion_info['idx_timestamp'].fillna(method='ffill')
# add 0.5 to the idx_timestamp when the interaction is silence
discussion_info.loc[discussion_info['interaction']=='Silence', 'idx_timestamp'] = discussion_info.loc[discussion_info['interaction']=='Silence', 'idx_timestamp'] + 0.5
## check how many data points for each [game idx_timestamp]
discussion_info.groupby(['game','idx_timestamp'])['pitch_angle'].count().reset_index().describe()
# mean       19.172311   104.231076
# std        14.536035   152.286778

# count the average number of frame for each interaction state in per person /109
discussion_info.groupby(['interaction'])['imgindex'].apply(lambda x: len(x)/109).reset_index()




## grou by [game,player, idx_timestamp]
# group by [game,player, idx_timestamp], for each group, get the average,sd and count of pitch, roll, yaw
timestamp_agg = discussion_info.groupby(['game','player', 'idx_timestamp','interaction','speaker'])['pitch_angle', 'roll_angle', 'yaw_angle'].agg(['mean', 'std', 'count']).reset_index()
# interaction sorting alternative, combine interaction and listening
timestamp_agg['interaction_alter'] = timestamp_agg['interaction']
timestamp_agg.loc[timestamp_agg['interaction'] == 'Interacting', 'interaction_alter'] = 'Listening'
# add player info
timestamp_agg = pd.merge(timestamp_agg, game_info.groupby(['group_id','codename'])[ ['winner', 'role', 'sex']].first().reset_index(), left_on=['game','player'], right_on=['group_id','codename'], how='left')
# Create a new column that indicates whether the speaker and player are in the same group
timestamp_agg.loc[:, 'group'] = timestamp_agg.apply( lambda row: same_group(row['speaker', ''], row['player', '']), axis=1)

# Flatten the MultiIndex for easier column access
timestamp_agg = timestamp_agg.rename(columns={('speaker', ''): 'speaker', ('player', ''): 'player', ('interaction', ''): 'interaction',('idx_timestamp', ''):'idx_timestamp',('game', ''):'game',('interaction_alter', ''):'interaction_alter'})

# create a new column that indicates whether the idx_timestamp is in the first half or second half in the game
# Calculate the median timestamp for each game
medians = timestamp_agg.groupby('game')['idx_timestamp'].median()

# Map the median timestamp back to the original dataframe
timestamp_agg['median_timestamp'] = timestamp_agg['game'].map(medians)

# Determine whether each timestamp is in the first half or second half
timestamp_agg['half'] = ['1st-Half' if ts <= median else '2rd-Half' for ts, median in zip(timestamp_agg['idx_timestamp'], timestamp_agg['median_timestamp'])]

#TODO: check the pooling mean. variance, then run analysis

# player,  interaction state, interacting with whom [same team or different team], first half or second half
# by player, by player + diff interaction state, by player + diff interaction + first half, second half

# by player, count the pooled std of pitch, roll, yaw at different interaction state


def pool_and_one_way_anova(df, grouByColumns, factor1):
    # Pool the standard deviation of the head pose for each player
    pooled_std_results = pool_group_by(df,grouByColumns)

    # Flatten the MultiIndex for easier column access
    # for i in range(3):
    #     pooled_std_results = pooled_std_results.rename(columns={grouByColumns[i]: grouByColumns[i][0]})

    results = []
    # Iterate over each head pose pooled standard deviation
    for headPose_std in ['pitch_angle_pooled_std', 'roll_angle_pooled_std', 'yaw_angle_pooled_std']:
        # Create the formula for ANOVA
        formula = f"{headPose_std} ~ {factor1}"

        # Perform ANOVA
        model = ols(formula, data=pooled_std_results).fit()
        aov_table = sm.stats.anova_lm(model, typ=2)  # Type 2 ANOVA DataFrame

        sns.boxplot(x=factor1, y=headPose_std, data=pooled_std_results)
        plt.show()
        # Append results
        results.append({
            'headPose_std': headPose_std,
            'f_value': aov_table['F'][0],
            'p_value': aov_table['PR(>F)'][0]
        })

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)
    # print('group by' + grouByColumns)
    print('independent_variable: ' + factor1)
    print(results_df)


def pool_and_two_way_anova(df, grouByColumns, factor1, factor2):

    # Pool the standard deviation of the head pose for each player
    pooled_std_results = pool_group_by(df,grouByColumns)

    # # Flatten the MultiIndex for easier column access
    results = []
    # Iterate over each head pose pooled standard deviation
    for headPose_std in ['pitch_angle_pooled_std', 'roll_angle_pooled_std', 'yaw_angle_pooled_std']:
        # Create the formula for two-way ANOVA
        formula = f"{headPose_std} ~ C({factor1}) + C({factor2}) + C({factor1}):C({factor2})"

        # Perform ANOVA
        model = ols(formula, data=pooled_std_results).fit()
        aov_table = sm.stats.anova_lm(model, typ=2)  # Type 2 ANOVA DataFrame

        sns.boxplot(x=factor1, y=headPose_std, hue=factor2, data=pooled_std_results)
        plt.show()

        # Append results
        results.append({
            'headPose_std': headPose_std,
            'f_value_interaction': aov_table.loc[f"C({factor1}):C({factor2})", 'F'],
            'p_value_interaction': aov_table.loc[f"C({factor1}):C({factor2})", 'PR(>F)'],
            'f_value_factor1': aov_table.loc[f"C({factor1})", 'F'],
            'p_value_factor1': aov_table.loc[f"C({factor1})", 'PR(>F)'],
            'f_value_factor2': aov_table.loc[f"C({factor2})", 'F'],
            'p_value_factor2': aov_table.loc[f"C({factor2})", 'PR(>F)']
        })

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)
    print('independent variables: ' + factor1 + ', ' + factor2)
    print(results_df)


grouByColumns=['game','player', 'interaction']
pool_and_one_way_anova(timestamp_agg, grouByColumns, 'interaction')
grouByColumns=['game','player', 'interaction_alter']
pool_and_one_way_anova(timestamp_agg, grouByColumns, 'interaction_alter')

grouByColumns=['game','player', 'interaction', 'role']
pool_and_one_way_anova(timestamp_agg, grouByColumns, 'role')
grouByColumns=['game','player', 'interaction_alter', 'role']
pool_and_one_way_anova(timestamp_agg, grouByColumns, 'role')

grouByColumns=['game','player', 'interaction', 'half']
pool_and_one_way_anova(timestamp_agg, grouByColumns, 'half')
grouByColumns=['game','player', 'interaction_alter', 'half']
pool_and_one_way_anova(timestamp_agg, grouByColumns, 'half')


# two-way anova
# show all row,column
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# 'interaction * group'
grouByColumns=['game','player', 'interaction','group']
pool_and_two_way_anova(timestamp_agg[timestamp_agg['interaction'] != 'Speaking'], grouByColumns, 'group', 'interaction')
grouByColumns=['game','player', 'interaction_alter', 'group']
pool_and_one_way_anova(timestamp_agg[timestamp_agg['interaction_alter'] != 'Speaking'], grouByColumns, 'group')
##interaction * group significant on roll and yaw


# 'interaction * role'
grouByColumns=['game','player', 'interaction','role']
pool_and_two_way_anova(timestamp_agg, grouByColumns, 'role', 'interaction')
grouByColumns=['game','player', 'interaction_alter', 'role']
pool_and_two_way_anova(timestamp_agg, grouByColumns, 'role', 'interaction_alter')

# 'interaction * half'
grouByColumns=['game','player', 'interaction','half']
pool_and_two_way_anova(timestamp_agg, grouByColumns, 'half', 'interaction')
grouByColumns=['game','player', 'interaction_alter', 'half']
pool_and_two_way_anova(timestamp_agg, grouByColumns, 'half', 'interaction_alter')

# TODO emotion analysis

