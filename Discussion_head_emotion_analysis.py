'''
Run anova tests on the head and emotion data in discussion phase.
Analysis the Intensity (mean) and Variability (std) of the head and emotion data in discussion phase.

Author: Saiying(Tina) Ge
Date: August, 2023
input: emotion_data.csv discussion_data_add_speakers.csv, qualtrics_manual.csv
output: anova_result_f{head,emotion variables}.csv
'''
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import sys
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


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

def categorize_player(player):
    if player in deceiversList:
        return 'deceiver'
    else:
        return 'truth_teller'


def get_timestamp_agg(test_variable, discussion_info, game_info):
    # test_variable in discussion
    discussion_info = discussion_info[info_List + [test_variable] + ['interaction'] + ['idx_timestamp']]

    # silence
    discussion_info['idx_timestamp'] = discussion_info['idx_timestamp'].fillna(method='ffill')
    # add 0.5 to the idx_timestamp when the interaction is silence
    discussion_info.loc[discussion_info['interaction'] == 'Silence', 'idx_timestamp'] = discussion_info.loc[
                                                                                            discussion_info[
                                                                                                'interaction'] == 'Silence', 'idx_timestamp'] + 0.5
    # discussion_info.groupby(['interaction']).size()
    ## check how many data points for each [game idx_timestamp]
    discussion_info.groupby(['game', 'idx_timestamp'])[test_variable].count().reset_index().describe()

    timestamp_agg = discussion_info.groupby(['game', 'player', 'idx_timestamp', 'interaction', 'speaker'])[
        test_variable].agg(['mean', 'std', 'count']).reset_index()
    # timestamp_agg.groupby(['interaction']).size()
    # interaction sorting alternative, combine interaction and listening
    timestamp_agg['interaction_alter'] = timestamp_agg['interaction']
    timestamp_agg.loc[timestamp_agg['interaction'] == 'Interacting', 'interaction_alter'] = 'Listening'

    # add player info
    timestamp_agg = pd.merge(timestamp_agg,
                             game_info.groupby(['group_id', 'codename'])[['winner', 'sex']].first().reset_index(),
                             left_on=['game', 'player'], right_on=['group_id', 'codename'], how='left')

    # Flatten the MultiIndex for easier column access
    # also change role to player ro
    timestamp_agg = timestamp_agg.rename(
        columns={('speaker', ''): 'speaker', ('player', ''): 'player', ('interaction', ''): 'interaction',
                 ('idx_timestamp', ''): 'idx_timestamp', ('game', ''): 'game',
                 ('interaction_alter', ''): 'interaction_alter'})

    # change the winner game result to
    # timestamp_agg['winner'] = timestamp_agg['winner'].map({'0': 'Truth_teller_win', '1': 'Deceiver_win'})
    # add speaker role
    timestamp_agg['speaker_role'] = timestamp_agg['speaker'].map(categorize_player)
    # add player role
    timestamp_agg['player_role'] = timestamp_agg['player'].map(categorize_player)

    # Create a new column that indicates whether the speaker and player are in the same group
    timestamp_agg.loc[:, 'group'] = timestamp_agg.apply(lambda row: same_group(row['speaker'], row['player']), axis=1)

    # Calculate the median timestamp for each game
    medians = timestamp_agg.groupby('game')['idx_timestamp'].median()
    # Map the median timestamp back to the original dataframe
    timestamp_agg['median_timestamp'] = timestamp_agg['game'].map(medians)
    # Determine whether each timestamp is in the first half or second half
    timestamp_agg['half'] = ['1st-Half' if ts <= median else '2rd-Half' for ts, median in
                             zip(timestamp_agg['idx_timestamp'], timestamp_agg['median_timestamp'])]

    # show the interaction_alter group and count
    timestamp_agg.groupby(['interaction_alter']).size()

    # listener only
    timestamp_agg_listen = timestamp_agg[timestamp_agg['interaction_alter'] != 'Speaking']
    # speaker only
    timestamp_agg_speak = timestamp_agg[timestamp_agg['interaction_alter'] == 'Speaking']

    return timestamp_agg, timestamp_agg_listen, timestamp_agg_speak


def mean_pool_and_anova(df, group, independent):
    # pool the data and get average
    df = df.groupby(group)['mean'].mean().reset_index(name=f"pooled_mean")
    # anova test
    df = df[df['pooled_mean'] <= df['pooled_mean'].quantile(0.95)]
    model = ols('pooled_mean' + ' ~ ' + independent, data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    # print the ANOVA table
    print('pooled_mean' + ' ~ ' + independent)
    print(anova_table)


def pool_std(df_pool):
    # ignore the nan value in std
    counts = df_pool['count'].values
    stds = df_pool['std'].values

    # Filter out NaN values in stds
    valid_indices = ~np.isnan(stds)
    counts = counts[valid_indices]
    stds = stds[valid_indices]

    numerator = np.sum((counts - 1) * stds ** 2)
    denominator = np.sum(counts) - len(counts)

    return np.sqrt(numerator / denominator)


def std_pool_and_anova(df, group, independent):
    # pool the data and get pooled std
    df_pooled = df.groupby(group).apply(pool_std).reset_index(name=f"pooled_std")

    df_pooled = df_pooled[df_pooled['pooled_std'] <= df_pooled['pooled_std'].quantile(0.95)]
    model = ols('pooled_std' + ' ~ ' + independent, data=df_pooled).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    # print the ANOVA table
    print('pooled_std' + ' ~ ' + independent)
    print(anova_table)


def anova_run(df, dependent, independent):
    df = df[df[dependent] <= df[dependent].quantile(0.95)]
    model = ols(dependent + ' ~ ' + independent, data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    # print the ANOVA table
    print(dependent + ' ~ ' + independent)
    print(anova_table)



def run_all_independent_variables(timestamp_agg, timestamp_agg_listen, timestamp_agg_speak, test_variable):
    # save the printed result to csv
    # Backup the original stdout

    filename = f'anova_result_{test_variable}.csv'
    with open(filename, 'w') as f:

        sys.stdout = f

        print('test_variable: ' + test_variable)

        # interaction, role, half, winner, group
        # interaction / speaking or not
        print('interaction')
        mean_pool_and_anova(timestamp_agg, ['game', 'player', 'interaction_alter'], 'interaction_alter')
        std_pool_and_anova(timestamp_agg, ['game', 'player', 'interaction_alter'], 'interaction_alter')

        # player role
        print('player role')
        mean_pool_and_anova(timestamp_agg, ['game', 'player', 'player_role'], 'player_role')
        std_pool_and_anova(timestamp_agg, ['game', 'player', 'player_role'], 'player_role')
        # truth teller and deceiver's variable change while listening
        print('player role, listening only')
        mean_pool_and_anova(timestamp_agg_listen, ['game', 'player', 'player_role'], 'player_role')
        std_pool_and_anova(timestamp_agg_listen, ['game', 'player', 'player_role'], 'player_role')
        # truth teller and deceiver's variable change while speaking
        print('player role, speaking only')
        mean_pool_and_anova(timestamp_agg_speak, ['game', 'player', 'player_role'], 'player_role')
        std_pool_and_anova(timestamp_agg_speak, ['game', 'player', 'player_role'], 'player_role')

        # speaker role while listening
        print('speaker role, player\'s variable change while listening to truth teller and deceiver speaking')
        # player's variable change while listening to truth teller and deceiver speaking
        mean_pool_and_anova(timestamp_agg_listen, ['game', 'player', 'speaker_role'], 'speaker_role')
        std_pool_and_anova(timestamp_agg_listen, ['game', 'player', 'speaker_role'], 'speaker_role')

        # player's variable change while at different time period of discussion
        print('half')
        mean_pool_and_anova(timestamp_agg, ['game', 'player', 'half'], 'half')
        std_pool_and_anova(timestamp_agg, ['game', 'player', 'half'], 'half')
        # player's variable change while listening at different time period of discussion
        print('half, listening only')
        mean_pool_and_anova(timestamp_agg_listen, ['game', 'player', 'half'], 'half')
        std_pool_and_anova(timestamp_agg_listen, ['game', 'player', 'half'], 'half')
        # player's variable change while speaking at different time period of discussion
        print('half, speaking only')
        mean_pool_and_anova(timestamp_agg_speak, ['game', 'player', 'half'], 'half')
        std_pool_and_anova(timestamp_agg_speak, ['game', 'player', 'half'], 'half')

        # game result
        # variable difference between spy winning or truth teller wining  game
        print('winner')
        mean_pool_and_anova(timestamp_agg, ['game', 'player', 'winner'], 'winner')
        std_pool_and_anova(timestamp_agg, ['game', 'player', 'winner'], 'winner')
        # variable difference between spy winning or truth teller wining  game, listening only
        print('winner, while listening ')
        mean_pool_and_anova(timestamp_agg_listen, ['game', 'player', 'winner'], 'winner')
        std_pool_and_anova(timestamp_agg_listen, ['game', 'player', 'winner'], 'winner')
        # variable difference between spy winning or truth teller wining  game, speaking only
        print('winner, while speaking')
        mean_pool_and_anova(timestamp_agg_speak, ['game', 'player', 'winner'], 'winner')
        std_pool_and_anova(timestamp_agg_speak, ['game', 'player', 'winner'], 'winner')

        # group
        # variable difference between same group or different group
        print('group')
        mean_pool_and_anova(timestamp_agg, ['game', 'player', 'group'], 'group')
        std_pool_and_anova(timestamp_agg, ['game', 'player', 'group'], 'group')

        # speaker role * player role, listening only
        # truth teller and deceiver's variable change while listening to truth teller and deceiver speaking
        print('speaker role * player role, listening only')
        mean_pool_and_anova(timestamp_agg_listen, ['game', 'player', 'speaker_role', 'player_role'],
                            'speaker_role*player_role')
        std_pool_and_anova(timestamp_agg_listen, ['game', 'player', 'speaker_role', 'player_role'],
                           'speaker_role*player_role')

        # player role * half
        print('player role * half')
        mean_pool_and_anova(timestamp_agg, ['game', 'player', 'player_role', 'half'], 'player_role*half')
        std_pool_and_anova(timestamp_agg, ['game', 'player', 'player_role', 'half'], 'player_role*half')
        # player role * half, listening only
        print('player role * half, listening only')
        mean_pool_and_anova(timestamp_agg_listen, ['game', 'player', 'player_role', 'half'], 'player_role*half')
        std_pool_and_anova(timestamp_agg_listen, ['game', 'player', 'player_role', 'half'], 'player_role*half')
        # player role * half, speaking only
        print('player role * half, speaking only')
        mean_pool_and_anova(timestamp_agg_speak, ['game', 'player', 'player_role', 'half'], 'player_role*half')
        std_pool_and_anova(timestamp_agg_speak, ['game', 'player', 'player_role', 'half'], 'player_role*half')

        # player role * winner
        print('player role * winner')
        mean_pool_and_anova(timestamp_agg, ['game', 'player', 'player_role', 'winner'], 'player_role*winner')
        std_pool_and_anova(timestamp_agg, ['game', 'player', 'player_role', 'winner'], 'player_role*winner')
        # player role * winner, listening only
        print('player role * winner, listening only')
        mean_pool_and_anova(timestamp_agg_listen, ['game', 'player', 'player_role', 'winner'], 'player_role*winner')
        std_pool_and_anova(timestamp_agg_listen, ['game', 'player', 'player_role', 'winner'], 'player_role*winner')
        # player role * winner, speaking only
        print('player role * winner, speaking only')
        mean_pool_and_anova(timestamp_agg_speak, ['game', 'player', 'player_role', 'winner'], 'player_role*winner')
        std_pool_and_anova(timestamp_agg_speak, ['game', 'player', 'player_role', 'winner'], 'player_role*winner')

    # Remember to reset stdout back to the original after you're done
    sys.stdout = sys.__stdout__



# TODO: check silence time

emotion_info = pd.read_csv('/Users/saiyingge/**Research Projects/Resume/data/emotion_data.csv')
discussion_info = pd.read_csv('/Users/saiyingge/Resume_Study_DATA/group_discussion/data/discussion_data_add_speakers.csv', dtype={19: 'object', 20: 'object'})
game_info=pd.read_csv('/Users/saiyingge/**Research Projects/Resume/data/qualtrics_manual.csv')

variable_List = ['smile'] + ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise'] +['pitch_angle', 'roll_angle', 'yaw_angle']
info_List = ['game','speaker','stage','imgindex','player']
playerList = ['Alpha', 'Bravo', 'Charlie', 'Delta', 'Echo']
deceiversList = ['Alpha', 'Delta']


# emotion_info[test_variable].describe()

# test_variable_agg = emotion_info.groupby(['game', 'player', 'speaker', 'stage'])['smile'].agg(['mean', 'std', 'count']).reset_index()


## in discussion phase only
original_stdout = sys.stdout

variable_List = ['smile'] + ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise'] +['pitch_angle', 'roll_angle', 'yaw_angle']

test_variable = 'yaw_angle'
timestamp_agg,timestamp_agg_listen,timestamp_agg_speak = get_timestamp_agg(test_variable,discussion_info,game_info)
run_all_independent_variables(timestamp_agg,timestamp_agg_listen,timestamp_agg_speak,test_variable)

# for x in variable_List:
#     print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
#     print(x)
#     timestamp_agg,timestamp_agg_listen,timestamp_agg_speak = get_timestamp_agg(x,discussion_info,game_info)
#     run_all_independent_variables(timestamp_agg,timestamp_agg_listen,timestamp_agg_speak,x)


# print(pairwise_tukeyhsd(timestamp_agg['mean'], timestamp_agg['half']))
# #Specifically, on average, the '2rd-Half ' group scores  6.6697  units Higher than the '1st-Half' group on the 'smile mean' variable
# sns.boxplot(x='half', y='mean',data=timestamp_agg, palette='Set3')
# plt.show()
