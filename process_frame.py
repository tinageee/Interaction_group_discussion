'''
This Python script read all frames info from face++ results in discussion, assign the speaker value to each frame, based on the frame_info.csv, and save discussion_data to csv.

processes group discussion video files based on timestamp data. It performs the following tasks:
1. read all frames info from complied file
2. group by game and idx_timestamp, for each group, rank speakers by their length of speech then the idx_transcripts
3. create a column count the length of speech
4. create a new df,frame_speaker, group by game ,speaker,and  idx_timestamp,for each group, combine the 'text' if same 'speaker', keep game ,speaker,and  idx_timestamp
5. long to wide
6. change the column names and save frame_speakers_wide to csv
7. read the group discussion data
8. assign the speaker value to each frame, based on the frame_info.csv
9. cross check if the speaker value is assigned correctly
10. save discussion_data to csv

Author: Saiying(Tina) Ge
Date: July, 2023
input: complied_data.csv, group discussion data
output: discussion_data_with_speaker.csv
'''
import pandas as pd
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# read all frames info from complied file
frame_info = pd.read_csv('/Users/saiyingge/Resume_Study_DATA/group_discussion/compiled_data.csv')

# group by game and idx_timestamp, for each group, rank speakers by their length of speech then the idx_transcripts
# create a column count the length of speech
# create a new df,frame_speaker, group by game ,speaker,and  idx_timestamp,for each group, combine the 'text' if same 'speaker', keep game ,speaker,and  idx_timestamp
frame_speakers = frame_info.groupby(['game', 'idx_timestamp','speaker','start_y','end_y'])['text'].apply(' '.join).reset_index()
# for each  game and  idx_timestamp,rank speaker by length of speech.
frame_speakers['length_speech'] = frame_speakers.groupby(['game', 'speaker', 'idx_timestamp','start_y','end_y'])['text'].transform(lambda x: len(' '.join(x)))
frame_speakers['rank'] = frame_speakers.groupby(['game', 'idx_timestamp','start_y','end_y'])['length_speech'].rank(method='first', ascending=False)

# long to wide
frame_speakers_wide = frame_speakers.pivot_table(index=['game', 'idx_timestamp','start_y','end_y'], columns='rank', values='speaker', aggfunc='first').reset_index()

frame_info=frame_speakers_wide
#  unique combination of game and idx_timestamp should same as number of unique combination should be the same as 'game', 'idx_timestamp','start_y','end_y'
frame_info.groupby(['game', 'idx_timestamp']).ngroups == frame_info.groupby(['game', 'idx_timestamp','start_y','end_y']).ngroups

# change the column names and save frame_speakers_wide to csv
frame_info.columns = ['game', 'idx_timestamp','start_frame', 'end_frame', 'speaker_1', 'speaker_2', 'speaker_3', 'speaker_4', 'speaker_5']
frame_info.to_csv('/Users/saiyingge/Resume_Study_DATA/group_discussion/frame_info.csv', index=False)

### read the group discussion data
file_fold ='/Users/saiyingge/**Research Projects/Resume/data/faceplus'

discussion_data = pd.DataFrame()

# read the group discussion data
for file in os.listdir(file_fold):
    # if no 'Note' or 'intro' in file:
    if 'Note' not in file and 'Intro' not in file:
        # print(file)
        try:
            discussion_data = discussion_data.append(pd.read_csv(file_fold + '/' + file))
        except:
            print(file)

# assign the speaker value to each frame, based on the frame_info.csv
# check if the imgindex not end with digit
discussion_data[~discussion_data['imgindex'].str.contains(r'\d$')]
# remove the imgindex not end with digit
discussion_data = discussion_data[discussion_data['imgindex'].str.contains(r'\d$')]
# create a column frame, extract the frame number from imgindex
discussion_data['frame'] = discussion_data['imgindex'].apply(lambda x: int(x.split('_')[1]))

# add 1 frame to start, 1 to frame to end, to make sure the frame falls in the range of start_frame and end_frame
frame_info['start_frame'] = frame_info['start_frame'] - 8
frame_info['end_frame'] = frame_info['end_frame'] + 8

for game in frame_info['game'].unique():
# for each game if the frame falls in the range of start_frame and end_frame, assign the speaker value to the frame
    for idx in frame_info[frame_info['game'] == game]['idx_timestamp'].unique():
            start_frame = frame_info[(frame_info['game'] == game) & (frame_info['idx_timestamp'] == idx)]['start_frame'].values[0]
            end_frame = frame_info[(frame_info['game'] == game) & (frame_info['idx_timestamp'] == idx)]['end_frame'].values[0]
            # if the frame falls in the range of start_frame and end_frame, assign the column  ['speaker_1','speaker_2','speaker_3','speaker_4','speaker_5']  to discussion_data
            discussion_data.loc[(discussion_data['game'] == game) & (discussion_data['frame'] >= start_frame) & (discussion_data['frame'] <= end_frame), ['speaker_1','speaker_2','speaker_3','speaker_4','speaker_5']] = frame_info[(frame_info['game'] == game) & (frame_info['idx_timestamp'] == idx)][['speaker_1','speaker_2','speaker_3','speaker_4','speaker_5']].values[0]

discussion_data.replace({'A': 'Alpha', 'B': 'Bravo', 'C': 'Charlie', 'D': 'Delta', 'E': 'Echo'}, inplace=True)
discussion_data.rename(columns={'Subject': 'player'}, inplace=True)
discussion_data['stage'] = 'Discussion'

# cross check if the speaker value is assigned correctly
# print out sg4111 and idx_timestamp 2 in frame_info
# show whole row
pd.set_option('display.max_columns', None)
frame_info[(frame_info['game'] == 'SG4111') & (frame_info['idx_timestamp'] == 2)]

# find SG4111, frame 12-72, speaker_1 should be 'Charlie', speaker_2 should be 'Bravo'
discussion_data[(discussion_data['game'] == 'SG4111') & (discussion_data['frame'] >= 12) & (discussion_data['frame'] <= 72)][['game','frame','speaker_1','speaker_2']]

# save discussion_data to csv
discussion_data.to_csv('/Users/saiyingge/Resume_Study_DATA/group_discussion/discussion_data_add_speakers.csv', index=False)