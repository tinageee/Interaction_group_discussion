'''
This Python script processes group discussion video files based on timestamp data. It performs the following tasks:

Checks if the timestamp file has the correct format, ensuring proper alignment and unique combinations of timestamps.

Finds the gallery view video for a specific game.

Cuts video clips based on timestamp data for both original and processed videos.

Extracts image frame indices and saves them in a CSV file.

Author: Saiying(Tina) Ge
Date: July, 2023

frame getting process:
1)Extract Timestamps: Extract timestamps from the transcript (.vtt) file. with get_timestamps.py
2)Match Timestamps: Match the extracted timestamps with the proofreading file.
3)Manual Timestamp Alignment: Manually adjust timestamps if required.
4)Video Cutting: Cut the video based on the adjusted timestamps for further verification.
5)Save Frame Indices: Save the frame indices to a CSV file.
6)Frame Verification: Cross-check the frames with the corresponding image files to ensure accuracy.

'''

from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import pandas as pd
import os
from datetime import datetime
import warnings
import numpy as np

warnings.simplefilter(action='ignore', category=FutureWarning)


# check the file to make sure it has the right format
def Pass_file_checking(game):
    """
    make sure the file has the right format
    :param df:
    :return: boolean
    """
    res = True
    # if the game name matches the file name
    df = pd.read_csv(file_path + game + '_checked.csv')
    if df['game'][0] != game:
        print('game name does not match')
        res = False
    # if the idx_raw contains all numbers from the smallest to the largest
    if df['idx_raw'].nunique() != df['idx_raw'].max() - df['idx_raw'].min() + 1:
        print('idx_raw does not contain all numbers in the timestamp file, missing '+str(df['idx_raw'].max() - df['idx_raw'].min() + 1-df['idx_raw'].nunique())+' rows')
        res = False
    # check the index is in order
    if not df['idx_raw'].is_monotonic_increasing:
        print('idx_raw is not in order')
        res = False
    # the numbers of unique combination of ['idx_raw','start','end'] should be the same as the unique idx_raw
    if df.groupby(['idx_raw', 'start', 'end']).ngroups != df['idx_raw'].nunique():
        print('idx_raw, start, end combination is not unique')
        res = False

    return res


# find GalleryView in the video folder for the game
def find_gallery_view_video(video_path, game):
    """
    find the gallery view for the game
    :param game:
    :return: the path to the gallery view
    """
    # match the game name and "GalleryView"
    for file in os.listdir(video_path):
        # 'gallery' could be 'Gallery' or 'gallery'
        if game in file and 'gallery' in file.lower():
            # print(file)
            return file


# find precessed video by game name and player name
def find_precessed_video(video_path, game, player):
    """
    find the gallery view for the game
    :param game:
    :return: the path to the gallery view
    """
    # match the game name and "GalleryView"
    for file in os.listdir(video_path):
        # match the game name and player name match the letter after '_'
        # and mp4 file only
        if file.endswith('.mp4') and game in file and player in file.split('_')[1]:
            print(file)
            return file


# change the time format to seconds
def time_to_seconds(timestr):
    min_sec = timestr.split(":")
    minutes = int(min_sec[0])
    seconds = float(min_sec[1])
    result = minutes * 60 + seconds

    # due to data transformation issue, the time does not include hour
    # the whole gae is smaller than 1.5 hr, and discussion starts after 30 minutes
    # add1 hr to the time if the time is smaller than 30 minutes
    if result < 1800:
        result += 3600

    return result


# cut the video
def cut_video_original(file_path, video_path, save_path, game):
    # timestamp file
    df = pd.read_csv(file_path + game + '_checked.csv')
    # video file
    input_file = video_path + find_gallery_view_video(video_path, game)

    # get the unique 'idx_raw','start','end' combinations
    unique_combinations = df[['idx_raw', 'start', 'end']].drop_duplicates()
    # Convert 'start' and 'end' columns to time in seconds
    unique_combinations['start'] = unique_combinations['start'].apply(time_to_seconds)
    unique_combinations['end'] = unique_combinations['end'].apply(time_to_seconds)

    # cut for unique 'idx_raw','start','end' combination
    for i, row in unique_combinations.iterrows():
        idx_raw, start, end = row  # Unpack the group tuple
        # save the file with the idx_raw
        output_file = f'{save_path}{game}_{int(idx_raw)}.mp4'
        # add half second to the start time and end time
        ffmpeg_extract_subclip(input_file, start - 0.5, end + 0.5, targetname=output_file)
        print(idx_raw, start, end)


# cut the processed video
def cut_video_processed(file_path, video_path, game, save_path, change_time, player):
    # timestamp file
    df = pd.read_csv(file_path + game + '_checked.csv')
    # video file
    input_file = video_path + find_precessed_video(video_path, game, player)

    # get the unique 'idx_raw','start','end' combinations
    unique_combinations = df[['idx_raw', 'start', 'end']].drop_duplicates()
    # Convert 'start' and 'end' columns to time in seconds
    unique_combinations['start'] = unique_combinations['start'].apply(time_to_seconds)
    unique_combinations['end'] = unique_combinations['end'].apply(time_to_seconds)

    # Set the original time earliest start time to 0
    # Calculate the earliest start time and adjust 'start' and 'end' accordingly
    earliest_start = unique_combinations['start'].min()
    unique_combinations['start'] -= earliest_start
    unique_combinations['end'] -= earliest_start

    # cut for unique 'idx_raw','start','end' combination
    for i, row in unique_combinations.iterrows():
        idx_raw, start, end = row  # Unpack the group tuple
        # save the file with the idx_raw
        output_file = f'{save_path}{game}_{player}_{int(idx_raw)}.mp4'
        # add half second to the start time and end time
        ffmpeg_extract_subclip(input_file, start + change_time - 0.5, end + change_time + 0.5, targetname=output_file)
        print(idx_raw, start, end)


# get the image frame index based on the time and save the index to csv file
def get_img_frame_indx(game, delta_time):
    '''
    get the image frame index based on the time, and save the index to csv file
    :param game:
    :param delta_time:
    :return:
    '''
    # timestamp file
    df = pd.read_csv(file_path + game + '_checked.csv')

    # get the unique 'idx_raw','start','end' combinations
    unique_combinations = df[['idx_raw', 'start', 'end']].drop_duplicates().reset_index(drop=True)

    # Convert 'start' and 'end' columns to time in seconds
    unique_combinations['start'] = unique_combinations['start'].apply(time_to_seconds)
    unique_combinations['end'] = unique_combinations['end'].apply(time_to_seconds)

    # Set the original time earliest start time to 0
    # Calculate the earliest start time and adjust 'start' and 'end' accordingly
    earliest_start = unique_combinations['start'].min()
    # get the frame, video is 25 fps, take floor to get integer
    unique_combinations['start'] = np.floor((unique_combinations['start'] - earliest_start + delta_time) * 25)
    unique_combinations['end'] = np.floor((unique_combinations['end'] - earliest_start + delta_time) * 25)

    # check if there is any overlap between intervals, if there is, print the warning
    for i, row in unique_combinations.iterrows():
        idx_raw, start, end = row  # Unpack the group tuple
        if i > 1 and start < unique_combinations.iloc[i - 1]['end']:
            print('warning: overlap between intervals', 'index', i)
            return

    # merge the start and end column to df by index
    # only keep 'idx_o', 'game', 'speaker',  'text',  'idx_raw','idx', 'start_y', 'end_y'
    df = pd.merge(df, unique_combinations, on=['idx_raw'], how='left')
    df = df[['idx_o', 'game', 'speaker', 'text', 'idx_raw', 'idx', 'start_y', 'end_y']]
    # rename index
    df.rename(columns={'idx_raw': 'idx_timestamp', 'idx': 'idx_transcripts', 'idx': 'idx_transcripts'}, inplace=True)
    df.to_csv(file_path + game + '_Frame_found.csv', index=False)


# To read the csv file:
file_path = '/Users/saiyingge/Resume_Study_DATA/group_discussion/timestamp/'
video_path_full = '/Users/saiyingge/Resume_Study_DATA/Unzips-all/mp4/'
video_path_processed = '/Users/saiyingge/Resume_Study_DATA/group_discussion/GD_cut/'

save_path_full = '/Users/saiyingge/Resume_Study_DATA/group_discussion/'
save_path_processed = '/Users/saiyingge/Resume_Study_DATA/group_discussion/cutted_video/'

# read the csv has game as the prefix

# game = 'SG4121'
# game = 'SG422A'
# game = 'XC419A'
# game = 'XC411A'
game = 'XC415B'

Pass_file_checking(game)
# cut_video_original(file_path, video_path_full, save_path_full, game)
cut_video_processed(file_path, video_path_processed, game, save_path_processed, change_time=-14, player='B')

get_img_frame_indx(game, delta_time=-14)



# -----
# organize all data
# compile csv file from the save_path

frame_complied=[]
for file in os.listdir(file_path):
    if 'Frame_found' in file:
        # Read the file and combine
        print(file)
        file_full_path = os.path.join(file_path, file)
        data = pd.read_csv(file_full_path, dtype={'start_y': int, 'end_y': int,'idx_o': int})  # Assuming the data is in CSV format
        frame_complied.append(data)

compiled_data = pd.concat(frame_complied, ignore_index=True)
#sort by idx_o
compiled_data=compiled_data.sort_values(by='idx_o')

# Check if the values in the 'ind_o' column are consecutive or have missing values
is_consecutive=(compiled_data['idx_o'].diff() == 1).all()
is_consecutive
compiled_data[~(compiled_data['idx_o'].diff() == 1)]

'''
     idx_o    game  speaker  ... idx_transcripts  start_y  end_y
358      0  SG4111    Bravo  ...               1       12     72
735    361  SG422A  Charlie  ...               8     4232   4240
'''

#should be 22 games
compiled_data['game'].nunique()

#save the complied data
compiled_data.to_csv(save_path_full+'compiled_data.csv',index=False)