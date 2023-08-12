'''
Description:
This Python script extracts time stamps for each speaker during the discussion stage from a transcript in .vtt format.
It uses fuzzy string matching to match the speaker names and their corresponding text in the proofread and raw transcripts.
The time stamps are then saved into separate CSV files for each game.

Author: Saiying(Tina) Ge
Date: July, 2023
Description: This Python script processes transcript (.vtt) files to extract time stamps for each speaker during the discussion stage.
'''

from fuzzywuzzy import fuzz
import pandas as pd

# remove the panda FutureWarning
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import re

folder_path = '/Users/saiyingge/Resume_Study_DATA/Unzips-all/vtt'
save_path = '/Users/saiyingge/Resume_Study_DATA/group_discussion/timestamp/from_get_time_stamp/'


def get_time_stamp(game, folder_path):
    '''
    use the transcript file to get the time stamp for each speaker at the discussion stage
    1. find the start and end of the group discussion, from the transcript_proofread file
    2. read the vtt file
    3. go over vtt file and find the transcripts between the start and end
    4. find the time stamp for each speaker
    :param folder_path: the folder path for the vtt file
    :param game: game name
    :return: dataframe with time stamp and transcript
    '''
    game_transcript = transcript_proofread[transcript_proofread['game'] == game]
    # find the start and end of the group discussion
    start_lag = 0
    end_lead = -1
    target_start_text = game_transcript['text'].iloc[start_lag]
    target_end_text = game_transcript['text'].iloc[end_lead]
    # if the target text is too short, use the next one as the target
    while len(target_start_text) < 15:
        start_lag += 1
        target_start_text = game_transcript['text'].iloc[start_lag]
    while len(target_end_text) < 15:
        end_lead -= 1
        target_end_text = game_transcript['text'].iloc[end_lead]

    start_speaker = game_transcript['speaker'].iloc[start_lag]
    end_speaker = game_transcript['speaker'].iloc[end_lead]

    threshold = 65  # the threshold for the match score
    transcripts = []

    # read the vtt file
    file_path = folder_path + '/' + game + '_Transcript.vtt'
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # go over vtt file and find the transcripts between the start and end
    i = int(len(lines) / 2)  # the discussion is at end of video, start from the middle
    end_of_line = len(lines)  # initial the end of line as the end of the file
    last_speaker = ''
    loop_status = 0  # initial the loop status: 0 = initial, 1= start discussion, 2 = end discussion

    while i < end_of_line:
        line = lines[i].strip()

        if line.isnumeric() or line == '' or '-->' in line:  # Skip cue number, empty lines, timestamps
            i += 1
            continue

        # for text lines
        if len(line.split(':')) == 2:
            speaker, text = line.strip().split(':', 1)
            # print(line,i)
            # check if the speaker is the same as the last speaker and the match score is high
            if (speaker == start_speaker) and (
                    fuzz.partial_ratio(target_start_text.lower(), text.lower()) > threshold):
                loop_status = 1
                # print('find the start of group discussion', i)
                i -= 1 + (start_lag * 4)  # Go back one line to find timestamp

        # find the start of group discussion, go to the next line to find the timestamp
        while loop_status > 0 and i < end_of_line:

            line = lines[i].strip()
            # print(line, i)

            # process the timestamp line and the text line
            if '-->' in line:
                start_o, end_o = line.split(' --> ')
                # print(start_o, end_o)
                # check the text line
                if len(lines[i + 1].strip().split(':')) == 1:
                    # if the speaker is not found, set it to unknown
                    speaker = 'unknown'
                    text = lines[i + 1].strip()
                else:
                    speaker, text = lines[i + 1].strip().split(':')

                # check the speaker is the same as the last speaker and the match score is high
                if (end_speaker == speaker) and (fuzz.partial_ratio(target_end_text.lower(), text.lower()) > threshold):
                    # print('find the end of group discussion', i)
                    loop_status = 2  # find end of group discussion
                    # break
                    # instead of break, update the end_of_line
                    end_of_line = i - (end_lead * 4)  # keep processing until the end of the conversation

                # add transcripts, if same_speaker combine the line
                if speaker == last_speaker:
                    transcripts[-1]['text'] = transcripts[-1]['text'] + ' ' + text
                    transcripts[-1]['end'] = end_o
                else:
                    transcripts.append({'start': start_o, 'end': end_o, 'speaker': speaker, 'text': text, 'game': game})

                # update the last_speaker and last_end_time
                last_speaker = speaker

            i += 1

        # break the outer while loop, only when the start and end of group discussion are found
        if loop_status == 2:
            break
        i += 1
    # df = pd.DataFrame(transcripts)
    # transcripts
    # game_transcript
    # check the loop status
    if loop_status != 2:
        print(game, '\'s loop status is', loop_status)

    return pd.DataFrame(transcripts)


def get_time_stamp_manually(game, folder_path, start_idx, end_idx):
    # cut the transcript by the start and end date
    transcripts = []
    last_speaker = ''
    i = start_idx

    # read the vtt file
    file_path = folder_path + '/' + game + '_Transcript.vtt'
    with open(file_path, 'r') as file:
        lines = file.readlines()

    while i <= end_idx:
        line = lines[i].strip()
        # process the timestamp line and the text line
        if '-->' in line:
            start_o, end_o = line.split(' --> ')
            # print(start_o, end_o)
            # check the text line
            if len(lines[i + 1].strip().split(':')) == 1:
                # if the speaker is not found, set it to unknown
                speaker = 'unknown'
                text = lines[i + 1].strip()
            else:
                speaker, text = lines[i + 1].strip().split(':')
            # add transcripts, if same_speaker combine the line

            if speaker == last_speaker:
                transcripts[-1]['text'] = transcripts[-1]['text'] + ' ' + text
                transcripts[-1]['end'] = end_o
            else:
                transcripts.append({'start': start_o, 'end': end_o, 'speaker': speaker, 'text': text, 'game': game})

            # update the last_speaker and last_end_time
            last_speaker = speaker
        i += 1

    return pd.DataFrame(transcripts)


# check the text match
def match_text(text1, text2, threshold):
    '''
    check if the text match
    :param text1: the text to be checked
    :param text2:  the text to be checked
    :param threshold: the threshold for the fazz match score
    :return: boolean
    '''

    # preprocess the text
    def preprocess(text):
        # remove the punctuation
        text = re.sub(r'[^\w\s]', '', text)
        # remove tex in []
        text = re.sub(r'\[.*?\]', '', text)
        # replace the number to word
        word_to_digit = {"one": "1", "two": "2", "three": "3", "four": "4"}
        text = ' '.join([word_to_digit.get(word, word) for word in text.split()])

        return text.lower().strip()

    # check if the text match
    if fuzz.partial_ratio(preprocess(text1), preprocess(text2)) > threshold:
        return True
    else:
        return False


def transcript_match(game, transcript_raw, transcript_proofread, save_path):
    '''
    Matches and updates the transcript for a specific game by comparing raw and proofread transcripts.

    :param game: The game identifier.
    :param transcript_raw: The raw transcript DataFrame.
    :param transcript_proofread: The proofread transcript DataFrame.
    :param save_path: The path to save the matched transcripts.
    '''
    raw = transcript_raw[transcript_raw['game'] == game]
    proofread = transcript_proofread[transcript_proofread['game'] == game]

    raw.loc[:, 'idx_proof'] = None  # initial the idx_proof column
    raw.loc[:, 'text_proof'] = ''
    raw.loc[:, 'checked'] = False
    raw.loc[:, 'idx'] = range(1, len(raw) + 1)  # add index to df

    # initial the idx_raw column, text_raw column, start column, end column, idx column
    proofread.loc[:, 'text_raw'] = ''
    proofread.loc[:, 'idx_raw'] = None
    proofread.loc[:, 'start'] = None
    proofread.loc[:, 'end'] = None
    proofread.loc[:, 'idx'] = range(1, len(proofread) + 1)  # reset the index
    proofread.loc[:, 'Notes'] = None

    # from the game transcript match the time stamp
    for i, line in proofread.iterrows():
        for j, line2 in raw.iterrows():
            # if the line is checked, skip
            if line2['text_proof'] not in [None, '']:
                continue
            # if the line is not checked, check if the text match
            if match_text(line['text'], line2['text'], 75):
                # if the text match, check if the speaker match

                if line['speaker'] != line2['speaker'] and line2['speaker'] != 'unknown':
                    # if the speaker is not the same, skip
                    continue
                else:
                    # if the speaker is same or unknown, update the df
                    raw.loc[j, 'text_proof'] = line['text']  # update the cleaned_text
                    raw.loc[j, 'speaker'] = line['speaker']  # update the speaker
                    raw.loc[j, 'idx_proof'] = line['idx']  # update the idx_proof
                    # update the checked status in the game_transcript
                    proofread.loc[i, 'checked'] = True
                    # update the timestamp to game_transcript
                    proofread.loc[i, 'start'] = line2['start']
                    proofread.loc[i, 'end'] = line2['end']
                    proofread.loc[i, 'idx_raw'] = line2['idx']
                    proofread.loc[i, 'text_raw'] = line2['text']

                break
            # if not find the match line in

    # save the transcript
    # reorganize the columns
    raw = raw[['idx', 'start', 'end', 'speaker', 'text', 'text_proof', 'idx_proof', 'game']]
    raw.to_csv(save_path + game + '_raw.csv', index=False, date_format='%H:%M:%S.%f')
    #proofread create a column "Notes'' to record the notes
    # proofread = proofread[['idx', 'start', 'end', 'speaker', 'text', 'text_raw', 'idx_raw', 'game', 'Notes']]
    proofread.to_csv(save_path + game + '_proofread.csv', index=False, date_format='%H:%M:%S.%f')


def print_transcript(game, folder_path):
    '''
    print the transcript from the vtt file
    :param game: the game name
    :param folder_path: the folder path
    :return: the transcript line by line
    '''
    # read the vtt file
    file_path = folder_path + '/' + game + '_Transcript.vtt'
    with open(file_path, 'r') as file: lines = file.readlines()
    # go over vtt file and find the transcripts between the start and end
    i = int(len(lines) / 2)  # the discussion is at end of video, start from the middle
    while i < len(lines):
        # process the timestamp line and the text line
        line = lines[i].strip()
        print(line, '[', i, ']')
        i += 1


# read the transcript file that is already proofread
transcript_proofread = pd.read_csv('/Users/saiyingge/Resume_Study_DATA/group_discussion/GD_transcript_proofread.csv')
# change col name
transcript_proofread.columns = ['idx_o', 'game', 'speaker', 'game_role', 'text']

# count the game number, should be 22
game_names = transcript_proofread['game'].unique()
transcript_proofread['game'].nunique()
# count the speaker number, goup by game
transcript_proofread['speaker'].groupby(transcript_proofread['game']).nunique()

# get the time stamp for all games
transcript_raw = pd.concat([get_time_stamp(game, folder_path) for game in game_names])
'''XC415B 's loop status is 0 XC415C 's loop status is 1'''

# find the game not find the time_stamps_preliminary
for game in game_names:
    # 'file' only keep the game name
    if game not in transcript_raw['game'].tolist():
        print(game)

# count # rows by game, sorted by the number of rows
transcript_raw.groupby('game').count().sort_values(by='text')


# add it to the

# manually go over the transcript by game
# find game that need to look into details in the transcript_all

# game = 'XC419A'
transcript_match(game, transcript_raw, transcript_proofread, save_path)

# _____


# manually check the transcript and find start and end date
print_transcript(game, folder_path)

start_idx = 709
end_idx = 977
manually_cut = get_time_stamp_manually(game, folder_path, start_idx, end_idx)



# ...

# Convert the 'start' and 'end' columns to datetime objects in raw DataFrame
manually_cut['start'] = pd.to_datetime(manually_cut['start'], format='%H:%M:%S.%f').dt.time
manually_cut['end'] = pd.to_datetime(manually_cut['end'], format='%H:%M:%S.%f').dt.time

transcript_match(game, manually_cut, transcript_proofread, save_path)
