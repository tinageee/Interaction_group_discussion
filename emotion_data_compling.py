
import pandas as pd
import os
import datetime
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)



dpath = '/Users/saiyingge/**Research Projects/Resume/data/faceplus/'
discussion_info = pd.read_csv('/Users/saiyingge/Resume_Study_DATA/group_discussion/data/discussion_data_add_speakers.csv', dtype={19: 'object', 20: 'object'})

variable_List = ['smile'] + ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']
info_List = ['game','speaker','stage','imgindex','player']
today = datetime.date.today()


discussion_info= discussion_info[info_List+ variable_List]

file_name = [file for file in os.listdir(dpath) if file.endswith('.csv')]
game_name = list(set([ee.split('_')[0] for ee in file_name])); game_name.sort();
print(len(game_name))

all_data = pd.DataFrame()
#read csv from folder
for file in file_name:
    df=pd.read_csv(dpath+file, index_col=None)
    # process intro & note data and discussion data separately
    # disucssion data
    if 'Speaker'  in file:
        continue
    elif 'Subject' in df.columns:
        continue
    elif not 'speaker' in df.columns:
        df = df [['game','imgindex']+variable_List]
        # add speaker, stage, player info, parse from file name, eg. ['XC415A', 'Bravo', 'Intro', 'Bravo.csv']
        df['speaker'] = file.split('_')[1]
        df['stage'] = file.split('_')[2]
        df['player'] = file.split('_')[3].split('.')[0]

    all_data = all_data.append(df[info_List+variable_List], ignore_index=True)

#concatenate discussion_info and all_data
all_data = pd.concat([discussion_info, all_data], ignore_index=True)

all_data['speaker'].value_counts()
all_data['stage'].value_counts()
#count unique game names, should be 22 games
all_data.nunique()['game']

# Check for duplicate entries in the index columns
duplicates = all_data.duplicated(subset=['game', 'speaker', 'stage', 'imgindex','player'])
all_data[duplicates]

# remove duplicates
all_data = all_data[~duplicates]
all_data = all_data[~all_data['imgindex'].str.endswith('(1)')]
# XC422 has only 4 speakers, remove Bravo from Intro stage
all_data = all_data[~((all_data['game']=='XC422A') & (all_data['stage']=='Intro') & (all_data['speaker']=='Bravo'))]

# check missing data
all_data.isnull().sum()
# when stage is discussion, count unique number of speaker for each game
len(all_data[all_data['stage']=='Discussion'].groupby(['game','stage']).nunique()['speaker'])
# ---- should be 22 games
# count unique number of speaker for each game ,show number that note and intro stage is smaller than 5
all_data[all_data['stage'].isin(['Intro', 'Note'])].groupby(['game','stage']).nunique()['speaker'][all_data[all_data['stage'].isin(['Intro', 'Note'])].groupby(['game','stage']).nunique()['speaker']<5]
# ------XC422A only has 4 player, XW428A player C did not present in note stage

#save all_data to csv
all_data.to_csv('/Users/saiyingge/**Research Projects/Resume/data/emotion_data.csv', index=False)
# save with timestamp
all_data.to_csv('/Users/saiyingge/**Research Projects/Resume/data/emotion_data_'+str(today)+'.csv', index=False)