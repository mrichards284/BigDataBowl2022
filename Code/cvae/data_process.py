#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 10:36:59 2021

@author: pengwei
"""

import os 
os.chdir('/Users/pengwei/Box/bigdatabowl/python_code')

import pandas as pd
import matplotlib.pyplot as plt  
%matplotlib inline
import numpy as np
import os 
import boto3
from functools import reduce 
import time
from datetime import datetime


###############################################################################
###############################################################################
##                                                                           ##
##                                                                           ##
##                                                                           ##
##                                PART 0                                     ##                                                                             
##                                                                           ##
##                                                                           ##
###############################################################################
###############################################################################


###############################################################################
# load data from s3 bucket 
###############################################################################


games = s3_to_csv('nfl-big-data-bowl-2022/games.csv') # games
plays = s3_to_csv('nfl-big-data-bowl-2022/plays.csv') # plays
players = s3_to_csv('nfl-big-data-bowl-2022/players.csv') # players
scouting = s3_to_csv('nfl-big-data-bowl-2022/PFFScoutingData.csv') # scout
tracking2018 = s3_to_csv('nfl-big-data-bowl-2022/tracking2018.csv') # track2018
tracking2019 = s3_to_csv('nfl-big-data-bowl-2022/tracking2019.csv') # track2019
tracking2020 = s3_to_csv('nfl-big-data-bowl-2022/tracking2020.csv') # track2020



###############################################################################
###############################################################################
##                                                                           ##
##                                                                           ##
##                                                                           ##
##                                PART 1                                     ##                                                                             
##                                                                           ##
##                                                                           ##
###############################################################################
###############################################################################



###############################################################################
# Get plays, tracking and scouting for punts plays only
###############################################################################


## 1. plays for punts

punts = plays.loc[plays['specialTeamsPlayType']=='Punt'] # 5991
punts = punts.set_index(['gameId','playId'])
print("............the number of punts:",len(punts))

## 2. scouting for punts

scouting = scouting.set_index(['gameId','playId'])
scout = scouting.loc[scouting.index.isin(punts.index)]
scout = scout.drop(columns=['kickoffReturnFormation']) # drop useless column
print("............the number of scouts:",len(scout))

## 3. tracking for punts

# 2018
tracking2018 = tracking2018.set_index(['gameId','playId'])
track2018 = tracking2018.loc[tracking2018.index.isin(punts.index)]
# 2019
tracking2019 = tracking2019.set_index(['gameId','playId'])
track2019 = tracking2019.loc[tracking2019.index.isin(punts.index)]
# 2020
tracking2020 = tracking2020.set_index(['gameId','playId'])
track2020 = tracking2020.loc[tracking2020.index.isin(punts.index)]
# combine 2018 - 2020
track2018_20 = pd.concat([track2018, track2019, track2020], axis=0)

print("............ len(track2018):", len(track2018))
print("............ len(track2019):", len(track2019))
print("............ len(track2020):", len(track2020))


###############################################################################
# Get timestamps of key events: snap_timestamp, punt_timesnap, land_timestamp
###############################################################################



football2018_20 = track2018_20.loc[track2018_20['displayName']=='football']
time2018_20 = football2018_20[['time','event']] # (time, event) for each play 
scout2018_20 = scout.loc[scout.index.isin(time2018_20.index)]
# scout2018_20_time = scout2018_20[['snapTime','operationTime','hangTime']] 

# time spent on each event
scout2018_20_time = scout[['snapTime','operationTime','hangTime']]


# 1. obtain snap timestamp 
time2018_20_snap = time2018_20.loc[time2018_20['event']=='ball_snap']['time']
time2018_20_snap = pd.DataFrame(time2018_20_snap)
time2018_20_snap.columns = ['snap_timestamp']

# 2. obtain punt timestamp
time2018_20_punt = time2018_20.loc[time2018_20['event']=='punt']['time']
time2018_20_punt = pd.DataFrame(time2018_20_punt)
time2018_20_punt.columns = ['punt_timestamp']

# 3. obtain land timestamp
time2018_20_land = time2018_20.loc[time2018_20['event'].isin(
        ['punt_received','punt_downed','fair_catch','punt_land'])]['time']
time2018_20_land = time2018_20_land.groupby(
        ['gameId','playId']).apply(lambda df: df.iloc[0])
time2018_20_land = pd.DataFrame(time2018_20_land)
time2018_20_land.columns = ['land_timestamp']

# 4. receive timestamp is not available and needs to be estimated 



## obtain the estimated timestamp of key events: snap, receive, punt, land 

time_record = pd.merge(time2018_20_snap, scout2018_20_time, 
                       how='left', 
                       on=['gameId','playId'])


def time_str2num(a):
    # time: convert from str to numeric 
    ts = []
    for x in a: 
        x_ts = datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%f")
        x_ts = time.mktime(x_ts.timetuple()) + (x_ts.microsecond / 1000000.0)
        
        ts.append(x_ts)

    return ts

def time_num2str(a):
    # time: convert from numeric to str
    ts = []
    for x in a: 
        if not np.isnan(x):
            x = round(x,1)
            x_ts = datetime.fromtimestamp(x).strftime("%Y-%m-%dT%H:%M:%S.%f")
            x_ts = x_ts[:-3]
            ts.append(x_ts)
        else:
            ts.append(np.nan)

    return ts

# obtain estimated receive timestamp, punt timestamp, land timestamp
    
time_record['receive_timestamp_est'] = time_num2str(
        time_str2num(time_record['snap_timestamp']) 
        + time_record['snapTime']
        )
time_record['punt_timestamp_est'] = time_num2str(
        time_str2num(time_record['snap_timestamp']) 
        + time_record['operationTime']
        )

time_record['land_timestamp_est'] = time_num2str(
        time_str2num(time_record['snap_timestamp'])
        + time_record['operationTime']
        + time_record['hangTime'])

time_record = time_record[
        ['snap_timestamp', 
       'receive_timestamp_est',
       'punt_timestamp_est', 
       'land_timestamp_est']]


data_frames = [time_record, time2018_20_punt, time2018_20_land]
time_record = reduce(lambda left, right: pd.merge(left, right,
                                                  on=['gameId','playId'],  
                                                  how='left'), data_frames)
time_record = time_record[['snap_timestamp',
                           'receive_timestamp_est', 
                           'punt_timestamp_est', 
                           'punt_timestamp', 
                           'land_timestamp_est', 
                           'land_timestamp']]
#  Need to check accuracy 
# Compare the estimated timestamps with the actual ones. They are close. 
# time_record[['land_timestamp_est','land_timestamp']]


# Keep the timestamps of the key events 
time_record = time_record[['snap_timestamp',
                           'receive_timestamp_est',
                           'punt_timestamp',
                           'land_timestamp']]

# gather multiple columns to one coloumn with name = 'keyEvent'
time_record = time_record.reset_index()
time_record = pd.melt(time_record , id_vars=['gameId','playId'])
time_record = time_record.set_index(['gameId','playId'])
time_record.columns = ['keyEvent','time']
time_record = time_record.sort_index()
"""
                                keyEvent                     time
gameId     playId                                                
2018090600 366            snap_timestamp  2018-09-07T01:18:15.000
           366     receive_timestamp_est  2018-09-07T01:18:15.800
           366            punt_timestamp  2018-09-07T01:18:17.200
           366            land_timestamp  2018-09-07T01:18:21.700
           872            snap_timestamp  2018-09-07T01:40:26.700
"""


# Adjust the following sample to make it more accurate. We use this play as an 
# example to emonstrate the power of our model.

game_id_ = 2018091001 # ,2020122709  #2021010313
play_id_  =  2597 #3288 #195


de = track2018_20.loc[(game_id_, play_id_)]
de_punter = de.loc[(de['position'].isin(['P']))]
de_ball = de.loc[(de['displayName']=='football')]
 
 
de_punter['distance'] = np.sqrt((de_punter['x'] - de_ball['x'])**2 +  (de_punter['y'] - de_ball['y']) **2)
plt.plot(de_punter['distance'].values)
time_ = de_punter.loc[de_punter['distance']==min(de_punter['distance'])]['time']
time_record.loc[(game_id_, play_id_)].iloc[1]['time'] = '2018-09-11T04:21:45.000'


###############################################################################
# Sort tracking by players' positions on the field and add labels 
###############################################################################


## add a column to tracking data: keyEvent 

augmented_track2018_20 = pd.merge(track2018_20, time_record, how='inner', on=['gameId','playId','time'])
# augmented_track2018_20.loc[(2018123000, 892 )]
# augmented_track2018_20.groupby(['gameId','playId']).size().value_counts()

## filter out num_events != 3
filtering = augmented_track2018_20.groupby(['gameId','playId']).size()
filtering_index = filtering[filtering == 92].index
augmented_track2018_20 = augmented_track2018_20.loc[filtering_index] # 5404 plays


## prepare data for sorting and labeling 
dat_to_create_label = augmented_track2018_20.loc[augmented_track2018_20['keyEvent']=='snap_timestamp']
dat_to_create_label = dat_to_create_label[['time', 'x', 'y', 'team',
                                           'position', 'nflId', 'jerseyNumber',
                                           'displayName', 'playDirection']]



def labeling(df):
    """
    sorting and labeling based the players' positons on the field
    """
    # df = dat_to_create_label.loc[(2018090600, 366)]
    df = df[['x', 'y', 'team',
             'position', 'jerseyNumber', 'nflId',
             'displayName', 'playDirection']]
    
    df_home = df.loc[df['team']=='home']
    df_away = df.loc[df['team']=='away']
    df_ball = df.loc[df['team']=='football']
    
    ## determine possesion team and return team 
    if 'P' in df_home['position'].values: # punter
        df_poss = df_home
        df_return = df_away 
    else:
        df_poss = df_away
        df_return = df_home
        
    ## sort possesion team 
    df_poss_rest = df_poss.loc[df_poss['position']!='P']
    df_poss_rest = df_poss_rest.sort_values(['y'])
    df_poss = pd.concat([df_poss.loc[df_poss['position']=='P'], df_poss_rest])
    
    ## sort return team 
    if df_return['playDirection'][0]=='right':
        
        df_return_rest = df_return.loc[df_return['x']!=max(df_return['x'])]
        df_return_rest = df_return_rest.sort_values(['y'])
    
        df_return = pd.concat(
                [df_return.loc[df_return['x']==max(df_return['x'])],
                               df_return_rest])
        
    else:
        df_return_rest = df_return.loc[df_return['x']!=min(df_return['x'])]
        df_return_rest = df_return_rest.sort_values(['y'])
    
        df_return = pd.concat(
                [df_return.loc[df_return['x']==min(df_return['x'])],
                               df_return_rest])
    
    
    
    df = pd.concat([df_poss, df_return, df_ball])
    df['label'] = list(range(1,24)) # labeling 

    
    return df[['nflId','team','jerseyNumber', 'displayName','label']]


labels = dat_to_create_label.groupby(['gameId','playId']).apply(labeling)

augmented_track2018_20 = augmented_track2018_20.drop(
        columns=['team', 'displayName','jerseyNumber'])

universe = pd.merge(labels, 
                    augmented_track2018_20, 
                    how='outer',
                    on=['gameId','playId','nflId'])

# universe contains all information for modeling 
# importantly, it is sorted according to the players' positions in the field

# reverse the play so that always punt from left to right

universe['x'] = universe['x']  + \
    (universe['playDirection'] == 'left') * (120 - 2 * universe['x'] )
universe['y'] = universe['y']  + \
    (universe['playDirection'] == 'left') * (53.3 - 2 * universe['y'] )
universe['o'] = ((universe['playDirection'] == 'left') * 180  + universe['o'] ) % 360 
universe['dir'] = ((universe['playDirection'] == 'left') * 180  + universe['dir'] ) % 360 

# check if revert correctly 

universe.loc[(universe['position']=='P') & (universe['keyEvent'] =='snap_timestamp')]['x'].hist()



# save data
labels.to_csv('/Users/pengwei/Box/bigdatabowl/python_code/labels.csv')
time_record.to_csv('/Users/pengwei/Box/bigdatabowl/python_code/keyEvent_timestamp.csv')
universe.to_csv('/Users/pengwei/Box/bigdatabowl/python_code/universe.csv')
punts.to_csv('/Users/pengwei/Box/bigdatabowl/python_code/punts.csv')
scout.to_csv('/Users/pengwei/Box/bigdatabowl/python_code/scout.csv')



