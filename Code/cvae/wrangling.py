#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 15:36:59 2021

@author: pengwei
"""


import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
import seaborn as sns 
import sys 
import os 
import boto3 # for Aws s3 bucket
import time
import datetime




"""
pure data Wranling to understand the data, no output
"""


###############################################################################
# 0. load data from s3 bucket 
###############################################################################


games = s3_to_csv('nfl-big-data-bowl-2022/games.csv')
plays = s3_to_csv('nfl-big-data-bowl-2022/plays.csv')
players = s3_to_csv('nfl-big-data-bowl-2022/players.csv')
scouting = s3_to_csv('nfl-big-data-bowl-2022/PFFScoutingData.csv')
tracking2018 = s3_to_csv('nfl-big-data-bowl-2022/tracking2018.csv')
tracking2019 = s3_to_csv('nfl-big-data-bowl-2022/tracking2019.csv')
tracking2020 = s3_to_csv('nfl-big-data-bowl-2022/tracking2020.csv')




###############################################################################
# 1. data wrangling and visulization
###############################################################################



plays.groupby(['gameId']).size() # number of plays in each game 
scouting.groupby(['gameId']).size()
sum(plays.groupby(['gameId']).size() == scouting.groupby(['gameId']).size())



print("........... # of games: ", len(games))
print("........... # of plays: ", len(plays))
print("........... # of kicks/punts", len(scouting))
print("........... # of plays in 2018 season", tracking2018.playId.nunique())
print("........... # of plays in 2019 season", tracking2019.playId.nunique())
print("........... # of plays in 2020 season", tracking2020.playId.nunique())
      
# games 
# players
# tracking 

tracking2018.event.value_counts()
      
# plays 
plays.iloc[0]


plays.loc[plays.specialTeamsResult == 'Non-Special Teams Result'].passResult

plays.iloc[0]

"""
gameId                                                           2018090600
playId                                                                   37
playDescription           J.Elliott kicks 65 yards from PHI 35 to end zo...
quarter                                                                   1
down                                                                      0
yardsToGo                                                                 0
possessionTeam                                                          PHI
specialTeamsPlayType                                                Kickoff
specialTeamsResult                                                Touchback
kickerId                                                              44966
returnerId                                                              NaN
kickBlockerId                                                           NaN
yardlineSide                                                            PHI
yardlineNumber                                                           35
gameClock                                                          15:00:00
penaltyCodes                                                            NaN
penaltyJerseyNumbers                                                    NaN
penaltyYards                                                            NaN
preSnapHomeScore                                                          0
preSnapVisitorScore                                                       0
passResult                                                              NaN
kickLength                                                               66
kickReturnYardage                                                       NaN
playResult                                                               40
absoluteYardlineNumber                                                   45
"""

plays.specialTeamsPlayType.value_counts() # 19979
"""
Kickoff        7843
Punt           5991
Extra Point    3488
Field Goal     2657
"""
scouting.kickoffReturnFormation.value_counts().sum() # only kick off has formation
plays.groupby(['specialTeamsPlayType','specialTeamsResult']).size()

"""
Extra Point           Blocked Kick Attempt          24
                      Kick Attempt Good           3252
                      Kick Attempt No Good         199
                      Non-Special Teams Result      13
                      
Field Goal            Blocked Kick Attempt          37
                      Downed                         1
                      Kick Attempt Good           2218
                      Kick Attempt No Good         386
                      Non-Special Teams Result      14
                      Out of Bounds                  1
                      
Kickoff               Downed                         4
                      Fair Catch                     5
                      Kickoff Team Recovery         16
                      Muffed                        60
                      Out of Bounds                 64
                      Return                      2921
                      Touchback                   4773
                      
Punt                  Blocked Punt                  39
                      Downed                       829
                      Fair Catch                  1640
                      Muffed                       154
                      Non-Special Teams Result      74
                      Out of Bounds                586
                      Return                      2286
                      Touchback                    383
                      
"""


plays.groupby(['specialTeamsPlayType','down']).size()
"""
specialTeamsPlayType  down
Extra Point           0       3488
Field Goal            1         79
                      2         88
                      3         72
                      4       2418
Kickoff               0       7843
Punt                  4       5991
"""


plays.specialTeamsResult.value_counts() # meaning, 19979
""""
Kick Attempt Good           5470
Return                      5207
Touchback                   5156
Fair Catch                  1645
Downed                       834
Out of Bounds                651
Kick Attempt No Good         585
Muffed                       214
Non-Special Teams Result     101
Blocked Kick Attempt          61
Blocked Punt                  39
Kickoff Team Recovery         16
"""

plays.returnerId.notnull().sum() # 6938, multiple returners
plays.yardlineSide.value_counts().count() # 33 OAK -> LV

plays.loc[plays.specialTeamsResult == 'Non-Special Teams Result'].passResult.value_counts(dropna=False)

plays.kickReturnYardage.notnull().sum()
plays.specialTeamsResult.value_counts()['Return'] # ?


# scouting 

scouting.iloc[0]

"""
gameId                         2018090600
playId                                 37
snapDetail                            NaN
snapTime                              NaN
operationTime                         NaN
hangTime                             3.85
kickType                                D
kickDirectionIntended                   R
kickDirectionActual                     R
returnDirectionIntended               NaN
returnDirectionActual                 NaN
missedTackler                         NaN
assistTackler                         NaN
tackler                               NaN
kickoffReturnFormation              8-0-2
gunners                               NaN
puntRushers                           NaN
specialTeamsSafeties       PHI 23; PHI 27
vises                                 NaN
kickContactType                       NaN
"""


scouting.snapDetail.value_counts()

scouting.kickType.value_counts()
"""
D    6944
N    4095
A    1781
F     319
P     226
O     155
Q     134
K      50
S      14
R       4
B       1
"""

scouting.kickDirectionIntended.value_counts().sum() # 13701
scouting.returnDirectionIntended.value_counts().sum() # 4754
scouting.missedTackler.notnull().sum() # 1348
scouting.tackler.notnull().sum() # 4759

scouting.kickoffReturnFormation.notnull().sum() # 7843 = number of kick off
scouting.gunners.notnull().sum() # 5901 < number of punts
scouting.puntRushers.notnull().sum() # 3010

scouting.kickContactType.value_counts() # 5879

"""
CC      3745
BF      1060
BB       311
OOB      284
CFFG     150
MBDR     145
DEZ       66
KTC       31
BC        23
KTB       22
ICC       16
KTF       12
MBC        9
BOG        5
"""



###############################################################################

###############################################################################



"""
Punt                  Blocked Punt                  39
                      Downed                       829
                      Fair Catch                  1640
                      Muffed                       154
                      Non-Special Teams Result      74
                      Out of Bounds                586
                      Return                      2286
                      Touchback                    383
"""
                     
"""
gameId                                                           2018090600
playId                                                                   37
playDescription           J.Elliott kicks 65 yards from PHI 35 to end zo...
quarter                                                                   1
down                                                                      0
yardsToGo                                                                 0
possessionTeam                                                          PHI
specialTeamsPlayType                                                Kickoff
specialTeamsResult                                                Touchback
kickerId                                                              44966
returnerId                                                              NaN
kickBlockerId                                                           NaN
yardlineSide                                                            PHI
yardlineNumber                                                           35
gameClock                                                          15:00:00
penaltyCodes                                                            NaN
penaltyJerseyNumbers                                                    NaN
penaltyYards                                                            NaN
preSnapHomeScore                                                          0
preSnapVisitorScore                                                       0
passResult                                                              NaN
kickLength                                                               66
kickReturnYardage                                                       NaN
playResult                                                               40
absoluteYardlineNumber                                                   45
"""

punts = plays.loc[plays['specialTeamsPlayType']=='Punt'] # 5991
punts = punts.set_index(['gameId','playId'])

punts.grouby['passResults']

punts.groupby(['specialTeamsResult']).size()
"""
specialTeamsResult
Blocked Punt                  39
Downed                       829
Fair Catch                  1640
Muffed                       154
Non-Special Teams Result      74
Out of Bounds                586
Return                      2286
Touchback                    383

"""

punts.loc[punts.specialTeamsResult=='Return'].index
"""
MultiIndex([(2018090600,  366),
            (2018090600, 1989),
            (2018090600, 2599),
            (2018090600, 3868),
            (2018090900,  485),
            (2018090900,  575),
            (2018090900,  674),
            (2018090900, 1136),
            (2018090900, 1360),
            (2018090900, 3079),
            ...
            (2021010313, 3431),
            (2021010313, 3537),
            (2021010314,  429),
            (2021010314, 1858),
            (2021010315,  175),
            (2021010315,  928),
            (2021010315, 1359),
            (2021010315, 1479),
            (2021010315, 2235),
            (2021010315, 2497)],
"""
punts.loc[(2018090600,  366)]
punts.loc[(2018090600,  366)].playDescription


scouting = scouting.set_index(['gameId','playId'])
scout = scouting.loc[scouting.index.isin(punts.index)]
scout['kickoffReturnFormation'].value_counts(dropna=False) # 5991 na

"""
snapDetail                                             OK
snapTime                                             0.84
operationTime                                        2.12
hangTime                                             4.46
kickType                                                N
kickDirectionIntended                                   C
kickDirectionActual                                     C
returnDirectionIntended                                 C
returnDirectionActual                                   R
missedTackler                                      PHI 57
assistTackler                                         NaN
tackler                                            PHI 54
gunners                                    PHI 18; PHI 29
puntRushers                                           NaN
specialTeamsSafeties                                  NaN
vises                      ATL 83; ATL 27; ATL 34; ATL 21
kickContactType                                        CC

"""

scout = scout.drop(columns=['kickoffReturnFormation']) # drop 
scout.iloc[0]


scout.kickType.value_counts()
"""
N    4095 : Normal - standard punt style
A    1779 : Rugby style punt
R       4 : Nose down or Aussie-style punts
"""

scout.groupby(['kickDirectionIntended','kickDirectionActual']).size()
"""
kickDirectionIntended  kickDirectionActual
C                      C                      2731
                       L                        15
                       R                        14
L                      C                        90
                       L                      1668
                       R                         1
R                      C                        71
                       R                      1275
"""
scout.groupby(['returnDirectionIntended','returnDirectionActual']).size()
"""
returnDirectionIntended  returnDirectionActual
C                        C                        763
                         L                         61
                         R                         52
L                        C                         28
                         L                        462
                         R                          5
R                        C                         27
                         L                          7
                         R                        527
"""                         

scout['kickContactType'].value_counts()
# Detail on how a punt was fielded, 
# or what happened when it wasn't fielded (text).
"""

BF      1060 : bounced forward
BB       310 : boounced backwrads
BC        23 : Bobbled Catch from Air
BOG        5 : Bobbled in Ground

CC      3745 : clean catch from air
CFFG     150 : clean field from groud ?


DEZ       66 : direct to EndZone
ICC       16 : incidental coverage team contact


KTC       31 : kick team catch
KTB       22 : kick team knocked back
KTF       12 : kcik team knocked forward

MBDR     145 : muffed by designated returner
MBC        9 : muffed by contact iwth non-designated returner

OOB      283 :Directly out of bound 

"""
"""
Punt                  Blocked Punt                  39
                      Downed                       829
                      Fair Catch                  1640
                      Muffed                       154
                      Non-Special Teams Result      74
                      Out of Bounds                586
                      Return                      2286
                      Touchback                    38
"""


tracking2018 = tracking2018.set_index(['gameId','playId'])
track2018 = tracking2018.loc[tracking2018.index.isin(punts.index)]



track2018['event'].value_counts()
track2018.index.value_counts()  # 2154

track2018.loc[(2018123000,892)]['time'].nunique() # 85 frame 
track2018.loc[(2018123000,892)]['event'].value_counts()
"""
None          1886
ball_snap       23
punt            23
fair_catch      23
"""
"""
(2018122400, 241)     6256
(2018100704, 324)     6026
(2018111105, 3239)    5888
(2018091610, 3871)    5750
(2018110401, 2180)    5589
(2018122305, 3416)    1196


(2018122305, 345)     1127
(2018122309, 4114)    1127
(2018102101, 122)     1035
(2018121603, 2601)    1012
"""

_gameId, _playId = (2018123000,892) # (2018091610, 3871)#(2018122305, 345) 

a = track2018.loc[(_gameId, _playId)]
b = a[a['displayName']=='football']
startTime = b.iloc[0]['time']
snapTime = b.loc[b['event'] =='ball_snap']['time'][0]
puntTime = b.loc[b['event'] == 'punt']['time'][0]
locateTime = b.loc[b['event'] == 'fair_catch']['time'][0]
print(startTime, snapTime )
print(startTime, puntTime)
print(puntTime, locateTime)



scout.loc[(_gameId, _playId)][['snapTime','operationTime','hangTime']]


football2018=track2018.loc[track2018['displayName']=='football']
football2018[football2018['event']!='None']
football2018.loc[(_gameId,_playId)]



myDate = "2014-08-01 04:41:52.117"

dt = datetime.datetime.strptime(myDate, "%Y-%m-%d %H:%M:%S.%f")
time.mktime(dt.timetuple()) + (dt.microsecond / 1000000.0)
1406864512.117

time = []

def fun(df):
    ts = df.iloc[0]['time']
    ts = datetime.datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S.%f")
    ts = time.mktime(ts.timetuple()) + (dt.microsecond / 1000000.0)

    return ts




startTime = football2018.groupby(['gameId','playId']).apply(fun)
catchTime = scout.loc[startTime.index]['snapTime'] + startTime
puntTime = scout.loc[startTime.index]['operationTime'] + startTime 
hangTime = scout.loc[startTime.index]['hangTime'] + startTime 
majorEvent = football2018.loc[football2018['event']!='None']['event'].groupby(['gameId','playId']).apply(list)


majorEvent.value_counts()

for x,y in majorEvent.value_counts().items():
    print(x,y)
    
# punt play without ball_snap 
for x,y in majorEvent.value_counts().items():
    if 'ball_snap' not in x or 'punt' not in x:
        print(x,y)
"""
['ball_snap', 'punt_fake', 'run', 'first_contact', 'tackle'] 4
['ball_snap', 'punt_fake', 'pass_forward', 'pass_arrived', 'pass_outcome_caught', 'tackle'] 3
['ball_snap', 'punt_fake', 'pass_forward', 'pass_outcome_incomplete'] 2
['ball_snap', 'punt_fake', 'run', 'out_of_bounds'] 2
['punt', 'punt_received', 'tackle'] 2
['ball_snap', 'punt_fake', 'run', 'tackle'] 2
['ball_snap', 'punt_fake', 'pass_forward', 'pass_arrived', 'pass_outcome_caught', 'first_contact', 'tackle'] 2
['line_set', 'ball_snap', 'punt_fake', 'pass_forward', 'pass_arrived', 'pass_outcome_caught', 'first_contact', 'tackle'] 1
['ball_snap', 'handoff', 'punt_fake', 'first_contact', 'out_of_bounds'] 1
['ball_snap', 'punt_fake', 'pass_forward', 'pass_arrived', 'pass_arrived', 'pass_outcome_incomplete'] 1
['ball_snap', 'line_set', 'punt_blocked', 'fumble_defense_recovered', 'tackle'] 1
['line_set', 'ball_snap', 'punt_fake', 'run', 'first_contact', 'tackle'] 1
['ball_snap', 'punt_fake', 'pass_forward', 'pass_arrived', 'pass_outcome_caught', 'out_of_bounds'] 1
['snap_direct', 'punt_fake', 'first_contact', 'tackle'] 1
['snap_direct', 'punt_fake', 'run', 'first_contact', 'tackle'] 1
['man_in_motion', 'ball_snap', 'punt_fake', 'run', 'tackle'] 1
['snap_direct', 'run', 'tackle'] 1


['punt', 'punt_received', 'first_contact', 'tackle'] 4
['punt', 'fair_catch'] 3
['punt', 'punt_land', 'punt_downed'] 3
['punt', 'punt_land', 'touchback'] 1
['punt', 'out_of_bounds'] 1


['ball_snap', 'fumble', 'fumble_offense_recovered', 'tackle'] 
['snap_direct', 'fumble', 'fumble_offense_recovered', 'first_contact', 'tackle'] 1

"""


# punt play without ball_snap 
for x,y in majorEvent.value_counts().items():
    if 'ball_snap'  in x and 'punt' in x:
        ball_snap_index = x.index('ball_snap')
        x = x[ball_snap_index:]
        print(x,y)
        
# punt play without ball_snap 
land_outcomes = {}
for x,y in majorEvent.value_counts().items():
    if 'ball_snap' in x and 'punt' in x:
        land_index = x.index('punt') + 1
        if land_index < len(x) - 1:
            outcome = x[land_index]
            if outcome in land_outcomes:
                land_outcomes[outcome] += 1
            else:
                land_outcomes[outcome] = 1
                
keys = list(land_outcomes.keys())
values = list(land_outcomes.values())

plt.barh(keys, values)

        
        

        
    
majorEvent = pd.DataFrame(majorEvent)

time2018 = football2018[['time','event']]
scout2018 = scout.loc[scout.index.isin(time2018.index)]
scout2018.iloc[0]



scout2018_time = scout2018[['snapTime','operationTime','hangTime']]

# punt timestamp
time2018_punt = time2018.loc[time2018['event']=='punt']['time']
time2018_punt = pd.DataFrame(time2018_punt)
time2018_punt.columns = ['punt_timestamp']




# snap timestamp 
time2018_snap = time2018.loc[time2018['event']=='ball_snap']['time']
time2018_snap = pd.DataFrame(time2018_snap)
time2018_snap.columns = ['snap_timestamp']


# land timestamp
time2018_land = time2018.loc[time2018['event'].isin(['punt_received','punt_downed','fair_catch','punt_land'])]['time']

time2018_land = time2018_land.groupby(['gameId','playId']).apply(lambda df: df.iloc[0])
time2018_land = pd.DataFrame(time2018_land)
time2018_land.columns = ['land_timestamp']



wp = pd.merge(time2018_snap, scout2018_time, how='left', on=['gameId','playId'])


from datetime import datetime


def fun(a):
    ts = []
    for x in a: 
        x_ts = datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%f")
        x_ts = time.mktime(x_ts.timetuple()) + (x_ts.microsecond / 1000000.0)
        
        ts.append(x_ts)

    return ts

def refun(a):
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

wp['reveive_timestamp_est'] = refun(fun(wp['snap_timestamp']) + wp['snapTime'])
wp['punt_timestamp_est'] = refun(fun(wp['snap_timestamp']) + wp['operationTime'])
wp['land_timestamp_est'] = refun(fun(wp['snap_timestamp']) + wp['hangTime'] + wp['operationTime'])

wp = wp[['snap_timestamp', 
       'reveive_timestamp_est',
       'punt_timestamp_est', 
       'land_timestamp_est']]

from functools import reduce
# compile the list of dataframes you want to merge
data_frames = [wp, time2018_punt,time2018_land]
wp = reduce(lambda  left,right: pd.merge(left,right,on=['gameId','playId'],  how='left'), data_frames)
wp = wp[['snap_timestamp', 'reveive_timestamp_est', 'punt_timestamp_est', 'punt_timestamp', 'land_timestamp_est', 'land_timestamp']]


wp[['land_timestamp_est','land_timestamp']]
# only interested in the following events 

wp = wp[['reveive_timestamp_est','punt_timestamp','land_timestamp']]


# need to check accuracy 
wp = wp.reset_index()

wp = pd.melt(wp, id_vars=['gameId','playId'])
wp = wp.set_index(['gameId','playId'])
wp.columns = ['event','time']
wp = wp.sort_index()

ww = pd.merge(track2018, wp, how='inner', on=['gameId','playId','time'])
ww.loc[(2018123000, 892 )]

ww.groupby(['gameId','playId']).size().value_counts()





























