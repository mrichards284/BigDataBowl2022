#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 15:57:03 2021

@author: pengwei
"""


import os 
os.chdir('/Users/pengwei/Box/bigdatabowl/python_code')

import pandas as pd
import matplotlib.pyplot as plt
# import matplotlib.pyplot as plt 
import numpy as np
import os 
import boto3
# from functools import reduce 
# import time
# from datetime import datetime
# from sklearn.model_selection import train_test_split
# from CVAE_sequence import c2VAE
# from CVAE_sequence import Dataset
import torch
# from torch.utils.data import DataLoader
# from collections import defaultdict
# from torch.nn import functional as F
import pickle
# import seaborn as sns 
# import sys 



    
with open('/Users/pengwei/Box/bigdatabowl/python_code/punt_vae.pkl', 'rb') as inp:
    vae = pickle.load(inp)
    
with open('/Users/pengwei/Box/bigdatabowl/python_code/model_x.pkl', 'rb') as inp:
    model_x = pickle.load(inp)
    
with open('/Users/pengwei/Box/bigdatabowl/python_code/model_y.pkl', 'rb') as inp:
    model_y = pickle.load(inp)
    
    

     
# all data
universe = pd.read_csv('/Users/pengwei/Box/bigdatabowl/python_code/universe.csv')
universe = universe.set_index(['gameId','playId'])

# check if revert correctly 
universe.loc[(universe['position']=='P') & (universe['keyEvent'] =='snap_timestamp')]['x'].hist()


# test data 
        
sub_universe = pd.read_csv('/Users/pengwei/Box/bigdatabowl/python_code/sub_universe.csv')
sub_universe = sub_universe.set_index(['gameId','playId'])


    
    



###############################################################################
# determine a sample play

###############################################################################



"""
candi_punts = punts.loc[X_test.index]
candi_scout = scout.loc[X_test.index]
candi_track = track2018_20.loc[X_test.index]

candi_direc = candi_track[candi_track['team']=='football'].groupby(['gameId','playId']).apply(lambda df: df.iloc[0])[['x','playDirection']]



### right punts 
right_punts = candi_punts.loc[candi_direc['playDirection']=='right','absoluteYardlineNumber']
right_punts = pd.merge(right_punts, candi_direc, on=['gameId','playId'], how='left')

right_punts['kickDirectionIntended'] = candi_scout.loc[right_punts.index]['kickDirectionIntended']
right_punts.loc[:,'appYardlineNumber'] = np.ceil(right_punts.loc[:,'absoluteYardlineNumber'] / 5) * 5
right_punts.groupby(['kickDirectionIntended','appYardlineNumber']).size()


right_punts
plt.scatter(right_punts['absoluteYardlineNumber'], right_punts['x'])
plt.xlabel('absoluteYardlineNumber')
plt.ylabel('x position')



### left punts 

left_punts = candi_punts.loc[candi_direc['playDirection']=='left','absoluteYardlineNumber']
left_punts = pd.merge(left_punts, candi_direc, on=['gameId','playId'], how='left')
left_punts.loc[:,'appYardlineNumber'] = np.ceil(left_punts.loc[:,'absoluteYardlineNumber'] / 5) * 5
left_punts['kickDirectionIntended'] = candi_scout.loc[left_punts.index]['kickDirectionIntended']
left_punts.groupby(['kickDirectionIntended','appYardlineNumber']).size()

plt.scatter(left_punts['absoluteYardlineNumber'], left_punts['x'])
plt.xlabel('absoluteYardlineNumber')
plt.ylabel('x position')

"""


punt_max_x = universe.loc[(universe['team']=='football') & (universe['keyEvent']=='receive_timestamp_est')]['x'].max() 

punt_min_x = universe.loc[(universe['team']=='football') & (universe['keyEvent']=='receive_timestamp_est')]['x'].min() 

print("max starting x of football:", punt_max_x) #  73.29
print("min starting y of football:", punt_min_x) # 0.02


land_max_x = universe.loc[(universe['team']=='football') & (universe['keyEvent']=='land_timestamp')]['x'].max() 
land_min_x = universe.loc[(universe['team']=='football') & (universe['keyEvent']=='land_timestamp')]['x'].min() 

land_max_y = universe.loc[(universe['team']=='football') & (universe['keyEvent']=='land_timestamp')]['y'].max() 
land_min_y = universe.loc[(universe['team']=='football') & (universe['keyEvent']=='land_timestamp')]['y'].min() 

print("land_x range:", land_min_x,  land_max_x) # 15.42 113.32 # blocked?
print("land_y range:", land_min_y,  land_max_y) # -8.4 58.57


# sub_universe = universe.loc[X_test.index]



# pick a play from testing data set 

sub_universe.groupby(['gameId','playId']).size()
comet = sub_universe.loc[(2021010313,  3431)]
comet_snap = comet.loc[comet['keyEvent']=='snap_timestamp']
comet_snap [['x','y','position','label']]

comet_receive = comet.loc[comet['keyEvent']=='receive_timestamp_est']
comet_receive[['x','y','position','label']]

res_part = comet_receive.iloc[-1]['x'] - int(comet_receive.iloc[-1]['x'])
int_part = int(comet_receive.iloc[-1]['x'])
comet_receive['x'] = comet_receive['x'] - res_part
comet_receive[['x','y','team','jerseyNumber']]
# make sure it is standard play (kicklength)



def getInitialStatus(region, direction):
    # x = 52.62, y = 30.11
    # region: (10)-15, (15)-20, .....65-70 
    # direction: L, C, R 
    
    if int_part % 5 == 0:
        region_0 = int_part
    else:
        region_0 = (int_part // 5  + 1) * 5
    region_0
        
        
    play = comet_receive[comet_receive.columns]
    play['x'] = play['x'] - (region_0 - region) # region_0 = 55
    # 1: punter 
    # 2,11: guner 
    # 13, 22: vise 
    # 23: football 
    to_change = [1,3,4,5,6,7,8,9,10, 
                 12, 14, 15, 16, 17, 18, 19, 20, 21, 23]
    
    if direction == 'C':
        play.loc[play.label.isin(to_change),'y'] = play.loc[play.label.isin(to_change),'y'] - (30.11 - 26.6)
    elif direction =='L':
        play.loc[play.label.isin(to_change),'y'] = play.loc[play.label.isin(to_change),'y'] - 2 * (30.11 - 26.6)
        
    return play
    

ice = getInitialStatus(region=70, direction='C')
ice[['x','y','position','label']]



###############################################################################
# distance, hangtime 
# kick position, kick distance
# -10 - 0: + 30 - 80 
# 0 - 10 : + 30 - 80
# 10 - 20: + 30 -  80
# 20 - 30: + 30 - 80
# 30 - 40: + 30 - 80 
# 40 + 50: + 30 - 60 
# 50 - 60: + 20 + 50 

## distance vs  hang time 
# 30 - 40: 2.5 - 5
# 40 - 50: 2.5 - 5.5 
# 50 - 60: 2.5 - 5.5 
# 60 - 70: 3 - 5.5 
# 70 - 80: 4 - 5
###############################################################################

def getDistanceRange(x):
    # x: x position of the football when kicking 
    distances = None

    if 0 <= x <= 50: 
        distances = list(range(30, 73, 2))
    elif 50 < x <= 60:
        distances = list(range(30, 61, 2))
    elif 60 < x <= 70:
        distances = list(range(20, 51, 2))
    
    return distances 

def getHangTimeRange(x, d):
    # x: x position of the football 
    # d : kick distance
    if 0 < x <= 30:
        region = "pinned"
    elif 30 < x <= 56: 
        region = "middle"
    elif 56 < x <= 70:
        region = "opponent_territory"
        
    hangTimes = None
    
    if region != "opponent_territory":
        
        if 30 <= d <= 40:
            hangTimes = np.linspace(2.5, 5, 26)
        elif 40 < d <= 50:
            hangTimes = np.linspace(2.5, 5.5, 31)
        elif 50 < d <= 60:
            hangTimes = np.linspace(2.5, 5.5, 31)
        elif 60 < d <= 78: 
            hangTimes = np.linspace(3, 5.5, 26)
        elif 68 < d <= 72:
            hangTimes = np.linspace(4, 5, 11)
    else:
        if 20 <= d <= 30:
            hangTimes = np.linspace(3.5, 5, 16)
        elif 30 < d <= 40:
            hangTimes = np.linspace(3, 5, 21)
        elif 40 < d <= 50:
            hangTimes = np.linspace(3.5, 5, 16)
        elif 50 < d <= 60:
            hangTimes = np.linspace(3.7, 5, 14)
        
    return hangTimes
        

getDistanceRange(62)
getHangTimeRange(x=57, d=50)




###############################################################################
# sample play 

###############################################################################



count = 0

pseudo = []

for region in [20,30,40,50,60,70]: #[15,20,25,30,35,40,45,50,55,60,65,70]:
    for direction in ['L','C','R']:
        
        play = getInitialStatus(region, direction)
        football_x = play.iloc[-1]['x']
        distances = getDistanceRange(football_x)
    
        for dist in distances:
            
            land_x = football_x + dist 
            hangTimes = getHangTimeRange(football_x, dist)
    
            for hang_time in hangTimes:
                for land_y in range(-6, 60, 2):
                    
                    playId = (str(region)
                    + '/' + direction 
                    + '/' + str(int(land_x))
                    + '/' + str(int(land_y))
                    + '/' + str(hang_time))
                    
                    vec = []
                    vec += [playId]
                    vec += play['x'].tolist()
                    vec += play['y'].tolist()
                    vec += [int(land_x), int(land_y), hang_time]
                    
                    pseudo.append(vec)
                
                    
                    count += 1
                    if count % 100000 == 0:
                        print(count/100000,'m')
                    
pseudo = pd.DataFrame(pseudo)

columns = ['playId'] + \
    [str(i)+'_x' for i in range(1, 24)] + \
    [str(i)+'_y' for i in range(1, 24)] + \
    ['land_x','land_y','hang_time']
    
pseudo.columns = columns 

pseudo = pseudo.set_index(['playId'])



###############################################################################
# 1. xgb 
###############################################################################

data = pseudo.iloc[:,[11, 34, 22, 45, 46, 47, 48]]

data.iloc[:,:6] = data.iloc[:,:6] / 100


data.columns = ['returner_x', 'returner_y', 'football_x', 'football_y', 'land_x','land_y','hang_time']




data['x_diff'] = data['returner_x'] - data['land_x']
data['y_diff'] = data['returner_y'] - data['land_y']
data['diff'] = np.sqrt(data['x_diff'] ** 2 + data['y_diff']** 2)



data['y_in_down'] = data['land_y'] > 0.0
data['y_in_up'] = data['land_y'] < 0.5
data['x_in'] = data['land_x'] < 1.1


data['x_sign'] = data['x_diff'] > 0
data['y_sign'] = data['y_diff'] > 0


data['x_speed'] = data['x_diff'] / data['hang_time']
data['y_speed'] = data['y_diff'] / data['hang_time']
data['speed'] = data['diff'] / data['hang_time']

plt.hist(data['speed'])

col_x = ['diff', 'speed','x_sign','x_in']
col_y = ['diff', 'speed','y_sign','y_in_down', 'y_in_up']


ret = pd.DataFrame()
ret['x'] = model_x.predict(data[col_x]) + data['land_x'].values
ret['y'] = model_y.predict(data[col_y]) + data['land_y'].values


## valide 

axis_difference = (data[['land_x','land_y']].values - ret.values) 
distance = np.sqrt(np.sum(axis_difference**2, axis=1)) * 100

plt.hist(distance, bins=20)
# plt.hist(distance, range=(0,2))  
print("............proportion dist < 2: ", sum(distance < 2) / len(distance)) # 0.29
print("............proportion dist < 4: ", sum(distance < 4) / len(distance)) # 0.44


plt.hist(ret['x']*100)
plt.hist(ret['y']*100)


ret_20_values = np.repeat(ret.values, 10, axis=0)
ret_20_values = torch.tensor(ret_20_values)



###############################################################################
# 2. vae
###############################################################################

pseudo_20_values = np.repeat(pseudo.values, 10, axis=0)
pseudo_20_index = [x +'/'+str(i) for x in pseudo.index for i in range(10)]


# pseudo_20 = pd.DataFrame(pseudo_20_values, pseudo_20_index)

sam_y = torch.tensor(pseudo_20_values)
sam_index = pseudo_20_index 

sam_y[:,:48] = sam_y[:,:48] / 100  # scale
n = sam_y.shape[0]

# x_recon = []
# sample_x = x_test[3:4].repeat(210000,1)
# sample_y = y_test[3:4].repeat(210000,1)


# kwargs = {'status': sample_y}
# sam_x = vae.sample(num_samples=n, **kwargs)


###### demo 



## strategy 1
# kwargs = {'status': sam_y}
# sam_x = vae.sample(num_samples=n, **kwargs)


# startegy 2


torch.manual_seed(17)

m = 1000000  # 10m 

sub_sam_y = sam_y[:m]
# sub_sam_index = sam_index[:m]
kwargs = {'status': sub_sam_y}
sam_x = vae.sample(num_samples=m, **kwargs)

i = 1

for i in range(1, n // m + 1):
    
    left = i * m 
    
    if i < n // m :
        
        right = (i+1) * m 
    else:
        right = n 
 
    sub_sam_y = sam_y[left:right]
    sub_sam_index = sam_index[left: right]
    

    kwargs = {'status': sub_sam_y}
    sub_sam_x = vae.sample(num_samples=right-left, **kwargs)
    
    
    sam_x = torch.cat((sam_x, sub_sam_x), 0)
    
    print("............ done with:", str(i)+'/'+str(n/m))
    
# sam_y, sam_x, sam_index
    
    
# add returner 
    
sam_x_merge = torch.zeros(sam_x.shape[0],44)
sam_x_merge[:,:11] = sam_x[:,:11]
sam_x_merge[:,11] =  ret_20_values[:,0]
sam_x_merge[:,12:33] = sam_x[:,11:32]
sam_x_merge[:,33] =  ret_20_values[:,1]
sam_x_merge[:,34:44] = sam_x[:,32:42]

s_plays = torch.cat((sam_y[:sam_x.shape[0]], sam_x_merge),1)


s_plays[:,:48] = s_plays[:,:48] * 100
s_plays[:,49:] = s_plays[:,49:] * 100

s_index = sam_index
s_columns = \
    [str(i)+'_x' for i in range(1, 24)] + \
    [str(i)+'_y' for i in range(1, 24)] + \
    ['23_x_land','23_y_land','hang_time'] + \
    [str(i)+'_x_land' for i in range(1, 23)] + \
    [str(i)+'_y_land' for i in range(1, 23)]
    
    
    
# validate 
    
  
football_x = s_plays[:,46]
football_y = s_plays[:,47]

returner_x = s_plays[:,60]
returner_y = s_plays[:,82]

print(s_columns[46],",", s_columns[47],",", s_columns[60],",", s_columns[82])
    
distance = (football_x - returner_x)**2 + (football_y - returner_y)**2
distance = np.sqrt(distance.detach().numpy())

plt.hist(distance, bins=20)
plt.xlabel('distance')
plt.show()
    
    
    
    
    
    
    

 
    
# save simulated plays

s_plays = s_plays.detach().numpy()
s_plays = np.round(s_plays,4)
np.savetxt('/Users/pengwei/Box/bigdatabowl/python_code/s_plays_02.txt',
           s_plays, 
           delimiter=',',
           fmt='%1.2f')

# dict 
s_dict = comet_receive[['nflId','team','jerseyNumber','label','position']]
s_dict.to_csv('/Users/pengwei/Box/bigdatabowl/python_code/s_dict_02.csv')

# columns
textfile = open("/Users/pengwei/Box/bigdatabowl/python_code/s_columns_02.txt", "w")
for element in s_columns:
    textfile.write(element + "\n")
textfile.close()

# index 
textfile = open("/Users/pengwei/Box/bigdatabowl/python_code/s_index_02.txt", "w")
for element in s_index:
    textfile.write(element + "\n")
textfile.close()






# upload file to s3 bucket 
# columns
client.upload_file(
        's_columns_02.txt',
        "elasticbeanstalk-us-east-1-320699877354",
        "nfl-big-data-bowl-2022/python_code/s_columns_02.txt")
# index
client.upload_file(
        's_index_02.txt',
        "elasticbeanstalk-us-east-1-320699877354",
        "nfl-big-data-bowl-2022/python_code/s_index_02.txt")
# dict
client.upload_file(
        's_dict_02.csv',
        "elasticbeanstalk-us-east-1-320699877354",
        "nfl-big-data-bowl-2022/python_code/s_dict_02.csv")
# plays
client.upload_file(
        's_plays_02.txt',
        "elasticbeanstalk-us-east-1-320699877354",
        "nfl-big-data-bowl-2022/python_code/s_plays_02.txt")
    




    
    
    
    
    
    
    
    
    

#################################################################################
# vectorize to visualize 
#################################################################################

    

sam_y = sam_y.detach().numpy()
sam_x = sam_x.detach().numpy()


def h2v(sam_index, sam_y, sam_x):
    
    g_df = pd.DataFrame()

    for i in range(len(sam_y)):
        

        vec_s = sam_y[i]
        vec_e = sam_x[i]
        
        df2 = pd.DataFrame()
    
        df2['x'] =  vec_s[:23].tolist() + vec_e [:22].tolist() + [vec_s[46]]
        df2['y'] =  vec_s[23:46].tolist() + vec_e [22:44].tolist() + [vec_s[47]]
        
        df2['x'] = df2['x'] * 100
        df2['y'] = df2['y'] * 100
        
        
        df2['event'] = ['start'] * 23 + ['end'] * 23 
        df2['hangtime'] = [vec_s[48]]* 46 
        df2['playId'] = sam_index[i]
    
        
        g_df = pd.concat([g_df, df2], axis=0)
        
    return g_df 



sam_x = np.round(sam_x,4)
sam_y = np.round(sam_y,4)


    
pseudo_plays = h2v(sam_index[:100], sam_y[:100], sam_x[:100])

pseudo_plays = pseudo_plays.set_index('playId')

pseudo_plays.loc[pseudo_plays['event']=='end']['x'].hist()
pseudo_plays.loc[pseudo_plays['event']=='end']['y'].hist()


pseudo_plays.to_csv('/Users/pengwei/Box/bigdatabowl/python_code/pseudo_plays.csv')









                    
                    
                        