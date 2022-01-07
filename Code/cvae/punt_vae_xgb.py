#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 31 19:31:37 2021

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
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from collections import defaultdict
from torch.nn import functional as F
import pickle

from c2vae import c2VAE
from c2vae import Dataset

import xgboost as xgb

# from sklearn.model_selection import RepeatedKFold
# from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
# import seaborn as sns 
# import sys 




universe = pd.read_csv('/Users/pengwei/Box/bigdatabowl/python_code/universe.csv')
punts = pd.read_csv('/Users/pengwei/Box/bigdatabowl/python_code/punts.csv')
scout = pd.read_csv('/Users/pengwei/Box/bigdatabowl/python_code/scout.csv')

universe = universe.set_index(['gameId','playId'])
punts = punts.set_index(['gameId','playId'])
scout = scout.set_index(['gameId','playId'])


# check if revert correctly 
universe.loc[(universe['position']=='P') & (universe['keyEvent'] =='snap_timestamp')]['x'].hist()






###############################################################################
###############################################################################
##                                                                           ##
##                                                                           ##
##                                                                           ##
##                                PART 2                                     ##                                                                             
##                                                                           ##
##                                                                           ##
###############################################################################
###############################################################################


###############################################################################
# -1. decide what infomation to put in the model 
###############################################################################


# before play
y1 = punts.loc[:,['yardsToGo','preSnapHomeScore','preSnapVisitorScore']]

# when snap
# y2 = scout.loc[:,['snapDetail','snapTime']] #,'kickDirectionIntended']]
y2 = scout.loc[:,'snapTime'] #,'kickDirectionIntended']]
# punt choice 
# y3 = scout.loc[:,['kickDirectionActual', 'hangTime','kickType']]

location = universe.loc[(universe['keyEvent']=='land_timestamp') & (universe['team']=='football')][['x','y']]
location['x'] = location['x'] / 100
location['y'] = location['y'] / 100 # 53.3

# time that the punter may control 
time_control = pd.DataFrame()
time_control['hold_time'] = scout['operationTime'] - scout['snapTime']
time_control['hangTime'] = scout['hangTime']

y3 = pd.merge(location, time_control, how='left',on=['gameId','playId'])
y3.columns = ['land_x','land_y','hold_time','hang_time'] # varaibles determined by the punter 
     

yy = pd.concat([y1,y2, y3], axis=1)

# the moment when the punter receive the ball 
x_0 = universe.loc[universe['keyEvent']=='receive_timestamp_est']
x_0 = x_0[['x','y','s','a','o','dir']]
x_0['x'] = x_0['x'] / 100 #  scale
x_0['y'] = x_0['y'] / 100 #  scale

x_0['s'] = x_0['s'] / 35 # max s
x_0['a'] = x_0['s'] / 48 # max a
x_0['o'] = x_0['o'] / 360
x_0['dir'] = x_0['dir'] / 360


# the moment when balll lands
x_n = universe.loc[universe['keyEvent']=='land_timestamp']
x_n = x_n[['x','y','s','a','o','dir']]
x_n['x'] = x_n['x'] / 100 # scale
x_n['y'] = x_n['y'] / 100 # scale 

x_n['s'] = x_n['s'] / 35 # max s
x_n['a'] = x_n['s'] / 48 # max a
x_n['o'] = x_n['o'] / 360
x_n['dir'] = x_n['dir'] / 360



x_init = x_0.groupby(['gameId','playId']).agg(lambda x: list(x))
x_end = x_n.groupby(['gameId','playId']).agg(lambda x: list(x))


info_con = pd.merge(x_init, yy, how='left', on=['gameId','playId']) # variables that are contional
info_out = x_end # variables that are observable 





###############################################################################
# 0. input data: vectorize 
###############################################################################


Y = []  # conditional variables 

for i in range(len(info_con)):
        
    # df = info_con.loc[(2018090600, 366)]
    df = info_con.iloc[i]
    # vec = df['x'] + df['y'] \
    # + [df['yardsToGo'], df['preSnapHomeScore'], df['preSnapVisitorScore']] \
    # + [df['land_x'], df['land_y'], df['hold_time'], df['hang_time']]
    # + [df['land_x'], df['land_y'], df['hang_time']]
    vec = df['x'] + df['y'] + [df['land_x'], df['land_y'], df['hang_time']]
    Y.append(vec)
    
    
Y = pd.DataFrame(Y, index=info_con.index)


nan_rows = Y[Y.isnull().any(axis=1)].index # no hangtime 
"""
nan_rows = [(2018092301,  120),
            (2018092302, 3437),
            (2018100711,  253),
            (2018102111, 3651),
            (2019090807, 1293),
            (2019102006, 2301),
            (2020100401,  211),
            (2020110103,  504)]
"""
Y = Y.loc[~Y.index.isin(nan_rows)]

    
 
X = [] # observable variables 
for i in range(len(info_out)):
        
    # df = info_con.loc[(2018090600, 366)]
    df = info_out.iloc[i]
    vec = df['x'][:22] + df['y'][:22]

    X.append(vec)
    
X = pd.DataFrame(X, index=info_out.index)
X = X.loc[~X.index.isin(nan_rows)]



###############################################################################
# 0. splitting train and test 
###############################################################################


X_train, X_test = train_test_split(X, test_size=0.05,random_state=42) # train 

Y_train = Y.loc[Y.index.isin(X_train.index)] # test 
Y_test = Y.loc[Y.index.isin(X_test.index)]


print("X_train:", X_train.shape, "X_test:", X_test.shape)
print("Y_train:", Y_train.shape, "Y_test:", Y_test.shape)


# pair up 
X_train = X_train.sort_index()
Y_train = Y_train.sort_index()

X_test = X_test.sort_index()
Y_test = Y_test.sort_index()



# torch.isnan(x.view(-1)).sum().item()==0
# torch.isnan(y.view(-1)).sum().item()==0



###############################################################################
# 1.0 supervised model 
###############################################################################

# In Y_train: 23 + 23 + 2 + 1
# initial loc of returner: 11, 34 
# initial loc of football: 22, 45 
# final loc of football: 46, 47
# hang time: 48 
# returner x, y, football x, y; land_x, land_y, hangtime

data_train = Y_train.iloc[:,[11, 34, 22, 45, 46, 47, 48]]
data_train.columns = ['returner_x', 'returner_y', 'football_x', 'football_y', 'land_x','land_y','hang_time']

data_train['x_diff'] =  data_train['returner_x'] - data_train['land_x']
data_train['y_diff'] = data_train['returner_y'] - data_train['land_y'] 
data_train['diff'] = np.sqrt(data_train['x_diff'] **2 + data_train['y_diff']**2)

data_train['y_in_down'] = data_train['land_y'] > 0.0
data_train['y_in_up'] = data_train['land_y'] < 0.5
data_train['x_in'] = data_train['land_x'] <  1.1

data_train['x_sign'] = data_train['x_diff'] > 0
data_train['y_sign'] = data_train['y_diff'] > 0

data_train['x_speed'] = data_train['x_diff'] / data_train['hang_time']
data_train['y_speed'] = data_train['y_diff'] / data_train['hang_time']
data_train['speed'] = np.sqrt(data_train['x_speed'] **2 + data_train['y_speed']**2)

data_train['speed'].hist()

# In X_train:
# final loc of returner: 11, 33 
label_train = X_train.iloc[:,[11, 33]]

res_train = label_train.values - data_train[['land_x','land_y']].values



# football - returner 
axis_difference = (data_train[['land_x','land_y']].values - label_train.values) 
distance = np.sqrt(np.sum(axis_difference**2, axis=1)) * 100

plt.hist(distance, bins=20)
# plt.hist(distance, range=(0,2))  
print("............proportion dist < 2: ", sum(distance < 2)/len(distance)) # 0.703




data_test = Y_test.iloc[:,[11,34, 22, 45, 46, 47, 48]]
data_test.columns = ['returner_x','returner_y', 'football_x', 'football_y', 'land_x','land_y','hang_time']

data_test['x_diff'] =  data_test['returner_x'] - data_test['land_x'] 
data_test['y_diff'] =  data_test['returner_y'] - data_test['land_y']
data_test['diff'] = np.sqrt(data_test['x_diff'] **2 + data_test['y_diff']**2)

data_test['y_in_down'] = data_test['land_y'] > 0.0
data_test['y_in_up'] = data_test['land_y'] < 0.5
data_test['x_in'] = data_test['land_x'] < 1.1
data_test['x_sign'] = data_test['x_diff'] > 0
data_test['y_sign'] = data_test['y_diff'] > 0

data_test['x_speed'] = data_test['x_diff'] / data_test['hang_time']
data_test['y_speed'] = data_test['y_diff'] / data_test['hang_time']
data_test['speed'] = np.sqrt(data_test['x_speed'] ** 2 + data_test['y_speed']**2)

data_test['speed'].hist()



label_test = X_test.iloc[:, [11,33]]
res_test = label_test.values - data_test[['land_x','land_y']].values



model_x = xgb.XGBRegressor(
        objective ='reg:squarederror',
        n_estimates=100,
        max_depth=7,
        eta=0.1,
        subsample=0.7,
        colsample_bytree=0.8)

# define model evaluation method
# cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
# scores = cross_val_score(model_x, data_train, label_train.iloc[:,0], scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# force scores to be positive
# scores = np.absolute(scores)
# print('Mean MAE: %.3f (%.3f)' % (scores.mean(), scores.std()) )

model_y = xgb.XGBRegressor(
        objective ='reg:squarederror',
        n_estimates=100,
        max_depth=7,
        eta=0.1,
        subsample=0.7,
        colsample_bytree=0.8)
# define model evaluation method
# cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
# scores = cross_val_score(model_y, data_train, label_train.iloc[:,1], scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# force scores to be positive
# scores = np.absolute(scores)
# print('Mean MAE: %.3f (%.3f)' % (scores.mean(), scores.std()) )

col_x = ['diff', 'speed','x_sign','x_in']
col_y = ['diff', 'speed','y_sign','y_in_down', 'y_in_up']


model_x.fit(data_train[col_x], res_train[:,0])
model_y.fit(data_train[col_y], res_train[:,1])


res_test_hatx = model_x.predict(data_test[col_x])
res_test_haty = model_y.predict(data_test[col_y])


res_test_hat = pd.DataFrame()
res_test_hat['x'] = [round(value,4) for value in res_test_hatx]
res_test_hat['y'] = [round(value,4) for value in res_test_haty]



## model accuracy

print("mse =",np.mean((res_test - res_test_hat)**2))

# football - predicted returner

axis_difference = res_test_hat.values 
distance = np.sqrt(np.sum(axis_difference**2, axis=1)) * 100
plt.hist(distance, bins=20)

# plt.hist(distance, range=(0,2))  
print("............proportion dist < 2: ", round(sum(distance < 2)/len(distance),3)) # 0.833

# save
pickle.dump(model_x, open("model_x.pkl", "wb"))
pickle.dump(model_y, open("model_y.pkl", "wb")) 











###############################################################################
# 2. vae model
###############################################################################

returner_columns = [11, 33]

x_train = X_train.iloc[:,~X_train.columns.isin([11,33])]
x_test = X_test.iloc[:,~X_test.columns.isin([11,33])]


x_train = torch.FloatTensor(x_train.values)
x_test = torch.FloatTensor(x_test.values)

# z = torch.rand(100, latent_dim + status_dim)
y_train = torch.FloatTensor(Y_train.values)
y_test = torch.FloatTensor(Y_test.values)



input_dim = x_train.shape[1] # 2 * 22 =  44
status_dim = y_train.shape[1] # 2 * 23 + 3 = 49 
latent_dim = 15 # 10 #15 # latent dimension = 15
print("............input dimension: X_dim =", input_dim)
print("...........status dimension: Y_dim =", status_dim)
print("...........latent dimension: Z_dim =", latent_dim)




# add more weights on returner 

# 1, 10 gunners, 23, 32
# 11 returner, 33
# 12, 21 vises,34, 43 

weight = torch.ones(42)

"""
    
    gunners = [1, 10, 23, 32]
    vises = [12, 21, 34, 43]
    returner = [11, 33]
    
    for i in gunners + vises:
        weight[i] = 2
        
    for i in returner:
        weight[i] = 4
    
    
    print("........... weight =", weight)
"""


epochs = 10
batch_size = 50
learning_rate = 0.001

params = {'batch_size': batch_size, 
          'shuffle': True,
          'num_workers': 0}

dataset = Dataset(input = x_train, labels = y_train)
data_loader = DataLoader(dataset, **params)


vae = c2VAE(input_dim, latent_dim, status_dim)
optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)


logs = defaultdict(list)
test_loss = []
# train the model 

for epoch in range(epochs):
    
    tracker_epoch = defaultdict(lambda: defaultdict(dict))

    for iteration, (x_b, y_b) in enumerate(data_loader):
        # train
        recon_x_b, x_b, mean, log_var = vae.forward(input = x_b, status = y_b)
        
        args = [recon_x_b, x_b, mean, log_var]
        
        kwargs = {'M_N': batch_size / len(x_train),
                  'weight': weight.repeat(x_b.shape[0],1)
                  }

        loss_components = vae.loss_function(*args, **kwargs)
        loss = loss_components['loss']
        recon_loss = loss_components['Reconstruction_Loss']
        
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        


        logs['loss'].append(loss.item())
        logs['recon_loss'].append(recon_loss.item())
        
    recon_x_test, _, _, _ = vae.forward(input = x_test, status = y_test)
    test_loss.append(F.mse_loss(recon_x_test, x_test))
        

    print("............ epoch = " + str(epoch+1))
   

plt.plot(logs['loss'])
plt.plot(logs['recon_loss'])
plt.ylabel('loss')
plt.xlabel('time')


plt.plot(test_loss)
#  list(vae.parameters())






###############################################################################
# 3. reconstruction 
###############################################################################


# model performance 
vae.eval()
# ValueError: Expected more than 1 value per channel when training, got input size torch.Size([1, 512])


def gather(recon_x, y, recon_returner): # recon_returner
    
    g_df = pd.DataFrame()
    
    for i in range(len(recon_x)):
        
        df_x = recon_x.iloc[i]
        df_y = y.iloc[i]
        df_z = recon_returner.iloc[i]
        
        df2 = pd.DataFrame()
        # 11, 33
        df2['x'] = df_x[:11].tolist() + [df_z['x']] + df_x[11:21].tolist() \
        + [df_y[46]] # football_x
        
        df2['y'] = df_x[21:32].tolist() + [df_z['y']] + df_x[32:42].tolist() \
        + [df_y[47]] # football_y
        
        df2['gameId'] = [df_x.name[0]] * 23
        df2['playId'] = [df_x.name[1]] * 23
        
        g_df = pd.concat([g_df, df2], axis=0)
        
    return g_df 


def reconstruction(x, y, data, x_index):
    # x, y for vae model 
    # data for xgb model
    recon_x, _, _, _ = vae.forward(input=x, status=y)
    
    recon_x = pd.DataFrame(recon_x.detach().numpy()) # pos of players except for returner
    
    recon_x.index = x_index 
    
    dataframe_y = pd.DataFrame(y.detach().numpy()) # pos of football
    
    recon_returner = pd.DataFrame() # pos of returner
    recon_returner['x'] = model_x.predict(data)
    recon_returner['y'] = model_y.predict(data)
    
    
    
    gather_recon_x = gather(recon_x, dataframe_y, recon_returner)
    gather_recon_x = gather_recon_x.set_index(['gameId','playId'])
    
    recon_labels = labels.loc[labels.index.isin(gather_recon_x.index)]
    
    recon_labels = recon_labels.sort_index()
    hat_x  = gather_recon_x.sort_index()
    
    
    # reconstruction positions
    
    hat_x['nflId'] = recon_labels['nflId']
    hat_x['jerseyNumber'] = recon_labels['jerseyNumber']
    hat_x['team'] = recon_labels['team']
    hat_x['displayName'] = recon_labels['displayName']
    hat_x['x'] = hat_x['x'] * 100  # convert back 
    hat_x['y'] = hat_x['y'] * 100  # convert back 
    
    return hat_x


hat_x_test = reconstruction(x_test, y_test, data_test, X_test.index)
hat_x_train = reconstruction(x_train, y_train, data_train, X_train.index)


# validate the reconstucted plays

plt.hist(hat_x_test['x'])
plt.hist(hat_x_train['x'])
plt.hist(universe.loc[universe['keyEvent']=='land_timestamp']['x'])



plt.hist(hat_x_train['y'])
plt.hist(hat_x_test['y'])
plt.hist(universe.loc[universe['keyEvent']=='land_timestamp']['y'])



# result checking: distance between football and the returner 

## testing dataset 

football_location = hat_x_test.loc[hat_x_test['team']=='football'][['x','y']]
returner_location = hat_x_test.iloc[11::23,:][['x','y']]

distance = np.sqrt((football_location['x'] - returner_location['x'])**2 + (football_location['y'] - returner_location['y'])**2)

distance.hist(bins=20)
plt.xlabel('distance reconstructed')
print(distance.median())


"""
## training dataset 
football_location = hat_x_train.loc[hat_x_train['team']=='football'][['x','y']]
returner_location = hat_x_train.iloc[11::23,:][['x','y']]

distance = np.sqrt((football_location['x'] - returner_location['x'])**2 + (football_location['y'] - returner_location['y'])**2)

distance.hist(bins=50)
plt.xlabel('distance reconstructed')
print(distance.median())
"""


## orginal dataset 

football_loc = Y.loc[Y.index.isin(Y_test.index)].iloc[:,46:48]
returner_loc = X.loc[X.index.isin(X_test.index)].iloc[:,[11,33]]

distance = 100 * np.sqrt( 
        (football_loc[46] - returner_loc[11]) **2 + \
         (football_loc[47] - returner_loc[33]) **2
         )

distance.hist(bins= 50)
plt.xlabel('distance original')
print(distance.median())






# save reconstructed positons for visulizing trajectory 

# track2018_20.to_csv("/Users/pengwei/Box/bigdatabowl/python_code/track.csv")

hat_x_test.to_csv('/Users/pengwei/Box/bigdatabowl/python_code/hat_x_test.csv')
hat_x_train.to_csv('/Users/pengwei/Box/bigdatabowl/python_code/hat_x_train.csv')


def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)
        
          
# save
pickle.dump(model_x, open("model_x.pkl", "wb"))
pickle.dump(model_y, open("model_y.pkl", "wb"))                
save_object(vae, '/Users/pengwei/Box/bigdatabowl/python_code/punt_vae.pkl')
# universe.to_csv('/Users/pengwei/Box/bigdatabowl/python_code/universe.csv')
sub_universe = universe.loc[X_test.index]
sub_universe.to_csv('/Users/pengwei/Box/bigdatabowl/python_code/sub_universe.csv')
















###############################################################################
# 3. demo: sampling
###############################################################################

game_id_ = 2018091001 # ,2020122709  #2021010313
play_id_  =  2597 #3288 #195


demo_X = pd.DataFrame([X_test.loc[(game_id_,play_id_ )]] * 10)
demo_Y =  pd.DataFrame([Y_test.loc[(game_id_, play_id_)]] * 10)


data_demo = demo_Y.iloc[:,[11,34, 22, 45, 46, 47, 48]]
data_demo.columns = ['returner_x','returner_y', 'football_x', 'football_y', 'land_x','land_y','hang_time']


data_demo['x_diff'] =  data_demo['returner_x'] - data_demo['land_x'] 
data_demo['y_diff'] =  data_demo['returner_y'] - data_demo['land_y']
data_demo['diff'] = np.sqrt(data_demo['x_diff'] **2 + data_demo['y_diff']**2)


data_demo['y_in_down'] = data_demo['land_y'] > 0.0
data_demo['y_in_up'] = data_demo['land_y'] < 0.5
data_demo['x_in'] = data_demo['land_x'] <  1.1

data_demo['x_sign'] = data_demo['x_diff'] > 0
data_demo['y_sign'] = data_demo['y_diff'] > 0

data_demo['x_speed'] = data_demo['x_diff'] / data_demo['hang_time']
data_demo['y_speed'] = data_demo['y_diff'] / data_demo['hang_time']
data_demo['speed'] = np.sqrt(data_demo['x_speed'] ** 2 + data_demo['y_speed']**2)



# x_demo = demo_X.iloc[:,~demo_X.columns.isin([11,33])]
# x_demo = torch.FloatTensor(x_demo.values)

col_x = ['diff', 'speed','x_sign','x_in']
col_y = ['diff', 'speed','y_sign','y_in_down', 'y_in_up']


y_demo = torch.FloatTensor(demo_Y.values)
# predicted return location 
ret_demo = pd.DataFrame()
ret_demo['x'] = model_x.predict(data_demo[col_x]) + data_demo['land_x'].values
ret_demo['y'] = model_y.predict(data_demo[col_y]) + data_demo['land_y'].values

# returner location 

data_demo[['land_x','land_y']]
ret_demo
demo_X.iloc[:,[11,33]]



np.random.seed(17)

kwargs = {'status': y_demo}
sam_demo = vae.sample(num_samples=10, **kwargs)

## jerseyNumber, team
moon = sub_universe.loc[(game_id_,play_id_ )]
moon = moon.loc[moon['keyEvent'].isin(['snap_timestamp'])][['team','jerseyNumber','nflId']]




# sub_universe.loc[(game_id_,play_id_ )].groupby(['jerseyNumber', 'nflId']).size()

# track2018_20.loc[(game_id_,play_id_ )].groupby(['jerseyNumber', 'nflId']).size()



def h2v(sam_y, sam_x, sam_ret):
    
    g_df = pd.DataFrame()

    for i in range(len(sam_y)):
        
        vec_s = sam_y[i]
        vec_e = sam_x[i]
        ret = sam_ret.iloc[i]
    
        df2 = pd.DataFrame()
    
        df2['x'] =  vec_s[:23].tolist() \
            + vec_e[:11].tolist() \
            + [ret[0]] \
            + vec_e[11:21].tolist() \
            + [vec_s[46].item()]
            
        df2['y'] =  vec_s[23:46].tolist() \
            + vec_e[21:32].tolist()  \
            + [ret[1]] \
            + vec_e[32:42].tolist() \
            + [vec_s[47].item()]
        
        df2['x'] = df2['x'] * 100
        df2['y'] = df2['y'] * 100
        
        df2['event'] = ['start'] * 23 + ['end'] * 23 
        df2['playId'] = [i] * 46
        df2['team'] = moon['team'].tolist() * 2
        df2['jerseyNumber'] = moon['jerseyNumber'].tolist() * 2
        df2['nflId'] =  moon['nflId'].tolist() * 2
    
        g_df = pd.concat([g_df, df2], axis=0)
        
    return g_df 

df = h2v(y_demo, sam_demo, ret_demo)
df = df.set_index('playId')

df['x'] = round(df['x'],2)
df['y'] = round(df['y'],2)


df.to_csv('/Users/pengwei/Box/bigdatabowl/python_code/demo.csv', float_format='%1.3f')


