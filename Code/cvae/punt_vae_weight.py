#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 16:21:31 2021

@author: pengwei
"""


import os 
os.chdir('/Users/pengwei/Box/bigdatabowl/python_code')

import pandas as pd
import matplotlib.pyplot as plt 
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
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
# import seaborn as sns 
# import sys 



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



universe = pd.read_csv('/Users/pengwei/Box/bigdatabowl/python_code/universe.csv')
labels = pd.read_csv('/Users/pengwei/Box/bigdatabowl/python_code/labels.csv')
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
# -1. decide what information to put in the model 
###############################################################################


# before play
y1 = punts.loc[:,['yardsToGo','preSnapHomeScore','preSnapVisitorScore']]

# when snap
# y2 = scout.loc[:,['snapDetail','snapTime']] #,'kickDirectionIntended']]
y2 = scout.loc[:,'snapTime'] #,'kickDirectionIntended']]
# punter choice 
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

# X,Y are paired 



###############################################################################
# 1. model training 
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


x_train = torch.FloatTensor(X_train.values)
x_test = torch.FloatTensor(X_test.values)

# z = torch.rand(100, latent_dim + status_dim)
y_train = torch.FloatTensor(Y_train.values)
y_test = torch.FloatTensor(Y_test.values)

# torch.isnan(x.view(-1)).sum().item()==0
# torch.isnan(y.view(-1)).sum().item()==0



###############################################################################
#  vae model
###############################################################################


input_dim = x_train.shape[1] # 2 * 22 =  44
status_dim = y_train.shape[1] # 2 * 23 + 3 = 49 
latent_dim = 15 # 10 #15 # latent dimension = 15
print("............input dimension: X_dim =",input_dim)
print("...........latent dimension: Y_dim =",status_dim)
print("...........status dimension: Z_dim =",latent_dim)

# Add more weights on returner 

# 1, 10 gunners, 23, 32
# 11 returner, 33
# 12, 21 vises,34, 43 

weight = torch.ones(44)

gunners = [1, 10, 23, 32]
vises = [12, 21, 34, 43]
returner = [11, 33]

for i in gunners + vises:
    weight[i] = 2
    
for i in returner:
    weight[i] = 4
print("........... weight =", weight)



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
        

    print("............ epoch=" + str(epoch+1))
   

plt.plot(logs['loss'])
plt.plot(logs['recon_loss'])
plt.ylabel('loss')
plt.xlabel('time')


plt.plot(test_loss)
#  list(vae.parameters())






###############################################################################
# 2. reconstruction error on test data set
###############################################################################


# model performance 
vae.eval()
# ValueError: Expected more than 1 value per channel when training, got input size torch.Size([1, 512])


def gather(X, Y):
    
    g_df = pd.DataFrame()
    
    for i in range(len(X)):
        
        df_x = X.iloc[i]
        df_y = Y.iloc[i]
        
        df2 = pd.DataFrame()
    
        df2['x'] = df_x[:22].tolist() + [df_y[46]] # football_x
        df2['y'] = df_x[22:44].tolist() + [df_y[47]] # football_y
        df2['gameId'] = [df_x.name[0]] * 23
        df2['playId'] = [df_x.name[1]] * 23
        # if df_x.name[0] == 2021010313 and df_x.name[1] == 3431:
        #    print(df2)
        #    print()
        
        g_df = pd.concat([g_df, df2], axis=0)
        
    return g_df 


def reconstruction(x, y, x_index):
    
    recon_x, _, _, _ = vae.forward(input=x, status=y)
    
    recon_x = pd.DataFrame(recon_x.detach().numpy())
    
    
    recon_x.index = x_index
    
    dataframe_y = pd.DataFrame(y.detach().numpy())
    
    
    gather_recon_x = gather(recon_x, dataframe_y)
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


hat_x_test = reconstruction(x_test, y_test, X_test.index)
hat_x_train = reconstruction(x_train, y_train, X_train.index)




plt.hist(hat_x_test['x'])
plt.hist(hat_x_test['y'])

plt.hist(hat_x_train['x'])
plt.hist(hat_x_train['y'])


plt.hist(universe.loc[universe['keyEvent']=='land_timestamp']['x'])
plt.hist(universe.loc[universe['keyEvent']=='land_timestamp']['y'])



## result checking: distance between football and the returner 

football_location = hat_x_test.loc[hat_x_test['team']=='football'][['x','y']]
returner_location = hat_x_test.iloc[11::23,:][['x','y']]

distance = np.sqrt((football_location['x'] - returner_location['x'])**2 + (football_location['y'] - returner_location['y'])**2)

distance.hist(bins=50)
plt.xlabel('distance reconstructed')
print(distance.median())



"""
football_location = hat_x_train.loc[hat_x_train['team']=='football'][['x','y']]
returner_location = hat_x_train.iloc[11::23,:][['x','y']]

distance = np.sqrt((football_location['x'] - returner_location['x'])**2 + (football_location['y'] - returner_location['y'])**2)

distance.hist(bins=50)
plt.xlabel('distance reconstructed')
print(distance.median())
"""

# 1, 10 gunners, 23, 32
# 11 returner, 33
# 12, 21 vises,34, 43 

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






# save reconstructed positions for visualizing trajectory 

# track2018_20.to_csv("/Users/pengwei/Box/bigdatabowl/python_code/track.csv")
hat_x_test.to_csv('/Users/pengwei/Box/bigdatabowl/python_code/hat_x_test.csv')
hat_x_train.to_csv('/Users/pengwei/Box/bigdatabowl/python_code/hat_x_train.csv')


def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)
        
          
save_object(vae, '/Users/pengwei/Box/bigdatabowl/python_code/bdb_vae.pkl')
universe.to_csv('/Users/pengwei/Box/bigdatabowl/python_code/universe.csv')
sub_universe = universe.loc[X_test.index]
sub_universe.to_csv('/Users/pengwei/Box/bigdatabowl/python_code/sub_universe.csv')









###############################################################################
# 3. demo: sampling
###############################################################################


x_recon = []
sample_x = x_test[3:4].repeat(210000,1)
sample_y = y_test[3:4].repeat(210000,1)
kwargs = {'status': sample_y}

sam = vae.sample(num_samples=210000, **kwargs)


