#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 25 15:34:47 2021

@author: pengwei
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 21:14:46 2021

@author: pengwei

c2vae is modified `from https://github.com/AntixK/PyTorch-VAE/

"""

import os 
os.chdir('/Users/pengwei/Box/bigdatabowl/python_code')
import torch
from torch import nn
from torch.nn import functional as F
from typing import List, TypeVar
# from torch import tensor as Tensor
Tensor = TypeVar('torch.tensor')
from torch.utils.data import DataLoader
from collections import defaultdict
import matplotlib.pyplot as plt 
import numpy as np
from mse_loss import mse_loss
from mse_loss import MSELoss


class Dataset(torch.utils.data.Dataset):
    
  'Characterizes a dataset for PyTorch'
  
  def __init__(self, input, labels):
        'Initialization'
        self.labels = labels
        self.input = input

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.labels)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        # ID = self.list_IDs[index]

        # Load data and get label
        
        # X = torch.load('data/' + ID + '.pt')
        # X = self.input[ID]
        # y = self.labels[ID]
        X = self.input[index]
        y = self.labels[index]

        return X, y

"""

## test class Dataset

params = {'batch_size': 2,
          'shuffle': True,
          'num_workers': 0}


input_dim = 6
latent_dim = 3
status_dim = 4

x = torch.rand(10, input_dim)
z = torch.rand(10, latent_dim + status_dim)
y = torch.rand(10, status_dim)
    
training_set = Dataset(input=x, labels=y)
training_generator = torch.utils.data.DataLoader(training_set, **params)

for local_batch, local_labels in training_generator:
    # Transfer to GPU
    X, y = local_batch, local_labels
    print(X,y)
    
"""



class c2VAE(nn.Module):

    def __init__(self,
                 input_dim: int,
                 latent_dim: int, 
                 status_dim: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(c2VAE, self).__init__() 

        self.latent_dim = latent_dim
    
        modules = []
        if hidden_dims is None: # list of hidden dimensions
            hidden_dims = [512, 256, 128]

        # Build Encoder: x -> z
        in_dim = input_dim + status_dim
        
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_features=in_dim, out_features=h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.LeakyReLU())
            )
            in_dim = h_dim

        self.encoder = nn.Sequential(*modules)
        
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)


        # Build Decoder: z -> x 
        modules = []

        self.decoder_input = nn.Linear(latent_dim + status_dim, hidden_dims[-1])

        hidden_dims.reverse()  # same structure 

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i],
                              hidden_dims[i + 1]),
                    nn.BatchNorm1d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )



        self.decoder = nn.Sequential(*modules)
        
        # Final Layer
        self.final_layer = nn.Sequential(
                            nn.Linear(in_features=hidden_dims[-1], out_features=input_dim),
                            nn.Tanh()
                            )


    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        # result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        
        result = self.decoder_input(z)
        # result = result.view(-1, 64, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor: # sample z 
        """
        Will a single z be enough to compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        
        return eps * std + mu

    ## 1.0
    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        
        y = kwargs['status'].float()
        # embedded_class = self.embed_class(y)
        # embedded_class = embedded_class.view(-1, self.img_size,     
        # self.img_size).unsqueeze(1)
        # embedded_input = self.embed_data(input)
        x = torch.cat([input, y], dim = 1)
        mu, log_var = self.encode(x)

        z = self.reparameterize(mu, log_var) # sample one z 

        z = torch.cat([z, y], dim = 1) # z + y 
        
        return  [self.decode(z), input, mu, log_var]

    
    ## 2.0 
    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset
        
        weight = kwargs['weight']
        
        recons_loss  = torch.mean(F.mse_loss(recons, input, reduction='none') * weight)
    
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 \
                                               - log_var.exp(), dim = 1), dim = 0)
    
        loss = recons_loss + kld_weight * kld_loss
        
        
        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':-kld_loss}

    def sample(self,
               num_samples:int,
               # current_device: int,
               **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        y = kwargs['status'].float()
        z = torch.randn(num_samples,
                        self.latent_dim)

        # z = z.to(current_device)

        z = torch.cat([z, y], dim=1)
        samples = self.decode(z)
        
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x, **kwargs)[0]
    

    
    
if __name__ == "__main__":
    
    input_dim = 4 * 23
    status_dim = 4 * 23 + 10 
    latent_dim = 10
    
    x = torch.rand(100, input_dim)
    # z = torch.rand(100, latent_dim + status_dim)
    y = torch.rand(100, status_dim)
    
    params = {'batch_size': 20,
          'shuffle': True,
          'num_workers': 0}
    
    dataset = Dataset(input=x, labels=y)
    data_loader = DataLoader(dataset, **params)
    
 
    epochs = 20
    batch_size = 20
    learning_rate = 0.01
    
    vae = c2VAE(input_dim, latent_dim, status_dim)
    optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)


    
    logs = defaultdict(list)
    # train the model 
    for epoch in range(epochs):
        
        tracker_epoch = defaultdict(lambda: defaultdict(dict))

        for iteration, (x_b, y_b) in enumerate(data_loader):

            recon_x_b, x_b, mean, log_var = vae.forward(input=x_b, status=y_b)
            
            args = [recon_x_b, x_b, mean, log_var]
         
            weight = torch.ones(x_b.shape[0], x_b.shape[1])
            
            
            kwargs={'M_N': batch_size / len(x),
                    'weight': weight
                    }

            loss_components = vae.loss_function(*args, **kwargs)
            
            loss = loss_components['loss']
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logs['loss'].append(loss.item())
            
        print("............ epoch="+str(epoch))
   
    
    plt.plot(logs['loss'])
    