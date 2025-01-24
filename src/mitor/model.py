# -*- coding: utf-8 -*-

from __future__ import print_function
import torch
from scipy.optimize import nnls

import torch.nn
from torch.nn import init
from torch.nn.parameter import Parameter
import math
import numpy as np
import os
import logging


class GammaPoissonModel(object):

    def __init__(self, K=3,
                 learning_rate=1e-2,
                 weight_decay = 1e-4,
                 device='cpu',
                 prefix='test'):
        self.K = K  # number of distributions
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.prefix = prefix
        self.log_fn = os.path.join(self.prefix+'.log')
        logging.basicConfig(filename=self.log_fn, level=logging.INFO)

    def loss_fun(self, k, x_i, x):
        u = self.U
        b = self.B

        loss = torch.tensor(0, dtype=torch.float, device=self.device)
        
        u = self.U[k]
        b = self.B[k]
            
        eps = 1e-10  # Small constant to avoid log(0)
        u_b = u * b
        b_clamped = torch.clamp(b, min=eps)
        u_b_clamped = torch.clamp(u_b, min=eps)
        
        fun1 = (u_b) * torch.log(b_clamped)
        fun2 = -torch.special.gammaln(u_b_clamped)
        fun3 = torch.special.gammaln(x + u_b_clamped)
        fun4 = -(x + u_b_clamped) * torch.log(1 + b_clamped)
        
        return -torch.sum(fun1+fun2+fun3+fun4)
    
    def fit(self, data_list, epoch=100, delta=0.0001,batch_size = 32, verbose=True):
        mean = []
        for data in data_list:
            mean.append(data.mean(axis=0))
        self.mean = np.array(mean)
        self.K = len(data_list)
        self.N = data_list[0].shape[1]
        self.U = torch.tensor(self.mean, dtype=torch.float,  device=self.device, requires_grad=True)
        self.B = torch.ones([self.K, self.N], dtype=torch.float,  device=self.device, requires_grad=True)


        self.optimizer = torch.optim.Adadelta([self.U, self.B],
                                          lr=self.learning_rate,
                                          weight_decay=self.weight_decay)  # append L2 penalty

        first_loss = 0
        current_loss = 2
        prev_loss = 1
        for i in range(epoch):
            if current_loss - prev_loss > 0 and current_loss - prev_loss < delta:
                break
            for _, batch in enumerate(self.data_loader(data_list, batch_size)):
                k, x_i, x = batch

                loss = self.loss_fun(k, x_i, x)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                current_loss = loss.item()
                info = '{}\t{}\t{}\t{}'.format(i, k, x_i, current_loss)
                logging.info(info)
            if verbose:
                print(info)
                logging.info(info)
            prev_loss = current_loss
        return 


    def data_loader(self, data_list,batch_size=32,  task='fit'):
        for k, data in enumerate(data_list):
            data = torch.tensor(data, dtype=torch.float, device=self.device)
            M, N = data.shape
            for i in range(0, M, batch_size):
                X = data[i:i + batch_size, :]
                yield k, i, X


class GammaPoissonMixModel(torch.nn.Module):
    
    def __init__(self, U, B,
                 learning_rate=1e-2,
                 device='cpu',
                 prefix='test'):
        self.device = device
        self.K = U.shape[0]
           
        self.U = torch.tensor(U, device=self.device)
        self.B = torch.tensor(B, device=self.device)
        self.learning_rate = learning_rate


        self.prefix = prefix

        self.log_fn = os.path.join(self.prefix+'.log')
        logging.basicConfig(filename=self.log_fn, level=logging.INFO)

    def loss_fun(self, x_i, x, theta):
        u = self.U
        b = self.B
 
        loss1 = torch.tensor(0, dtype=torch.float, device=self.device)
        loss2 = torch.tensor(0, dtype=torch.float, device=self.device)
        for k in range(self.K):
            u = self.U[k]
            b = self.B[k]
            f = self.F[k, x_i]

            eps = 1e-10  # Small constant to avoid log(0)
            u_b = u * b
            b_clamped = torch.clamp(b, min=eps)
            u_b_clamped = torch.clamp(u_b, min=eps)

            f = torch.sigmoid(f)

            fun1 = (u_b_clamped ) * torch.log(b_clamped)
            fun2 = - torch.special.gammaln(u_b_clamped )
            fun3 = torch.special.gammaln(x + u_b_clamped )
            fun4 = -(x + (u_b_clamped ))*torch.log(1 + b_clamped)
            loss1 += torch.sum(fun1 + fun2 + fun3 + fun4)*f             
            loss2 += f
        loss1 = -loss1
        loss2 = torch.abs(loss2 - 1)*theta
        return loss1, loss2
    
    
    def loss_F(self, x_i, theta):
        loss = (self.F[:, x_i].sum() - 1)*theta
        return loss
    
    def fit(self, data, epoch=10, delta=0.0001, theta=1, batch_size=32, verbose=True):

        self.M = data.shape[0]
        
        F = NNLSModel(self.U).fit(data).T
        self.F = torch.tensor(F, dtype=torch.float,  device=self.device, requires_grad=True)

        self.optimizer = torch.optim.Adadelta([self.F], 
                                          lr=self.learning_rate)
                                          

        current_loss = 2
        prev_loss = 1
        info = ''
        for _, batch in enumerate(self.data_loader(data, batch_size=batch_size)):
            for i in range(epoch):
                if current_loss - prev_loss > 0 and current_loss - prev_loss < delta:
                    break
            
                x_i, x = batch
                
                loss1, loss2 = self.loss_fun(x_i, x, theta) 
                loss = loss1 + loss2
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                current_loss = loss.item()
                info = '{}\t{}\t{}\t{}\t{}'.format(x_i, i, current_loss, loss1.item(), loss2.item())
                logging.info(info)
            
                if i % 100 == 0 and verbose:                
                    print(info)
            prev_loss = current_loss
            logging.info(info)
        

        return 


    def data_loader(self, data, batch_size=32, task='fit'):
        data = torch.tensor(data, dtype=torch.float, device=self.device)
        M, N = data.shape
        for i in range(0, M, batch_size):
            X = data[i:i + batch_size, :]
            yield i, X
            

class NNLSModel(torch.nn.Module):
    
    def __init__(self, U, prefix='test'):
        self.U = U.T
        self.prefix = prefix

        self.log_fn = os.path.join(self.prefix+'.log')
        logging.basicConfig(filename=self.log_fn, level=logging.INFO)

    def fit(self, data):

        F = []
        error = []
        for i in range(data.shape[0]):
            y = data[i,:]
            f, e = nnls(self.U,y)
    

            f = np.reshape(f, (1, f.shape[0]))            
            F.append(f)
            error.append(e)
        F = np.concatenate(F, axis=0)    
        self.F = F
        return F   