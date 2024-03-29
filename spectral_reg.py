# -*- coding: utf-8 -*-
"""
Spectral Regularization
"""

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import itertools

from tictoc import tic,toc
import wandb


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#DEVICE = "cpu"

words_of_length = lambda length,voc_size : torch.tensor(list(itertools.product(range(voc_size), repeat=length)))
all_words_of_length = []

class SpectralRegularization(nn.Module):  
    def __init__(self, ):
        super(SpectralRegularization, self).__init__()

    # @profile
    def forward(self, model, VOCAB_SIZE, stopProb=0.2, hankelSizeCap=10, russian_roulette_type='L_shape', verbose=-1, use_wandb=False):
        tau = min(np.random.geometric(stopProb), hankelSizeCap)   
        if use_wandb:
            wandb.log({"tau":tau})
        tic()

        logH = self.make_hankel(model, tau, stopProb, VOCAB_SIZE, russian_roulette_type)
        if verbose > 0:
            print(f"running time for Hankel of size {logH.shape} (tau = {tau}, {russian_roulette_type}): {toc()} seconds")

        out = torch.norm(torch.exp(logH - logH.max()), p='nuc') #Hankel Loss...
        return out


    # @profile
    def get_Hankel_tensor(self,model, length, VOCAB_SIZE):
        global all_words_of_length
        if len(all_words_of_length) < length+1:
            all_words_of_length = [words_of_length(L, VOCAB_SIZE-2) for L in range(length+1)]

        words = all_words_of_length[length]
        BOS_vec = (VOCAB_SIZE-2)*torch.ones(len(words),1)
        EOS_vec = (VOCAB_SIZE-1)*torch.ones(len(words),1)
        words = torch.cat((BOS_vec,words,EOS_vec),1).long().to(DEVICE)
        vals = F.log_softmax(model(words),dim=2)
        words = words[:,1:]
        vals = torch.gather(vals,2,words.unsqueeze(-1)).squeeze(-1)
        vals = vals.sum(dim=1).reshape([VOCAB_SIZE-2] * length)
        return vals

    # @profile
    def make_hankel(self, model, tau, stopProb, VOCAB_SIZE, russian_roulette_type='L_shape'):
        if not (russian_roulette_type in "L_shape block_diag block_diag_no_norm".split()):
            raise(NotImplementedError())

        #model_values = {}
        max_length = 2*tau
        hankel_tensors = []

        for l in range(max_length+1):
            if russian_roulette_type == 'L_shape':
                hankel_tensors.append(self.get_Hankel_tensor(model,l,VOCAB_SIZE))
            if 'block_diag' in russian_roulette_type:
                if l > tau:
                    hankel_tensors.append(-1 * torch.ones([VOCAB_SIZE-2] * l).to(DEVICE) * torch.inf)
                else:
                    hankel_tensors.append(self.get_Hankel_tensor(model,l,VOCAB_SIZE))
                    if russian_roulette_type == 'block_diag': # else no normalization, i.e. block_diag_no_norm
                        hankel_tensors[-1] -= l*np.log(1-stopProb)

        logH = hankel_tensors[0].reshape([1,1])
        n_letters = VOCAB_SIZE-2

        for t in range(1,tau+1):
            new_row_block = torch.cat([H.reshape([n_letters**t,-1]) for H in hankel_tensors[t:2*t]],dim=1) 
            new_col_block = torch.cat([H.reshape([-1,n_letters**t]) for H in hankel_tensors[t:2*t+1]],dim=0) 
            if russian_roulette_type == 'L_shape':
                new_row_block -= t*np.log(1-stopProb)
                new_col_block -= t*np.log(1-stopProb)
            logH = torch.cat([logH,new_row_block], dim=0)
            logH = torch.cat([logH,new_col_block], dim=1)
        return logH






    # # @profile
    # def get_Hankel_tensor_batchified(self,model, length, VOCAB_SIZE, batch_size=10000):
    #     global all_words_of_length
    #     if len(all_words_of_length) < length+1:
    #         all_words_of_length = [words_of_length(L, VOCAB_SIZE-2) for L in range(length+1)]

    #     words = all_words_of_length[length]
    #     BOS_vec = (VOCAB_SIZE-2)*torch.ones(len(words),1)
    #     EOS_vec = (VOCAB_SIZE-1)*torch.ones(len(words),1)
    #     words = torch.cat((BOS_vec,words,EOS_vec),1).long().to(DEVICE)
    #     vals = None
    #     # print("length:", length, "words shape:", words.shape, "batch size:",batch_size)

    #     for words_batch in torch.split(words, batch_size):
    #         # print("batch shape:", words_batch.shape)
    #         vals_batch = F.log_softmax(model(words_batch),dim=2)
    #         words_batch = words_batch[:,1:]
    #         vals_batch = torch.gather(vals_batch,2,words_batch.unsqueeze(-1)).squeeze(-1)
    #         vals_batch = vals_batch.sum(dim=1)
    #         vals = torch.cat((vals,vals_batch)) if vals is not None else vals_batch
    #     return vals.reshape([VOCAB_SIZE-2] * length)
