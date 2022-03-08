# -*- coding: utf-8 -*-
"""
Spectral Regularization

@author: Kaiwen Hou
kaiwen.hou@mila.quebec
"""

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class SpectralRegularization(nn.Module):
        
    def __init__(self, ):
        super(SpectralRegularization, self).__init__()
        
    def generate_affixes(self, nbL = 2, L = 3):
        keys = [[]]
        new_key = [[]]
        for l in range(L):
            new_key = [a+[b] for a in new_key for b in range(nbL)]
            keys.extend(new_key)
        return keys 
    
    def log_prob_word(self, model, word, VOCAB_SIZE):
        '''Calculate the log probability of one word'''
        word_batch = torch.tensor([VOCAB_SIZE-1] + word).unsqueeze(0).long().to(DEVICE)  #add sos and make batch
        #word_batch = torch.tensor(word).unsqueeze(0).long().to(DEVICE)
        output = model(word_batch).squeeze(0)  # len(word)+1 x V
        output = F.log_softmax(output, dim=1)  # len(word)+1 x V 
        output = output[:,:-1]  # len(word) x V  remove last output
        output = output[torch.arange(len(word)), word]  #len(word)
        return output.sum().item() 

    def make_hankel(self, model, prefixes, suffixes, stopProb, VOCAB_SIZE):
        log_hankel = torch.empty(len(prefixes), len(suffixes))
        for i, prefix in enumerate(prefixes):
            for j, suffix in enumerate(suffixes):
                word = list(prefix) + list(suffix)
                log_hankel[i, j] = self.log_prob_word(model, word, VOCAB_SIZE) - (max(len(prefix),len(suffix))-1)*np.log(1-stopProb)
        return log_hankel

    def forward(self, model, VOCAB_SIZE, stopProb=0.2, hankelSizeCap=10):
        τ = min(np.random.geometric(stopProb), hankelSizeCap)
        prefixes = self.generate_affixes(L=τ)
        suffixes = self.generate_affixes(L=τ)
        logH = self.make_hankel(model, prefixes, suffixes, stopProb, VOCAB_SIZE)
        return torch.norm(torch.exp(logH - logH.max()), p='nuc') #Hankel Loss