# -*- coding: utf-8 -*-
"""
Character Language Model

@author: Kaiwen Hou
kaiwen.hou@mila.quebec
"""

import torch.nn as nn

from torch.nn import functional as F
import torch
from torch.distributions import Categorical

class CharLanguageModel(nn.Module):

    def __init__(self, vocab_size, embed_size, hidden_size, nlayers, rnn_type='RNN', nonlinearity='tanh'):
        super(CharLanguageModel, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.nlayers = nlayers

        # layers
        if self.embed_size > 0:
            self.embedding = nn.Embedding(vocab_size, embed_size)   # Embedding layer
        else:
            self.embedding = lambda x: F.one_hot(x,num_classes=vocab_size).float()
            self.embed_size = vocab_size
        
        if rnn_type == 'RNN':
            self.rnn = nn.RNN(input_size = self.embed_size, hidden_size=hidden_size, num_layers=nlayers, dropout=0, batch_first=True, nonlinearity=nonlinearity) # Recurrent network
        elif rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_size = self.embed_size, hidden_size=hidden_size, num_layers=nlayers, dropout=0, batch_first=True)
        elif rnn_type  == 'GRU':
            self.rnn = nn.GRU(input_size = self.embed_size, hidden_size=hidden_size, num_layers=nlayers, dropout=0, batch_first=True)
        else:
            raise ValueError("RNN, LSTM & GRU only")
        # self.dense = nn.Linear(hidden_size, dense_size)
        

            
        self.scoring = nn.Linear(hidden_size, vocab_size)  # Projection layer to pdf

    def forward(self, seq_batch):  # N x L
        # returns 3D logits
        batch_size = seq_batch.shape[0]
        # print("seq batch", seq_batch.shape)
        # print(seq_batch)
        embed = self.embedding(seq_batch)  # N x L x E
        # print("embed", embed.shape)
        # print(embed)

        hidden = None
        output_rnn, hidden = self.rnn(embed, hidden)  # N x L x H
        
        output_rnn_flatten = output_rnn.contiguous().view(-1, self.hidden_size)  # (N*L) x H
        # output_dense = self.dense(output_rnn_flatten)  # (N*L) x D
        # output_dense = F.relu(output_dense, inplace=True)
        output_flatten = self.scoring(output_rnn_flatten)  # (N*L) x V
        return output_flatten.view(batch_size, -1, self.vocab_size)  # N x L x V

    def sample_word(self, max_length=1e10):
        symb = torch.tensor(self.vocab_size-2).reshape([1,1]) # start of seq
        hidden = None
        out = []
        while True:
            embed = self.embedding(symb.reshape([1,1])) if self.embed_size > 0 else symb
            output_rnn, hidden = self.rnn(embed, hidden)
            logits = self.scoring(output_rnn).squeeze()
            # print(logits)
            probas = F.softmax(logits,dim=0)
            #print(probas)

            # print(probas)
            symb = Categorical(probas).sample()
            # print(symb)
            if symb == self.vocab_size-1:
                return "".join(out)

            out.append(str(symb.item()))
            if len(out) == max_length:
                return "".join(out)