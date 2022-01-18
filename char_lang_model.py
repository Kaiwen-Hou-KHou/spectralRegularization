# -*- coding: utf-8 -*-
"""
Character Language Model

@author: Kaiwen Hou
kaiwen.hou@mila.quebec
"""

import torch.nn as nn

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
        
            if rnn_type == 'RNN':
                self.rnn = nn.RNN(input_size = embed_size, hidden_size=hidden_size, num_layers=nlayers, dropout=0, batch_first=True, nonlinearity=nonlinearity) # Recurrent network
            elif rnn_type == 'LSTM':
                self.rnn = nn.LSTM(input_size = embed_size, hidden_size=hidden_size, num_layers=nlayers, dropout=0, batch_first=True)
            elif rnn_type  == 'GRU':
                self.rnn = nn.GRU(input_size = embed_size, hidden_size=hidden_size, num_layers=nlayers, dropout=0, batch_first=True)
            else:
                raise ValueError("RNN, LSTM & GRU only")
            # self.dense = nn.Linear(hidden_size, dense_size)
            
        else:
            
            if rnn_type == 'RNN':
                self.rnn = nn.RNN(input_size = 1, hidden_size=hidden_size, num_layers=nlayers, dropout=0, batch_first=True, nonlinearity=nonlinearity) # Recurrent network
            elif rnn_type == 'LSTM':
                self.rnn = nn.LSTM(input_size = 1, hidden_size=hidden_size, num_layers=nlayers, dropout=0, batch_first=True)
            elif rnn_type  == 'GRU':
                self.rnn = nn.GRU(input_size = 1, hidden_size=hidden_size, num_layers=nlayers, dropout=0, batch_first=True)
            else:
                raise ValueError("RNN, LSTM & GRU only")
            # self.dense = nn.Linear(hidden_size, dense_size)
            
        self.scoring = nn.Linear(hidden_size, vocab_size)  # Projection layer to pdf

    def forward(self, seq_batch):  # N x L
        # returns 3D logits
        batch_size = seq_batch.shape[0]
        if self.embed_size > 0:
            embed = self.embedding(seq_batch)  # N x L x E
        else:
            embed = seq_batch.unsqueeze(-1) # N x L x 1
        hidden = None
        output_rnn, hidden = self.rnn(embed, hidden)  # N x L x H
        
        output_rnn_flatten = output_rnn.contiguous().view(-1, self.hidden_size)  # (N*L) x H
        # output_dense = self.dense(output_rnn_flatten)  # (N*L) x D
        # output_dense = F.relu(output_dense, inplace=True)
        output_flatten = self.scoring(output_rnn_flatten)  # (N*L) x V
        return output_flatten.view(batch_size, -1, self.vocab_size)  # N x L x V
