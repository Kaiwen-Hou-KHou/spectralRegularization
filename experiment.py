# -*- coding: utf-8 -*-
"""
Experiments

@author: Kaiwen Hou
kaiwen.hou@mila.quebec
"""

import numpy as np
import pandas as pd
import random
from toy_datasets import *
from simple_datasets import *
from char_lang_model import CharLanguageModel
from early_stop import EarlyStopping
from train_loop import *
from plot_loss import *
from tqdm import tqdm
from itertools import repeat

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE

def pad_data(dataset, max_len):
    split_str = [list(np.array(list(i)).astype(int)) for i in dataset]
    return [l + [3] * (max_len - len(l)) for l in split_str]

data_split = 0.99999999
rng_key = jax.random.PRNGKey(12345)

# Fixing hyperparams
def sample_experiment(tomita_num, train_len=15, test_len_list=[15, 17, 20, 25],
               λ_list = [0, 0.1, 0.2, 0.4, 0.8, 1.6],
               N_list = [100, 200, 400, 800, 1600],
               repeat_times = 5):

    # generate test data
    data_dict = {}
    test_dataset_dict = {}
    test_loader_dict = {}
    
    for max_len in tqdm(test_len_list):
        dataset = tomita_dataset(rng_key, data_split, max_len, 
                                         tomita_num=tomita_num, min_len=max_len)[0] # fix length on test dataset
        data_dict[max_len] = pad_data(dataset, max_len)
        dataForTesting = np.array(random.choices(data_dict[max_len],k=2000))
        test_data = torch.tensor(dataForTesting)
        test_dataset_dict[max_len] = SimpleDataset(test_data)
        test_loader_dict[max_len] = DataLoader(test_dataset_dict[max_len], shuffle=False, batch_size=len(test_data), collate_fn = collate, drop_last=True)
        
    # generate training data
    dataset = tomita_dataset(rng_key, data_split, train_len, tomita_num=tomita_num)[0] 
    data = pad_data(dataset, train_len)
    train_loader, val_loader, VOCAB_SIZE = get_random_training_data(data, 2000, split=0.8)
    
    # storing results
    min_train_acc = pd.DataFrame(index=N_list, 
                 columns=pd.MultiIndex.from_tuples(zip([x for item in list(range(repeat_times)) for x in repeat(item, len(λ_list))],
                                                       λ_list*repeat_times)))
    min_val_acc = copy.deepcopy(min_train_acc)
    min_test_acc = copy.deepcopy(min_train_acc)
    models = copy.deepcopy(min_train_acc)
    
    min_train_acc_ignore_markers = pd.DataFrame(index=N_list, 
                 columns=pd.MultiIndex.from_tuples(zip([x for item in list(range(repeat_times)) for x in repeat(item, len(λ_list))],
                                                       λ_list*repeat_times)))
    min_val_acc_ignore_markers = copy.deepcopy(min_train_acc_ignore_markers)
    min_test_acc_ignore_markers = copy.deepcopy(min_train_acc_ignore_markers)
    
    min_train_loss = pd.DataFrame(index=N_list, 
                 columns=pd.MultiIndex.from_tuples(zip([x for item in list(range(repeat_times)) for x in repeat(item, len(λ_list))],
                                                       λ_list*repeat_times)))
    min_val_loss = copy.deepcopy(min_train_loss)
    min_test_loss = copy.deepcopy(min_train_loss)
        
    
    # experiment
    for N in N_list:
        for ii in tqdm(range(repeat_times)):
            train_loader, val_loader, VOCAB_SIZE = get_random_training_data(data, N, split=0.8)
        
            for λ in tqdm(λ_list):
                model = CharLanguageModel(vocab_size = VOCAB_SIZE, embed_size = VOCAB_SIZE, hidden_size=50, nlayers=1, rnn_type='RNN', 
                              nonlinearity='tanh').to(DEVICE)
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-1, amsgrad=True)
                early_stopping = EarlyStopping(patience=10, verbose=True)
                stoppingEpoch, losses, accs, accs_ignore_markers, best_model = train_model(model, VOCAB_SIZE, optimizer, train_loader, val_loader, 
                                                                               list(test_loader_dict.values()), λ=λ, 
                                                                               early_stopping=early_stopping, 
                                                                               n_epoch=100)
                
                train_loss, val_loss, test_loss_list = losses
                train_acc, val_acc, test_acc_list = accs
                train_acc_ignore_markers, val_acc_ignore_markers, test_acc_ignore_markers_list = accs_ignore_markers
                
                # storing model
                models.loc[N][ii,λ] = best_model
                
                # storing accuracy
                min_train_acc.loc[N][ii,λ] = train_acc[stoppingEpoch]
                min_val_acc.loc[N][ii,λ] = val_acc[stoppingEpoch]
                min_test_acc.loc[N][ii,λ] = test_acc_list[stoppingEpoch]
                display(min_train_acc, min_val_acc)
                
                # storing accuracy ignoring markers
                min_train_acc_ignore_markers.loc[N][ii,λ] = train_acc_ignore_markers[stoppingEpoch]
                min_val_acc_ignore_markers.loc[N][ii,λ] = val_acc_ignore_markers[stoppingEpoch]
                min_test_acc_ignore_markers.loc[N][ii,λ] = test_acc_ignore_markers_list[stoppingEpoch]
                display(min_train_acc_ignore_markers, min_val_acc_ignore_markers)
                
                # storing loss
                min_train_loss.loc[N][ii,λ] = train_loss[stoppingEpoch]
                min_val_loss.loc[N][ii,λ] = val_loss[stoppingEpoch]
                min_test_loss.loc[N][ii,λ] = test_loss_list[stoppingEpoch]
                display(min_train_loss, min_val_loss)
                
    return [models, [min_train_acc, min_val_acc, min_test_acc],
            [min_train_acc_ignore_markers, min_val_acc_ignore_markers, min_test_acc_ignore_markers],
            [min_train_loss, min_val_loss, min_test_loss]]