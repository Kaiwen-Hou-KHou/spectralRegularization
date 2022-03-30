# -*- coding: utf-8 -*-
"""
Dataset Loaders
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import random

def load_simple_data(data_path):
    return torch.from_numpy(np.load(data_path)).long()

# Dataset class
class SimpleDataset(Dataset):
    ''' Transform toydata to dataset
    '''
    def __init__(self, simple_data):
        self.data = simple_data  # N x seq_len
        # VOC SIZE IS HARD CODED !!!!!
        self.sos_token = 2#simple_data.max() +1  # SOS token    
        self.eos_token = 3#simple_data.max() +2  # EOS token

    def __getitem__(self, i):
        seq = self.data[i]
        seq = torch.cat((torch.tensor([self.sos_token]), seq, torch.tensor([self.eos_token])))  # add SOS, EOS token
        return seq[:-1], seq[1:]  # labels are the input sequence shifted by 1
    
    def __len__(self):
        return self.data.shape[0]


def pad_data(dataset, split_method='any', remove_header=True):
    max_len = max([len(s) for s in dataset])
    if split_method == 'any':
        split_str = [list(np.array(list(i)).astype(int)) for i in dataset]
        VOCAB_SIZE = max([max(i) for i in split_str]) - min([min(i) for i in split_str]) + 1 + 2 
        # +1 due to inclusiveness; +2 due to SOS and EOS
        
    elif split_method == 'space':
        split_str = [i.split() for i in dataset]
        split_str.remove([])
        if remove_header:
            print('Header of dataset: '+str(split_str[0]))
            split_str = [list(np.array(i).astype(int)) for i in split_str[1:]]
            print('Length of dataset: '+str(len(split_str)))
            VOCAB_SIZE = max([max(i) for i in split_str]) - min([min(i) for i in split_str]) + 1 + 2
        else:
            split_str = [list(np.array(i).astype(int)) for i in split_str]
            print('Length of dataset: '+str(len(split_str)))
            VOCAB_SIZE = max([max(i) for i in split_str]) - min([min(i) for i in split_str]) + 1 + 2
        
    else:
        raise ValueError('Unsupported split method')
    
    EOS = VOCAB_SIZE - 1
    
    return [ll + [EOS] * (max_len - len(ll)) for ll in split_str]



def collate(seq_list):
    '''Transform a list of sequences into a batch
    Returns data of the format batch_size x seq_len'''
    inputs = torch.stack([s[0] for s in seq_list], dim=0)
    targets = torch.stack([s[1] for s in seq_list], dim=0)
    return inputs, targets 

def get_data_split(data,train_len,val_len,test_len,batch_size=128,overlap=False):
    VOCAB_SIZE = len(set("".join(data))) + 2 # add BOS and EOS to the alphabet
    data = pad_data(data, VOCAB_SIZE=VOCAB_SIZE)
    random.shuffle(data)
    if overlap:
        test,train,val = [torch.tensor(random.choices(data,k=n)) for n in [test_len,train_len,val_len]]
    else:
        print(f"{train_len}+{val_len}+{test_len} --- {len(data)}")
        assert(train_len+val_len+test_len <= len(data)), f"{train_len}+{val_len}+{test_len} > {len(data)}"
        val,data = torch.tensor(data[:val_len]),data[val_len:]
        train,data = torch.tensor(data[:train_len]),data[train_len:]
        test,data = torch.tensor(data[-test_len:]),data[:test_len]
    datasets = []
    datasets.append(DataLoader(SimpleDataset(train), shuffle=True, batch_size=batch_size, collate_fn = collate, drop_last=True) if train_len > 0 else None) 
    datasets.append(DataLoader(SimpleDataset(val), shuffle=False, batch_size=len(val), collate_fn = collate) if val_len > 0 else None)
    datasets.append(DataLoader(SimpleDataset(test), shuffle=False, batch_size=len(test), collate_fn = collate) if test_len > 0 else None)

    return datasets








def get_first_N_training_data(data, N, split=0.7):
    
    simple_data = torch.tensor(data[:int(N*split)])
    val_data = torch.tensor(data[int(N*split):int(N)])
    VOCAB_SIZE = 4

    train_dataset = SimpleDataset(simple_data)
    val_dataset = SimpleDataset(val_data)

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=len(simple_data), collate_fn = collate)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=len(val_data), collate_fn = collate, drop_last=True)
    
    return [train_loader, val_loader, VOCAB_SIZE]


def get_random_training_data(data, N, split=0.7, batch_size=128):
    import random
    simple_data = torch.tensor(random.choices(data,k=int(N*split)))
    val_data = torch.tensor(random.choices(data,k=int(N*(1-split))))   

    VOCAB_SIZE = 4  # hardcoded !!!

    train_dataset = SimpleDataset(simple_data)
    val_dataset = SimpleDataset(val_data)

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, collate_fn = collate)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=len(val_data), collate_fn = collate, drop_last=True)
    
    return [train_loader, val_loader, VOCAB_SIZE]       