# -*- coding: utf-8 -*-
"""
training tools

@author: Kaiwen Hou
kaiwen.hou@mila.quebec
"""

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import copy
from spectral_reg import *
from accuracy import *

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train_epoch(model, VOCAB_SIZE, optimizer, train_loader, λ, stopProb = 0.5):
    model.train()
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    spc_loss = SpectralRegularization().to(DEVICE)
    train_loss = 0
    #print("training", len(train_loader), "number of batches")
    
    if λ > 0:
        hankel_loss = spc_loss.forward(model, VOCAB_SIZE)
    else:
        hankel_loss = 0
    
    acc = []
    acc_ignore_markers = []
    for batch_idx, (inputs,targets) in enumerate(train_loader):
        inputs = inputs.to(DEVICE).long()
        targets = targets.to(DEVICE).long()
        outputs = model(inputs)  # N x L x V
        loss = criterion(outputs.view(-1, outputs.shape[2]), targets.view(-1)) # Loss of the flattened outputs
        total_loss = loss + λ*hankel_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        train_loss += total_loss.item()
        
        acc.append(Accuracy(outputs, targets, ignore_markers=False))
        acc_ignore_markers.append(Accuracy(outputs, targets, ignore_markers=True))

    return train_loss / len(train_loader), np.mean(acc), np.mean(acc_ignore_markers)


def eval_epoch(model, val_loader, λ):  
    model.eval()
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    val_loss = 0
    #print("validating", len(val_loader), "number of batches")
    with torch.no_grad():
        acc = []
        acc_ignore_markers = []
        for batch_idx, (inputs,targets) in enumerate(val_loader):
            inputs = inputs.to(DEVICE).long()
            targets = targets.to(DEVICE).long()
            outputs = model(inputs)
        
            #logH = make_hankel(model, prefixes, suffixes)
            loss = criterion(outputs.view(-1, outputs.shape[2]), targets.view(-1))
            #hankel_loss = torch.norm(torch.exp(logH - logH.max()), p='nuc')
            total_loss = loss #+ λ*hankel_loss
            val_loss+= total_loss.item()
            
            acc.append(Accuracy(outputs, targets, ignore_markers=False))
            acc_ignore_markers.append(Accuracy(outputs, targets, ignore_markers=True))
    
    return val_loss / len(val_loader), np.mean(acc), np.mean(acc_ignore_markers)


def train_model(model, VOCAB_SIZE, optimizer, train_loader, val_loader, test_loader_list, λ, early_stopping=None, n_epoch=200):
    
    train_loss, train_acc, train_acc_ignore_markers = [], [], []
    val_loss, val_acc, val_acc_ignore_markers = [], [], []
    test_loss_list, test_acc_list, test_acc_ignore_markers_list = [], [], []
    best_model_trained = None
    
    for epoch in tqdm(range(n_epoch)):
        
        val_loss_epoch, val_acc_epoch, val_acc_ignore_markers_epoch = eval_epoch(model, val_loader, λ=λ)
        val_loss.append(val_loss_epoch)
        val_acc.append(val_acc_epoch)
        val_acc_ignore_markers.append(val_acc_ignore_markers_epoch)
        
        test_loss, test_acc, test_acc_ignore_markers = [], [], []
        for test_loader in test_loader_list:
            test_loss_epoch, test_acc_epoch, test_acc_ignore_markers_epoch = eval_epoch(model, test_loader, λ=λ)
            test_loss.append(test_loss_epoch)
            test_acc.append(test_acc_epoch)
            test_acc_ignore_markers.append(test_acc_ignore_markers_epoch)
        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)
        test_acc_ignore_markers_list.append(test_acc_ignore_markers)
            
        train_loss_epoch, train_acc_epoch, train_acc_ignore_markers_epoch = train_epoch(model, VOCAB_SIZE, optimizer, train_loader, λ=λ)    
        train_loss.append(train_loss_epoch)
        train_acc.append(train_acc_epoch)
        train_acc_ignore_markers.append(train_acc_ignore_markers_epoch)
        
        if epoch > 10:
            if early_stopping:
                early_stopping(val_loss[-1], model)
                if early_stopping.early_stop:
                    break
                else:
                    best_model_trained = copy.deepcopy(model)

    return [np.argmin(val_loss), [train_loss, val_loss, test_loss_list], [train_acc, val_acc, test_acc_list],
            [train_acc_ignore_markers, val_acc_ignore_markers, test_acc_ignore_markers_list], best_model_trained]