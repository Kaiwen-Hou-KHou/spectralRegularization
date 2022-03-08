# -*- coding: utf-8 -*-
"""
training tools

@author: Kaiwen Hou
kaiwen.hou@mila.quebec
"""

import torch
import torch.nn as nn
import copy
from spectral_reg import *
from accuracy import *

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train_epoch(model, VOCAB_SIZE, optimizer, train_loader, lam, stopProb = 0.2):
    model.train()
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    spc_loss = SpectralRegularization().to(DEVICE)
    train_loss = 0
    #print("training", len(train_loader), "number of batches")
    
    if lam > 0:
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
        total_loss = loss + lam*hankel_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        train_loss += total_loss.item()
        
        acc.append(Accuracy(outputs, targets, ignore_markers=False))
        acc_ignore_markers.append(Accuracy(outputs, targets, ignore_markers=True))

    return train_loss / len(train_loader), torch.mean(torch.tensor(acc)), torch.mean(torch.tensor(acc_ignore_markers))


def eval_epoch(model, val_loader, lam):  
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
            total_loss = loss #+ lam*hankel_loss
            val_loss+= total_loss.item()
            
            acc.append(Accuracy(outputs, targets, ignore_markers=False))
            acc_ignore_markers.append(Accuracy(outputs, targets, ignore_markers=True))
    
    return val_loss / len(val_loader), torch.mean(torch.tensor(acc)), torch.mean(torch.tensor(acc_ignore_markers))


def train_model(model, VOCAB_SIZE, optimizer, scheduler, train_loader, val_loader, test_loader_list, lam, early_stopping=None, n_epoch=200):
    
    results = {}
    results['train_losses'], results['val_losses'], results['test_losses'] = [], [], []
    results['train_acc'], results['val_acc'], results['test_acc'] = [], [], []
    results['train_aim'], results['val_aim'], results['test_aim'] = [], [], []
    
    for epoch in tqdm(range(n_epoch)):
        
        val_loss_epoch, val_acc_epoch, val_acc_ignore_markers_epoch = eval_epoch(model, val_loader, lam=lam)
        val_loss.append(val_loss_epoch)
        val_acc.append(val_acc_epoch)
        val_acc_ignore_markers.append(val_acc_ignore_markers_epoch)
        
        test_loss, test_acc, test_acc_ignore_markers = [], [], []
        for test_loader in test_loader_list:
            test_loss_epoch, test_acc_epoch, test_acc_ignore_markers_epoch = eval_epoch(model, test_loader, lam=lam)
            test_loss.append(test_loss_epoch)
            test_acc.append(test_acc_epoch)
            test_acc_ignore_markers.append(test_acc_ignore_markers_epoch)
        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)
        test_acc_ignore_markers_list.append(test_acc_ignore_markers)
            
        train_loss_epoch, train_acc_epoch, train_acc_ignore_markers_epoch = train_epoch(model, VOCAB_SIZE, optimizer, train_loader, lam=lam) 
        train_loss.append(train_loss_epoch)
        train_acc.append(train_acc_epoch)
        train_acc_ignore_markers.append(train_acc_ignore_markers_epoch)
        
        scheduler.step(val_loss_epoch)
        
        if epoch > 20:
            if early_stopping:
                early_stopping(val_loss[-1], model)
                if early_stopping.early_stop:
                    break
                else:
                    best_model_trained = copy.deepcopy(model)

    return [torch.argmin(torch.tensor(val_loss)), [train_loss, val_loss, test_loss_list], [train_acc, val_acc, test_acc_list],
            [train_acc_ignore_markers, val_acc_ignore_markers, test_acc_ignore_markers_list], best_model_trained]