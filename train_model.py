import torch
import torch.nn as nn

from toy_datasets import tomita_dataset,tomita_2,tomita_3,tomita_4,tomita_5,tomita_6,tomita_7
from simple_datasets import get_data_split

from char_lang_model import CharLanguageModel
from spectral_reg import SpectralRegularization
from early_stop import EarlyStopping
from accuracy import Accuracy, ratio_correct_samples

#import copy
from tqdm import tqdm
from utils import seed_everything
#import random

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#DEVICE = "cpu"

import wandb 

import argparse
from argparse import RawTextHelpFormatter
USE_WANDB = True

tomita_function=None


def train_epoch(model, VOCAB_SIZE, optimizer, train_loader):
    model.train()
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    spc_loss = SpectralRegularization().to(DEVICE)
    train_loss = 0
    train_cce_loss = 0
    train_hankel_loss = 0
    

    
    acc = []
    acc_ignore_markers = []
    for batch_idx, (inputs,targets) in enumerate(train_loader):
        # print("BATCH SIZE",inputs.shape)
        if wandb.config.lambd > 0:
            hankel_loss = spc_loss.forward(model, VOCAB_SIZE, stopProb=wandb.config.stop_proba, use_wandb=USE_WANDB, 
                            hankelSizeCap=wandb.config.hankel_size_cap, russian_roulette_type=wandb.config.hankel_russ_roul_type)
        else:
            hankel_loss = torch.tensor(0)
        inputs = inputs.to(DEVICE).long()
        targets = targets.to(DEVICE).long()
        outputs = model(inputs)  # N x L x V
        loss = criterion(outputs.view(-1, outputs.shape[2]), targets.view(-1)) # Loss of the flattened outputs
        total_loss = loss + wandb.config.lambd*hankel_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        train_loss += total_loss.item()
        train_cce_loss += loss.item()
        train_hankel_loss += hankel_loss.item()
        
        acc.append(Accuracy(outputs, targets, ignore_markers=False))
        acc_ignore_markers.append(Accuracy(outputs, targets, ignore_markers=True))

    return {"train_loss":train_loss, "train_cce_loss":train_cce_loss, "train_hankel_loss":train_hankel_loss,
            "train_acc":torch.mean(torch.tensor(acc)), "train_aim":torch.mean(torch.tensor(acc_ignore_markers))}

def eval_epoch(model, data_loader):  
    model.eval()
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    total_loss = 0
    #print("validating", len(val_loader), "number of batches")
    with torch.no_grad():
        acc = []
        acc_ignore_markers = []
        for batch_idx, (inputs,targets) in enumerate(data_loader):
            inputs = inputs.to(DEVICE).long()
            targets = targets.to(DEVICE).long()
            outputs = model(inputs)

            loss = criterion(outputs.view(-1, outputs.shape[2]), targets.view(-1))
            total_loss += loss.item()
            
            acc.append(Accuracy(outputs, targets, ignore_markers=False))
            acc_ignore_markers.append(Accuracy(outputs, targets, ignore_markers=True))

    return total_loss / len(data_loader), torch.mean(torch.tensor(acc)), torch.mean(torch.tensor(acc_ignore_markers))

def compute_val_test_metrics(model,val_loader,test_loader_dict,results,compute_ratio_correct_samples=False):
    p='val_'
    results[p+'loss'],results[p+'acc'],results[p+'aim'] = eval_epoch(model, val_loader)
    for test_len, test_loader in test_loader_dict.items(): 
        p='test' + str(test_len)
        results[p+'loss'],results[p+'acc'],results[p+'aim'] = eval_epoch(model, test_loader)

    if compute_ratio_correct_samples:
        with torch.no_grad():
            results['ratio_correct_samples'] = ratio_correct_samples(model,tomita_function)
        # print(f"{results['ratio_correct_samples']*100}/100 correct samples")

def train_model(model, VOCAB_SIZE, optimizer, scheduler, train_loader, val_loader, test_loader_dict, early_stopping=None):
    results = {}
    #compute test and valid metrics
    compute_val_test_metrics(model,val_loader,test_loader_dict,results,True)
    if USE_WANDB: wandb.log(results)

    for epoch in tqdm(range(wandb.config.n_epochs)):
        
        # with torch.no_grad():
        #     samples = []
        #     for i in range(10):
        #         samples.append(model.sample_word())
        #     print("|".join(samples))

        res = train_epoch(model, VOCAB_SIZE, optimizer, train_loader) 
        results.update(res)
        
        #compute test and valid metrics
        compute_val_test_metrics(model,val_loader,test_loader_dict,results,True)
        if USE_WANDB: wandb.log(results)

        val_loss = results['val_loss']
        scheduler.step(val_loss)
        if USE_WANDB:
            for i, param_group in enumerate(optimizer.param_groups):
                 wandb.log({'lr':float(param_group['lr'])})


        if early_stopping:
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                break
                    

def parse_option():
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument('--tomita_number', type=int, default=3)
    parser.add_argument('--train_size', type=int, default=1000)
    parser.add_argument('--test_size', type=int, default=2000)
    parser.add_argument('--lambd', type=float, default=0)
    parser.add_argument('--stop_proba', type=float, default=0.2)
    parser.add_argument('--hankel_size_cap', type=int, default=10)
    parser.add_argument('--hankel_russ_roul_type', type=str, default='block_diag')
    parser.add_argument('--earlystop_patience', type=int, default=20)
    parser.add_argument('--reduceonplateau_patience', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--hidden_size', type=int, default=50)
    parser.add_argument('--n_epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--train_len', type=int, default=10)
    parser.add_argument('--test_len_list', nargs='+', type=int, default=[10,12,14])
    parser.add_argument('--tag', nargs='+', type=str, default=None, action="append")

    opt = parser.parse_args()
    return opt


def main():

    opt = parse_option()

    if USE_WANDB:
        wandb.init()
        wandb.config.update(opt)
        wandb.run.name = f"tom #{wandb.config.tomita_number}, lambd={wandb.config.lambd},"
        wandb.run.name += f" train_size={wandb.config.train_size}, {wandb.config.hankel_russ_roul_type}-{wandb.config.stop_proba}"
        
        if opt.tag:
            wandb.run.tags += tuple([" ".join(t) for t in opt.tag])
    else:
        wandb.config = opt

    wandb.config["best_model_path"] = f"models/{wandb.run.id}-best_model.pt" if USE_WANDB else "models/checkpoint.pt"
    global tomita_function
    f = globals()[f"tomita_{opt.tomita_number}"]
    tomita_function = lambda w: f(w) and not "2" in w
    print("tomita number:", wandb.config.tomita_number)
    print("train size:", wandb.config.train_size)
    VOCAB_SIZE = 4
        
    # generate test data
    data_split = 0.999999999  # ???
    #import jax
    rng_key = seed_everything(42)
    #data_dict = {}
    test_loader_dict = {}
    
    overlaping_datasets = False

    if overlaping_datasets:
        # generate data
        train_size, val_size = int(0.8*wandb.config.train_size),int(0.2*wandb.config.train_size)
        test_size = wandb.config.test_size

        for max_len in tqdm(wandb.config.test_len_list):
            dataset = tomita_dataset(rng_key, data_split, max_len=max_len, tomita_num=wandb.config.tomita_number, min_len=max_len)[0] # fix length on test dataset
            _,_,test_loader_dict[max_len] = get_data_split(dataset, 0, 0, test_size, overlap=True)

        dataset = tomita_dataset(rng_key, data_split, wandb.config.train_len, tomita_num=wandb.config.tomita_number)[0] 
        train_loader, val_loader, _ = get_data_split(dataset, train_size, val_size, 0, batch_size=wandb.config.batch_size, overlap=True)
    else:
        train_size, val_size = int(0.8*wandb.config.train_size),int(0.2*wandb.config.train_size)
        test_size = wandb.config.test_size

        dataset = tomita_dataset(rng_key, data_split, wandb.config.train_len, tomita_num=wandb.config.tomita_number)[0] 
        if wandb.config.train_len in wandb.config.test_len_list:
            train_loader, val_loader, test_loader_dict[wandb.config.train_len] = get_data_split(dataset, train_size, val_size, test_size, batch_size=wandb.config.batch_size, overlap=False)
        else:
            train_loader, val_loader, test_loader_dict[max_len] = get_data_split(dataset, train_size, val_size, 0, batch_size=wandb.config.batch_size, overlap=False)
        for max_len in tqdm(wandb.config.test_len_list):
            if max_len == wandb.config.train_len: continue
            dataset = tomita_dataset(rng_key, data_split, max_len=max_len, tomita_num=wandb.config.tomita_number, min_len=max_len)[0] # fix length on test dataset
            _,_,test_loader_dict[max_len] = get_data_split(dataset, 0, 0, test_size, overlap=False)



    # generate training data
    # train model
    model = CharLanguageModel(vocab_size = VOCAB_SIZE, embed_size = VOCAB_SIZE, hidden_size=wandb.config.hidden_size, nlayers=1, rnn_type='RNN', 
                              nonlinearity='tanh').to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.lr, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',patience=wandb.config.reduceonplateau_patience,factor=0.5)
    early_stopping = EarlyStopping(patience=wandb.config.earlystop_patience, verbose=True, path=wandb.config.best_model_path)
    res = train_model(model, VOCAB_SIZE, optimizer, scheduler, train_loader, val_loader, test_loader_dict, early_stopping=early_stopping)

    if USE_WANDB:
        wandb.save(wandb.config.best_model_path)


if __name__ == '__main__':

    #import sys, os
    #print(os.path.basename(sys.argv[0]), sys.argv[1:])
    main()
