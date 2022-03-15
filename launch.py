import torch
import torch.nn as nn
import copy
from spectral_reg import *
from accuracy import *
import random
from toy_datasets import *
from simple_datasets import *
from char_lang_model import CharLanguageModel
from early_stop import EarlyStopping
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

import argparse
from argparse import RawTextHelpFormatter

def parse_option():
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)

    parser.add_argument('--tomita_number', type=int)
    parser.add_argument('--train_size', type=int)
    parser.add_argument('--name', type=str)
    opt = parser.parse_args()
    return opt

def pad_data(dataset, max_len, VOCAB_SIZE=4):
    split_str = [list(np.array(list(i)).astype(int)) for i in dataset]
    return [l + [VOCAB_SIZE-1] * (max_len - len(l)) for l in split_str]

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

def train_model(model, VOCAB_SIZE, optimizer, scheduler, train_loader, val_loader, test_loader_dict, lam, early_stopping=None, n_epoch=200):
    
    opt = parse_option()
    
    results = {}
    results['train_losses'], results['val_losses'], results['test_losses'] = [], [], {test_len:[] for test_len in test_loader_dict.keys()}
    results['train_acc'], results['val_acc'], results['test_acc'] = [], [], {test_len:[] for test_len in test_loader_dict.keys()}
    results['train_aim'], results['val_aim'], results['test_aim'] = [], [], {test_len:[] for test_len in test_loader_dict.keys()}
    
    for epoch in tqdm(range(n_epoch)):
        
        val_loss, val_acc, val_aim = eval_epoch(model, val_loader, lam=lam)
        results['val_losses'].append(val_loss)
        results['val_acc'].append(val_acc)
        results['val_aim'].append(val_aim)

        for test_len, test_loader in test_loader_dict.items():
            test_loss, test_acc, test_aim = eval_epoch(model, test_loader, lam=lam)
            results['test_losses'][test_len].append(test_loss)
            results['test_acc'][test_len].append(test_acc)
            results['test_aim'][test_len].append(test_aim)
            
        train_loss, train_acc, train_aim = train_epoch(model, VOCAB_SIZE, optimizer, train_loader, lam=lam) 
        results['train_losses'].append(train_loss)
        results['train_acc'].append(train_acc)
        results['train_aim'].append(train_aim)
        
        scheduler.step(val_loss)
        
        if epoch > 20:
            if early_stopping:
                early_stopping(val_loss, model)
                if early_stopping.early_stop:
                    break
                else:
                    best_model_trained = copy.deepcopy(model)
                    
        results['last_model'] = model

        torch.save(results,f"T{opt.tomita_number}-N{opt.train_size}-Lambda{lam}-results.pt")
        
    return results


def main(train_len=15, test_len_list=[15, 17, 20], lam_list = [0, 0.001, 0.01, 0.1, 0.2, 0.3, 0.5, 1, 2]):
    
    opt = parse_option()
    print("tomita number:", opt.tomita_number)
    print("train size:", opt.train_size)
        
    # generate test data
    data_split = 0.99999999
    rng_key = jax.random.PRNGKey(12345)
    
    data_dict = {}
    test_dataset_dict = {}
    test_loader_dict = {}
    
    for max_len in tqdm(test_len_list):
        dataset = tomita_dataset(rng_key, data_split, max_len, tomita_num=opt.tomita_number, min_len=max_len)[0] # fix length on test dataset
        data_dict[max_len] = pad_data(dataset, max_len)
        dataForTesting = np.array(random.choices(data_dict[max_len],k=2000))
        test_data = torch.tensor(dataForTesting)
        test_dataset_dict[max_len] = SimpleDataset(test_data)
        test_loader_dict[max_len] = DataLoader(test_dataset_dict[max_len], shuffle=False, batch_size=len(test_data), collate_fn = collate, drop_last=True)
    
    # generate training data
    dataset = tomita_dataset(rng_key, data_split, train_len, tomita_num=opt.tomita_number)[0] 
    data = pad_data(dataset, train_len)
    train_loader, val_loader, VOCAB_SIZE = get_random_training_data(data, opt.train_size, split=0.8)
    
    # store experiments
    results = {}
    for lam in tqdm(lam_list):
        model = CharLanguageModel(vocab_size = VOCAB_SIZE, embed_size = VOCAB_SIZE, hidden_size=50, nlayers=1, rnn_type='RNN', 
                              nonlinearity='tanh').to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-1, amsgrad=True)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        early_stopping = EarlyStopping(patience=10, verbose=True)
        res = train_model(model, VOCAB_SIZE, optimizer, scheduler, train_loader, val_loader, test_loader_dict, lam=lam, 
                                                                               early_stopping=early_stopping, 
                                                                               n_epoch=1000)
        results[lam] = res
        
    return results


if __name__ == '__main__':
    main()