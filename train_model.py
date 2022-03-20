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
#DEVICE = "cpu"

import wandb 

import argparse
from argparse import RawTextHelpFormatter
USE_WANDB = True


def pad_data(dataset, max_len, VOCAB_SIZE=4):
    split_str = [list(np.array(list(i)).astype(int)) for i in dataset]
    return [l + [VOCAB_SIZE-1] * (max_len - len(l)) for l in split_str]

def train_epoch(model, VOCAB_SIZE, optimizer, train_loader):
    model.train()
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    spc_loss = SpectralRegularization().to(DEVICE)
    train_loss = 0
    train_cce_loss = 0
    train_hankel_loss = 0
    
    if wandb.config.lambd > 0:
        hankel_loss = spc_loss.forward(model, VOCAB_SIZE, stopProb=wandb.config.stop_proba, use_wandb=USE_WANDB, 
                        hankelSizeCap=wandb.config.hankel_size_cap, russian_roulette_type=wandb.config.hankel_russ_roul_type)
    else:
        hankel_loss = torch.tensor(0)
    
    acc = []
    acc_ignore_markers = []
    for batch_idx, (inputs,targets) in enumerate(train_loader):
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

def train_model(model, VOCAB_SIZE, optimizer, scheduler, train_loader, val_loader, test_loader_dict, early_stopping=None):
    results = {}
    for epoch in tqdm(range(wandb.config.n_epochs)):
        val_loss, val_acc, val_aim = eval_epoch(model, val_loader)
        results['val_loss'] = val_loss
        results['val_acc'] = val_acc
        results['val_aim'] = val_aim

        for test_len, test_loader in test_loader_dict.items():
            test_loss, test_acc, test_aim = eval_epoch(model, test_loader)
            results['test_loss-' + str(test_len)] = test_loss
            results['test_acc-' + str(test_len)] = test_acc
            results['test_aim-' + str(test_len)] = test_aim
            
        res = train_epoch(model, VOCAB_SIZE, optimizer, train_loader) 
        results.update(res)
        
        scheduler.step(val_loss)
        
        if USE_WANDB:
            wandb.log(results)

        if epoch > 20:
            if early_stopping:
                early_stopping(val_loss, model)
                if early_stopping.early_stop:
                    break
                    

def parse_option():
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)

    parser.add_argument('--tomita_number', type=int, default=3)
    parser.add_argument('--train_size', type=int, default=1000)
    parser.add_argument('--lambd', type=float, default=0)
    parser.add_argument('--stop_proba', type=float, default=0.2)
    parser.add_argument('--hankel_size_cap', type=int, default=11)
    parser.add_argument('--hankel_russ_roul_type', type=str, default='block_diag')
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--hidden_size', type=int, default=50)
    parser.add_argument('--n_epochs', type=int, default=1000)
    parser.add_argument('--train_len', type=int, default=15)
    parser.add_argument('--test_len_list', nargs='+', type=int, default=[15,17,20])
    parser.add_argument('--tag', nargs='+', type=str, default=None, action="append")

    opt = parser.parse_args()
    return opt


def main():

    opt = parse_option()

    if USE_WANDB:
        wandb.init()
        wandb.config.update(opt)
        wandb.run.name = f"tom #{wandb.config.tomita_number}, lr={wandb.config.lr}, lambd={wandb.config.lambd}, train_size={wandb.config.train_size}"
        print() 
        if opt.tag:
            wandb.run.tags += tuple([" ".join(t) for t in opt.tag])
    else:
        wandb.config = opt

    wandb.config["best_model_path"] = f"{wandb.run.id}-best_model.pt" if USE_WANDB else "checkpoint.pt"

    print("tomita number:", wandb.config.tomita_number)
    print("train size:", wandb.config.train_size)
        
    # generate test data
    data_split = 0.99999999  # ???
    import jax
    rng_key = jax.random.PRNGKey(12345)
    
    data_dict = {}
    test_dataset_dict = {}
    test_loader_dict = {}
    
    # generate test data
    for max_len in tqdm(wandb.config.test_len_list):
        dataset = tomita_dataset(rng_key, data_split, max_len, tomita_num=wandb.config.tomita_number, min_len=max_len)[0] # fix length on test dataset
        data_dict[max_len] = pad_data(dataset, max_len)
        dataForTesting = np.array(random.choices(data_dict[max_len],k=2000))
        test_data = torch.tensor(dataForTesting)
        test_dataset_dict[max_len] = SimpleDataset(test_data)
        test_loader_dict[max_len] = DataLoader(test_dataset_dict[max_len], shuffle=False, batch_size=len(test_data), collate_fn = collate, drop_last=True)
    
    # generate training data
    dataset = tomita_dataset(rng_key, data_split, wandb.config.train_len, tomita_num=wandb.config.tomita_number)[0] 
    data = pad_data(dataset, wandb.config.train_len)
    train_loader, val_loader, VOCAB_SIZE = get_random_training_data(data, opt.train_size, split=0.8)
    
    # train model
    model = CharLanguageModel(vocab_size = VOCAB_SIZE, embed_size = VOCAB_SIZE, hidden_size=wandb.config.hidden_size, nlayers=1, rnn_type='RNN', 
                              nonlinearity='tanh').to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.lr, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    early_stopping = EarlyStopping(patience=wandb.config.patience, verbose=True, path=wandb.config.best_model_path)
    res = train_model(model, VOCAB_SIZE, optimizer, scheduler, train_loader, val_loader, test_loader_dict, early_stopping=early_stopping)

    if USE_WANDB:
        wandb.save(wandb.config.best_model_path)


if __name__ == '__main__':

    import sys, os
    #print(os.path.basename(sys.argv[0]), sys.argv[1:])
    main()
