import torch
from simple_datasets import get_data_split, pad_data
#from utils import seed_everything

from char_lang_model import CharLanguageModel
from spectral_reg import SpectralRegularization
from early_stop import EarlyStopping
from train_model import train_epoch, eval_epoch

from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

import wandb 

import argparse
from argparse import RawTextHelpFormatter
USE_WANDB = True


def train_model(model, VOCAB_SIZE, optimizer, scheduler, train_loader, val_loader, test_loader_dict, early_stopping=None):
    results = {}
    #compute test and valid metrics
    #compute_val_test_metrics(model,val_loader,test_loader_dict,results,True)
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
        #compute_val_test_metrics(model,val_loader,test_loader_dict,results,True)
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
    parser.add_argument('--pautomac_number', type=int, default=1)
    parser.add_argument('--train_size', type=int, default=1000)
    #parser.add_argument('--test_size', type=int, default=2000)
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
    #parser.add_argument('--train_len', type=int, default=10)
    #parser.add_argument('--test_len_list', nargs='+', type=int, default=[10,12,14])
    parser.add_argument('--tag', nargs='+', type=str, default=None, action="append")
    parser.add_argument('--overlap_datasets', type=bool, default=False)

    opt = parser.parse_args()
    return opt


def main():
    
    opt = parse_option()

    if USE_WANDB:
        wandb.init()
        wandb.config.update(opt)
        wandb.run.name = f"pauto #{wandb.config.pautomac_number}, lambd={wandb.config.lambd},"
        wandb.run.name += f" train_size={wandb.config.train_size}, {wandb.config.hankel_russ_roul_type}-{wandb.config.stop_proba}"
        
        if opt.tag:
            wandb.run.tags += tuple([" ".join(t) for t in opt.tag])
    else:
        wandb.config = opt
    
    wandb.config["best_model_path"] = f"models/{wandb.run.id}-best_model.pt" if USE_WANDB else "models/checkpoint.pt"
    print("pautomac number:", wandb.config.pautomac_number)
    print("train size:", wandb.config.train_size)

    # data loaders
    with open('./PAutomaC-competition_sets/'+str(wandb.config.pautomac_number)+'.pautomac.train') as f:
        dataset = f.readlines()
    _, VOCAB_SIZE = pad_data(dataset, split_method='space')
    with open('./PAutomaC-competition_sets/'+str(wandb.config.pautomac_number)+'.pautomac.test') as f:
        testData = f.readlines()
    train_size, val_size = int(0.8*wandb.config.train_size),int(0.2*wandb.config.train_size)
    train_loader, val_loader, test_loader = get_data_split(dataset, train_size, val_size, 0, batch_size=wandb.config.batch_size, 
                                                           overlap=wandb.config.overlap_datasets, split_method='space', remove_header=True, 
                                                           testData=testData)
    test_loader_dict = {0:test_loader}

    # train model
    model = CharLanguageModel(vocab_size = VOCAB_SIZE, embed_size = VOCAB_SIZE, hidden_size=wandb.config.hidden_size, 
                              nlayers=1, rnn_type='RNN', 
                              nonlinearity='tanh').to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.lr, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',patience=wandb.config.reduceonplateau_patience,factor=0.5)
    early_stopping = EarlyStopping(patience=wandb.config.earlystop_patience, verbose=True, path=wandb.config.best_model_path)
    res = train_model(model, VOCAB_SIZE, optimizer, scheduler, train_loader, val_loader, test_loader_dict, early_stopping=early_stopping)

    if USE_WANDB:
        wandb.save(wandb.config.best_model_path)


if __name__ == '__main__':

    main()