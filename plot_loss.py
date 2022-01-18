# -*- coding: utf-8 -*-
"""
plots

@author: Kaiwen Hou
kaiwen.hou@mila.quebec
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.5)
import copy

def plotLossCurves(test_len_list, stoppingEpoch, losses, label):
    train_loss, val_loss, test_loss_list = losses
    test_df = pd.DataFrame(test_loss_list, columns=test_len_list)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18,4))
    ax[0].plot(train_loss, label='training')
    ax[0].plot(val_loss, label='validation')
    ax[0].scatter(stoppingEpoch, val_loss[stoppingEpoch])
    ax[0].legend()
    ax[0].set_xlabel('Training Epoch')
    ax[0].set_title('Training and Validation '+label)
    for i in test_len_list:
        ax[1].plot(test_df[i], label=i)
    ax[1].scatter(stoppingEpoch, val_loss[stoppingEpoch])
    ax[1].legend()
    ax[1].set_xlabel('Training Epoch')
    ax[1].set_title(label+' on Test Set of Various Lengths')
    
    
def plotLoss(loss_df, title, test_len_list, plot_ci=False):
    
    λ_list = loss_df[0].columns
    
    if 'est' in title:
        
        for test_idx in range(len(test_len_list)):
            df = copy.deepcopy(loss_df)
            for i in range(len(df.columns)):
                for idx in df.index:
                    df[df.columns[i]][idx] = df[df.columns[i]][idx][test_idx]
            
            fig, ax = plt.subplots(figsize=(18,4))
            for λ in λ_list:
                y = df.T.swaplevel().loc[λ].mean()
                ax.plot(df.T.columns, y,'*-', label='λ = '+str(np.round(λ,2)))
                if plot_ci:
                    #ci = 1.96 * df.T.swaplevel().loc[λ].std()/y
                    ci = df.T.swaplevel().loc[λ].std()/y
                    ax.fill_between(df.T.columns, (y-ci), (y+ci), alpha=.1)
                ax.set_xlabel('N')
                ax.set_title(title+' on Data of Length '+str(test_len_list[test_idx]))
                ax.legend()
            plt.xscale('log')
        
    else:
    
        fig, ax = plt.subplots(figsize=(18,4))
        for λ in λ_list:
            y = loss_df.T.swaplevel().loc[λ].mean()
            ax.plot(loss_df.T.columns, y,'*-', label='λ = '+str(np.round(λ,2)))
            if plot_ci:
                #ci = 1.96 * loss_df.T.swaplevel().loc[λ].std()/y
                ci = loss_df.T.swaplevel().loc[λ].std()/y
                ax.fill_between(loss_df.T.columns, (y-ci), (y+ci), alpha=.1)
            ax.set_xlabel('N')
            ax.set_title(title)
            ax.legend()
        plt.xscale('log')
#     plt.yscale('log')