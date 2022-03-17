# -*- coding: utf-8 -*-
"""
Accuracy Metric

@author: Kaiwen Hou
kaiwen.hou@mila.quebec
"""

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#DEVICE = "cpu"

def get_mask(targets):
    mask = []
    for lst in targets:
        boolean = [1]
        for i in range(1,len(lst)):
            if lst[i-1] != 3:
                boolean.append(1)
            else:
                boolean.append(0)
        mask.append(boolean)
    return torch.tensor(mask).to(DEVICE)

def Accuracy(outputs, targets, ignore_markers=False):
    """
    outputs: B * (L+1) * VOCAB_SIZE
    targets: B * (L+1)
    """
    B, L = targets.size()
    mask = get_mask(targets)
    if not ignore_markers:
        return 1 - sum(sum(mask * (torch.argmax(outputs, axis=2) != targets).to(DEVICE)))/(B * L)
    else:
        return 1 - sum(sum(mask * (torch.argmax(outputs[:,:,:-2], axis=2) != targets).to(DEVICE)))/(B * L)
