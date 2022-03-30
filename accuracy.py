# -*- coding: utf-8 -*-
"""
Accuracy Metrics
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

def ratio_correct_samples(model, language, N_trials=100):
    count = 0
    for n in range(N_trials):
        w = model.sample_word(max_length=50)
        if language(w): count += 1
    return count / N_trials

