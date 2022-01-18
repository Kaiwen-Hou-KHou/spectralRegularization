# -*- coding: utf-8 -*-
"""
Main

@author: Kaiwen Hou
kaiwen.hou@mila.quebec
"""

from experiment import *
tomita_num = 7

M, ACC, AIM, CE = sample_experiment(tomita_num)
min_train_acc, min_val_acc, min_test_acc = ACC
min_train_acc_ignore_markers, min_val_acc_ignore_markers, min_test_acc_ignore_markers = AIM
min_train_loss, min_val_loss, min_test_loss = CE