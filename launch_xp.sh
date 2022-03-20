#!/bin/bash

tag="xp 5 - tomita 3 full"

learning_rates=(0.001 0.01 0.1)
lambdas=(0 0.005 0.01 0.05 0.1)
train_sizes=(100 250 500 1000 2000)
russ_types=('block_diag' 'block_diag_no_norm')
stop_probas=(0.0000001 0.1 0.2)

tomita_number=3
train_length=10
test_lengths='10 12 14'  

for lambda in ${lambdas[@]}; do
    for lr in ${learning_rates[@]}; do
        for train_size in ${train_sizes[@]}; do
            for russ_type in ${russ_types[@]}; do
	            for stop_proba in ${stop_probas[@]}; do
	            	CMD="./train_model.sh --lr $lr --tomita_number $tomita_number \
		               --train_size $train_size --lambd $lambda --stop_proba $stop_proba --hankel_russ_roul_type $russ_type\
		               --train_len $train_length --test_len_list $test_lengths --tag '$tag'"
		            #sbatch --mem 10g -c 2 -G 1 -t 01:00:00 $CMD
		            echo "launching: " $CMD
	            done
	        done
        done
    done
done

