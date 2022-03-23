#!/bin/bash

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -n|--tomita_number) tomita_number="$2"; shift ;;
        -t|--tag) tag="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done



learning_rates=(0.01)
lambdas=(0.001 0.005 0.01)
train_sizes=(100 250 500 1000 2000)
russ_types=('block_diag' 'block_diag_no_norm')
stop_probas=(0.0000001 0.1 0.2)

train_length=10
test_lengths='10 12 14'  

# No regularization
for lambda in ${lambdas[@]}; do
    for lr in ${learning_rates[@]}; do
        for train_size in ${train_sizes[@]}; do
        	CMD="./train_model.sh --lr $lr --tomita_number $tomita_number \
               --train_size $train_size --lambd $lambda \
               --train_len $train_length --test_len_list $test_lengths --tag $tag --batch_size 128"
            echo "launching: " $CMD 
            sbatch --mem 10g -c 2 -G 1 -t 02:00:00 $CMD
        done
    done
done

# Regularization
for lambda in ${lambdas[@]}; do
    for lr in ${learning_rates[@]}; do
        for train_size in ${train_sizes[@]}; do
            for russ_type in ${russ_types[@]}; do
	            for stop_proba in ${stop_probas[@]}; do
	            	CMD="./train_model.sh --lr $lr --tomita_number $tomita_number \
		               --train_size $train_size --lambd $lambda --stop_proba $stop_proba --hankel_russ_roul_type $russ_type\
		               --train_len $train_length --test_len_list $test_lengths --tag $tag --batch_size 128"
                    echo "launching: " $CMD 
		            sbatch --mem 10g -c 2 -G 1 -t 02:00:00 $CMD
	            done
	        done
        done
    done
done

