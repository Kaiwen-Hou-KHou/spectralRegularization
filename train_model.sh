#!/bin/bash
SOURCE_DIR=~/repos/spectralRegularization
ENV_DIR=~
module load python/3.8
source $ENV_DIR/env/bin/activate
python $SOURCE_DIR/train_model.py "$@"
