#!/bin/bash
SOURCE_DIR=~/repos/spectralRegularization
module load python/3.8
source $SOURCE_DIR/env/bin/activate
python $SOURCE_DIR/train_model.py "$@"
