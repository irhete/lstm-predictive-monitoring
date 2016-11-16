#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=traffic_output_neg.txt
#SBATCH --mem=10000

THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32
python train_LSTM.py "/storage/hpc_irheta/bpm_data/traffic_fines_train_neg.csv" "traffic_neg.model"
