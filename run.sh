#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=bpic2012_output.txt
#SBATCH --mem=23000

THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32
python train_LSTM_process_discovery.py
