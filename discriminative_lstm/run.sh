#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=lstm_output.txt
#SBATCH --mem=20000

THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32
python run_LSTM.py
