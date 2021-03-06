#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=/storage/anna_irene/lstm_output_complete_1sample.txt
#SBATCH --mem=20000

THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32
python run_LSTM_complete.py

chmod -R 777 /storage/anna_irene
