#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=/storage/anna_irene/output_sample1000/output_lstmsize16_lstm2size16_dropout0_lr100_epoch3000_batchsize8_sample1000_batchnormalized.txt
#SBATCH --mem=20000
THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32
python run_LSTM_smallsample.py 16 16 0 0.001 3000 8 1000

chmod -R 777 /storage/anna_irene