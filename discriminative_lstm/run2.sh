#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=results/output_all/output_val_predict_lstmsize32_lstm2size32_dropout0_lr100_epoch3000_batchsize16_sample50000_batchnormalized.txt
#SBATCH --mem=20000
THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32
python run_LSTM_smallsample_with_val.py 32 32 0 0.001 3000 16 50000 50000

