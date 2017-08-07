#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=results/output_files/output_evaluate_singletask_bpic2017_outcome_all_data.csv
#SBATCH --mem=20000

THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32
python evaluate_singletask_outcome_all_data.py
