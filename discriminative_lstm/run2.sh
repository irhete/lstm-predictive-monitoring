#!/bin/bash
#SBATCH --partition=long
#SBATCH --output=results/output_all/output_rf_evaluate_bpic2017_sample50000.txt
#SBATCH --mem=20000
python evaluate_RF.py 50000

