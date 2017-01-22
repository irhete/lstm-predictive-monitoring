#!/bin/bash

#SBATCH --partition=long
#SBATCH --output=output_evaluate.txt
#SBATCH --mem=5000

python evaluate_log_enhancement_cv.py
