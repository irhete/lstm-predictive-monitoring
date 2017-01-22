#!/bin/bash

#SBATCH --partition=long
#SBATCH --output=output_generate.txt
#SBATCH --mem=20000

python generate_enhanced_logs.py
