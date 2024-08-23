#!/bin/bash
#SBATCH -A csb180
#SBATCH --job-name="simlm"
#SBATCH --output="simlm.%j.%N.out"
#SBATCH --error="simlm.%j.%N.err"
#SBATCH --partition=gpu-shared
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=32G
#SBATCH -t 3:00:00



source ~/.bashrc
source activate splade

python gen_simlm_scores.py