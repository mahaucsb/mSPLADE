#!/bin/bash
#SBATCH -A csb176
#SBATCH --job-name="index"
#SBATCH --output="index.%j.%N.out"
#SBATCH --error="index.%j.%N.err"
#SBATCH --partition=gpu-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --no-requeue
#SBATCH --gpus=1
#SBATCH -t 24:00:00

/bin/bash -c "python inference_SPLADE.py training_with_sentence_transformers/output/distilSplade_0.01_0.008_Luyu-co-condenser-marco-batch_size_16x4-2022-04-11_12-47-14/100000/0_MLMTransformer training_with_sentence_transformers/output/distilSplade_0.01_0.008_Luyu-co-condenser-marco-batch_size_16x4-2022-04-11_12-47-14/index_100000"

