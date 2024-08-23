#!/bin/bash
#SBATCH -A csb180
#SBATCH --job-name="index"
#SBATCH --output="index.%j.%N.out"
#SBATCH --error="index.%j.%N.err"
#SBATCH --partition=gpu-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --no-requeue
#SBATCH --gpus=1
#SBATCH -t 24:00:00

source activate splade
MODEL_DIR=splade_distill_num1_klar-0.01-0.015_denoiseFalse_num20_5-lr1e-05-batch_size_32x4-2022-12-15
/bin/bash -c "python inference_SPLADE.py training_with_sentence_transformers/output/$MODEL_DIR/4000/0_MLMTransformer training_with_sentence_transformers/output/$MODEL_DIR/index_4000"

