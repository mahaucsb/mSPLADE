#!/bin/bash
#SBATCH --job-name="index"
#SBATCH --output="index.%j.%N.out"
#SBATCH --error="index.%j.%N.err"
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --no-requeue
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=32G
#SBATCH -t 5:00:00

source ~/.bashrc
source activate indexing_env
/bin/bash -c "python buildIndex_pisa_target_size.py training_with_sentence_transformers/output/splade_distill_num1_kldiv_position_focal_mrr_diff_gamma5.0-alpha1.0_denoiseFalse_num20_kldiv_position_focal5-lambda0.0-0.0_lr1e-05-batch_size_32x4-2022-11-24/index/file_ training_with_sentence_transformers/output/splade_distill_num1_kldiv_position_focal_mrr_diff_gamma5.0-alpha1.0_denoiseFalse_num20_kldiv_position_focal5-lambda0.0-0.0_lr1e-05-batch_size_32x4-2022-11-24/index/inverted_index 300 89"