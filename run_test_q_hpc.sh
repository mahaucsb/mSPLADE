#!/bin/bash
#SBATCH --job-name="indexq"
#SBATCH --output="indexq.%j.%N.out"
#SBATCH --error="indexq.%j.%N.err"
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --no-requeue
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=32G
#SBATCH -t 00:30:00
#SBATCH --nodelist=gpu2

source activate splade
MODEL_DIR=splade_distill_num1_kldiv_position_focal_mrr_diff_gamma5.0-alpha1.0_denoiseFalse_num20_kldiv_position_focal5-lambda0.0-0.0_lr1e-05-batch_size_32x4-2022-11-24
/bin/bash -c "python inference_q_SPLADE.py training_with_sentence_transformers/output/$MODEL_DIR/25000/0_MLMTransformer training_with_sentence_transformers/output/$MODEL_DIR/index"

