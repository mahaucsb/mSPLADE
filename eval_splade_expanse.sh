#!/bin/bash
#SBATCH -A csb176
#SBATCH --job-name="eval_3v_tokensparse_prednosparse_80k"
#SBATCH --output="eval_3v_tokensparse_prednosparse_80k.%j.%N.out"
#SBATCH --error="eval_3v_tokensparse_prednosparse_80k.%j.%N.err"
#SBATCH --partition=gpu-shared
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=32G
#SBATCH -t 4:00:00



source ~/.bashrc
source activate splade

#MODEL_DIR=splade_distill_num1_kldiv_mrr_diff_denoiseFalse_num20_kldiv5-lambda0.0008-0.001_lr1e-05-batch_size_64x2-2022-11-09
#MODEL_DIR=splade_distill_num1_kldiv_focal_mrr_diff_gamma5.0_denoiseFalse_num20_kldiv_focal5-lambda0.0-0.0_lr1e-05-batch_size_32x4-2022-11-22
#MODEL_DIR=splade_distill_num1_kldiv_mrr_diff_denoiseFalse_num20_kldiv5-lambda0.0-0.0_lr1e-05-batch_size_32x4-2022-11-17
#MODEL_DIR=splade_distill_num1_klar-0.02-0.03_denoiseFalse_num20_5-lr1e-05-batch_size_32x4-2022-12-11
MODEL_DIR=splade_3V_distill_tokensparseTrue_marginmse_denoiseFalse_num20_marginmse-lambda0.08-0.001_lr1e-05-batch_size_32x4-2023-02-11

python rerank_eval_fast_splade_mv.py training_with_sentence_transformers/output/$MODEL_DIR/80000/0_MVMLMTransformer ../msmarco/wentai_splade_dev_top100.tsv ../msmarco/qrels.dev.tsv test.tsv 0 0
#python rerank_eval_fast_splade.py training_with_sentence_transformers/output/$MODEL_DIR/5600/0_MLMTransformer ../msmarco/wentai_splade_dev_top100.tsv ../msmarco/qrels.dev.tsv output_dev_splade_iter5600_klfr_gamma5.0_l1_64x2_0.0.tsv 0.03 0.43

