#!/bin/bash
#SBATCH --job-name="splade_eval_kldiv_l1_32x4_65k"
#SBATCH --output="splade_eval_kldiv_l1_32x4_65k.%j.out"
#SBATCH --error="splade_eval_kldiv_l1_32x4_65k.%j.err"
#SBATCH --partition="gpu-shared"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --export=ALL
#SBATCH --account=csb175
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=16G
#SBATCH -t 01:30:00

#YR=19
MODEL_DIR=splade_distill_num1_kldiv_mrr_diff_denoiseFalse_num20_kldiv5-lambda0.01-0.008_lr2e-05-batch_size_32x4-2022-11-06
#python rerank_eval_fast_splade.py training_with_sentence_transformers/output/$MODEL_DIR/${n}000/0_MLMTransformer ../msmarco/run.msmarco-passage.ance.bf.20${YR}.pair.tsv ../msmarco/20${YR}qrels-pass.txt test.run.json splade 0.0
#YR=20
#python rerank_eval_fast_splade.py training_with_sentence_transformers/output/$MODEL_DIR/${n}000/0_MLMTransformer ../msmarco/run.msmarco-passage.ance.bf.20${YR}.pair.tsv ../msmarco/20${YR}qrels-pass.txt test.run.json splade 0.0
n=40
python rerank_eval_fast_splade.py training_with_sentence_transformers/output/$MODEL_DIR/${n}000/0_MLMTransformer ../msmarco/wentai_splade_dev_top100.tsv ../msmarco/qrels.dev.tsv output_dev_splade_iter65_kldiv_l1_32x4_1e-5.tsv splade 0.0