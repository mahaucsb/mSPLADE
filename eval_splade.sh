#!/bin/bash
#SBATCH --job-name="eval_splade"
#SBATCH --output="eval_splade.%j.%N.out"
#SBATCH --error="eval_splade.%j.%N.err"
#SBATCH --partition=gpu
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=32G
#SBATCH -t 5:00:00

source ~/.bashrc
source activate splade

#MODEL_DIR=splade_distill_num1_updatemrr2000_denoiseFalse_num20_kldiv_multipos_position_focal5-lambda0.0-0.0_gamma5.0-1.0_lr1e-05-batch_size_16x2-2023-01-13
#for N in 16
#do 
#echo $N

#python rerank_eval_fast_splade.py training_with_sentence_transformers/output/$MODEL_DIR/${N}000/0_MLMTransformer ../msmarco/wentai_splade_dev_top100.tsv ../msmarco/qrels.dev.tsv output_dev_splade_updatemrr5-1.0_iter${N}000_16x2.tsv 0 0.67
#python rerank_eval_fast_splade.py training_with_sentence_transformers/output/$MODEL_DIR/5600/0_MLMTransformer ../msmarco/wentai_splade_dev_top100.tsv ../msmarco/qrels.dev.tsv output_dev_splade_iter5600_klfr_gamma5.0_l1_64x2_0.0.tsv 0.03 0.43
#MODEL_DIR=../msmarco/warmup_Splade_0_MLMTransformer

#python rerank_eval_splade_summax.py $MODEL_DIR ../msmarco/wentai_splade_dev_top100.tsv ../msmarco/qrels.dev.tsv output_dev_splade_warmup_max_sum.tsv 0 0

#yr=19
#python rerank_eval_fast_splade.py training_with_sentence_transformers/output/$MODEL_DIR/${N}000/0_MLMTransformer ../msmarco/wentai_splade_20${yr}_top1000.tsv ../msmarco/20${yr}qrels-pass.txt output_20${yr}_splade_updatemrr5-1.0_iter${N}000_16x2.tsv 0 0.67
#python rerank_eval_splade_summax.py $MODEL_DIR ../msmarco/wentai_splade_20${yr}_top1000.tsv ../msmarco/20${yr}qrels-pass.txt output_${yr}_splade_warmup_max_sum.tsv 0 0

#yr=20
#python rerank_eval_fast_splade.py training_with_sentence_transformers/output/$MODEL_DIR/${N}000/0_MLMTransformer ../msmarco/wentai_splade_20${yr}_top1000.tsv ../msmarco/20${yr}qrels-pass.txt output_20${yr}_splade_updatemrr5-1.0_iter${N}000_16x2.tsv 0 0.67
#done
#python rerank_eval_splade_summax.py $MODEL_DIR ../msmarco/wentai_splade_20${yr}_top1000.tsv ../msmarco/20${yr}qrels-pass.txt output_${yr}_splade_warmup_max_sum.tsv 0 0

#MODEL_DIR=distilSplade_fromcocondenser_summax_0.0_0.0_Luyu-co-condenser-marco-batch_size_64-2023-03-07_15-33-13
MODEL_DIR=distilSplade_fromcocondenser_meanmean_0.0_0.0_..-..-msmarco-warmup_Splade_0_MLMTransformer-batch_size_128-2023-04-11_14-26-46
#python rerank_eval_fast_splade_mv.py training_with_sentence_transformers/output/$MODEL_DIR/37600/0_MLMTransformer ../msmarco/wentai_splade_dev_top100.tsv ../msmarco/qrels.dev.tsv test.tsv 0 0
python rerank_eval_splade_summax.py training_with_sentence_transformers/output/$MODEL_DIR/60000/0_MLMTransformer ../msmarco/wentai_splade_dev_top100.tsv ../msmarco/qrels.dev.tsv test_splade_meanmean_60k.tsv 0 0
