#!/bin/bash
#SBATCH --job-name="eval_colbert"
#SBATCH --output="eval_colbert_from40k_klar4.0-0.0_iters_newnorm_128.%j.%N.out"
#SBATCH --error="eval_colbert_from40k_klar4.0-0.0_iters_newnorm_128.%j.%N.err"
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --no-requeue
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=32G
#SBATCH -t 7:00:00

#"eval_col_kl_neg16_num1_25k.%j.%N.out"

source ~/.bashrc
source activate splade


#MODEL_DIR=colbert_splade_distill_num1_kldiv_multipos_focal_gamma5.0-alpha0.0_denoiseFalse_num50_kldiv_multipos_focal16-lr1e-05-batch_size_8x4-2023-02-02
#MODEL_DIR=colbert_splade_distill_num1_kldiv_denoiseFalse_num50_kldiv16-lr1e-05-batch_size_8x4-2023-02-02
#MODEL_DIR=colbert_splade_wmp-128_kldiv-upw0.0_denoiseFalse_num20_kldiv5-lr2e-05-batch_size_16x2-2023-05-09
#MODEL_DIR=colbert_splade_wmp-768_kldiv-upw0.0_denoiseFalse_num20_kldiv5-lr2e-05-batch_size_16x2-2023-05-09
#MODEL_DIR=colbert_splade_wmp-128_kldiv_multipos_focal_gamma1.5-alpha0.5_denoiseFalse_num20_kldiv_multipos_focal5-lr2e-05-batch_size_16x2-2023-05-10
MODEL_DIR=colbert_splade_distill-40k-128_kldiv_multipos_focal_gamma4.0-alpha0.0_denoiseFalse_num20_kldiv_multipos_focal5-lr1e-05-batch_size_16x2-2023-05-14
for N in 30 40 50 60 70 75
do 
echo $N
python rerank_eval_fast.py training_with_sentence_transformers/output/$MODEL_DIR/${N}000/0_ColBERTTransformer ../msmarco/wentai_splade_dev_top100.tsv ../msmarco/qrels.dev.tsv output_dev_colbert_from40k_klar4.0-0.0_newnorm_128_${N}k.tsv 128

#python rerank_eval_fast_summax.py $MODEL_DIR ../msmarco/wentai_splade_dev_top100.tsv ../msmarco/qrels.dev.tsv test.tsv #  output_dev_col_kl_neg16_num1_25k.tsv
#python rerank_eval_fast.py training_with_sentence_transformers/output/$MODEL_DIR/${N}000/0_ColBERTTransformer ../msmarco/splade_klfr_5-1_num1.dev.top100.trec.tsv ../msmarco/qrels.dev.tsv output_dev_colbert_splade_updatemrr_2k_sh5-1.0_${N}k.tsv
#python rerank_eval_fast.py training_with_sentence_transformers/output/$MODEL_DIR/${N}000/0_ColBERTTransformer ../msmarco/splade_klfr_5-1_num1.dev.top100.trec.tsv ../msmarco/qrels.dev.tsv output_dev_colbert_splade-new-updatemrr_ckl-ib-0.1_${N}k.tsv

yr=19
python rerank_eval_fast.py training_with_sentence_transformers/output/$MODEL_DIR/${N}000/0_ColBERTTransformer ../msmarco/wentai_splade_20${yr}_top1000.tsv ../msmarco/20${yr}qrels-pass.txt  output_20${yr}_colbert_from40k_klar4.0-0.0_newnorm_128_${N}k.tsv 128 # output_20${yr}_col_kl_neg16_num1_25k.tsv
#python rerank_eval_fast.py training_with_sentence_transformers/output/$MODEL_DIR/${N}000/0_ColBERTTransformer ../msmarco/splade_klfr_5-1_num1.2019.top1000.trec.tsv ../msmarco/20${yr}qrels-pass.txt output_20${yr}_colbert_splade_updatemrr_2k_sh5-1.0_${N}k.tsv
#python rerank_eval_fast.py training_with_sentence_transformers/output/$MODEL_DIR/${N}000/0_ColBERTTransformer ../msmarco/splade_klfr_5-1_num1.2019.top100.trec.tsv ../msmarco/20${yr}qrels-pass.txt output_20${yr}_colbert_splade-new-updatemrr_ckl-ib-0.1_${N}k.tsv

yr=20
python rerank_eval_fast.py training_with_sentence_transformers/output/$MODEL_DIR/${N}000/0_ColBERTTransformer ../msmarco/wentai_splade_20${yr}_top1000.tsv ../msmarco/20${yr}qrels-pass.txt output_20${yr}_colbert_from40k_klar4.0-0.0_newnorm_128_${N}k.tsv 128
#python rerank_eval_fast.py training_with_sentence_transformers/output/$MODEL_DIR/${N}000/0_ColBERTTransformer ../msmarco/splade_klfr_5-1_num1.2020.top1000.trec.tsv ../msmarco/20${yr}qrels-pass.txt output_20${yr}_colbert_splade_updatemrr_2k_sh5-1.0_${N}k.tsv
#python rerank_eval_fast.py training_with_sentence_transformers/output/$MODEL_DIR/${N}000/0_ColBERTTransformer ../msmarco/splade_klfr_5-1_num1.2020.top100.trec.tsv ../msmarco/20${yr}qrels-pass.txt output_20${yr}_colbert_splade-new-updatemrr_ckl-ib-0.1_${N}k.tsv

done
