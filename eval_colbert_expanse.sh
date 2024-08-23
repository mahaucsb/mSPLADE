#!/bin/bash
#SBATCH -A csb176
#SBATCH --job-name="eval_colbert"
#SBATCH --output="eval_colbert_128_ckl-5-1_newnorm_temp2.%j.%N.out"
#SBATCH --error="eval_colbert_128_ckl-5-1_newnorm_temp2.%j.%N.err"
#SBATCH --partition=gpu-shared
#SBATCH --no-requeue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=32G
#SBATCH -t 2:00:00


source ~/.bashrc
source activate splade


#MODEL_DIR=colbert_splade_distill_num1_updatemrr1000_kldiv_multipos_position_focal_default_alpha1.5_denoiseFalse_num20_kldiv_multipos_position_focal5-lr1e-05-batch_size_16x2-2023-01-07
#MODEL_DIR=colbert_splade_distill_num1_updatemrr2000_kldiv_multipos_position_focal_default_alpha1.0_denoiseFalse_num20_kldiv_multipos_position_focal5-lr1e-05-batch_size_16x2-2023-01-07
#MODEL_DIR=colbert_splade_distill_num1_updatemrr2000_kldiv_multipos_position_focal_default_gamma4.0-alpha1.0_denoiseFalse_num20_kldiv_multipos_position_focal5-lr1e-05-batch_size_16x2-2023-01-09

#MODEL_DIR=colbert_\$768_wmp_ce_default_alpha0.0_denoiseFalse_num20_ce5-lr2e-05-batch_size_8x4-2023-05-04
#MODEL_DIR=colbert_\$768_distill_wmp_kldiv-upw0.0_denoiseFalse_num20_kldiv5-lr2e-05-batch_size_8x4-2023-05-04
#MODEL_DIR=colbert_\$768_distill_simlm_wmp_kldiv-upw0.0_denoiseFalse_num20_kldiv5-lr2e-05-batch_size_8x4-2023-05-04
MODEL_DIR=colbert_splade_distill-128_ckl_gamma5.0-alpha1.0-temp2.0_denoiseFalse_num20_ckl5-lr1e-05-batch_size_8x4-2023-07-03
NAME=dim128_num1_ckl-5-1-temp2_newnorm
for N in 30
do 
echo $N

python rerank_eval_fast.py training_with_sentence_transformers/output/$MODEL_DIR/${N}000/0_ColBERTTransformer  ../msmarco/splade_klfr_5-1_num1.dev.top100.trec.tsv ../msmarco/qrels.dev.tsv output_dev_colbert_${NAME}_${N}k.tsv 128
#../msmarco/effi.dev.trec.tsv
#python rerank_eval_fast.py training_with_sentence_transformers/output/$MODEL_DIR/${N}000/0_ColBERTTransformer ../msmarco/splade_klfr_5-1_num1.dev.top100.trec.tsv ../msmarco/qrels.dev.tsv output_dev_colbert_splade_updatemrr_2k_sh5-1.0_${N}k.tsv
#python rerank_eval_fast.py training_with_sentence_transformers/output/$MODEL_DIR/${N}000/0_ColBERTTransformer ../msmarco/splade_klfr_5-1_num1.dev.top100.trec.tsv ../msmarco/qrels.dev.tsv output_dev_colbert_splade-new-updatemrr_ckl-ib-0.1_${N}k.tsv

yr=19
#python rerank_eval_fast.py training_with_sentence_transformers/output/$MODEL_DIR/${N}000/0_ColBERTTransformer ../msmarco/effi_V_${yr}.trec.tsv ../msmarco/20${yr}qrels-pass.txt output_20${yr}_colbert_${NAME}_${N}k.tsv
python rerank_eval_fast.py training_with_sentence_transformers/output/$MODEL_DIR/${N}000/0_ColBERTTransformer ../msmarco/splade_klfr_5-1_num1.2019.top1000.trec.tsv ../msmarco/20${yr}qrels-pass.txt output_20${yr}_colbert_${NAME}_${N}k.tsv 128
#python rerank_eval_fast.py training_with_sentence_transformers/output/$MODEL_DIR/${N}000/0_ColBERTTransformer ../msmarco/splade_klfr_5-1_num1.2019.top1000.trec.tsv ../msmarco/20${yr}qrels-pass.txt output_20${yr}_colbert_splade_updatemrr_2k_sh5-1.0_${N}k.tsv
#python rerank_eval_fast.py training_with_sentence_transformers/output/$MODEL_DIR/${N}000/0_ColBERTTransformer ../msmarco/splade_klfr_5-1_num1.2019.top100.trec.tsv ../msmarco/20${yr}qrels-pass.txt output_20${yr}_colbert_splade-new-updatemrr_ckl-ib-0.1_${N}k.tsv

yr=20
python rerank_eval_fast.py training_with_sentence_transformers/output/$MODEL_DIR/${N}000/0_ColBERTTransformer ../msmarco/splade_klfr_5-1_num1.2020.top1000.trec.tsv ../msmarco/20${yr}qrels-pass.txt output_20${yr}_colbert${NAME}_${N}k.tsv 128

#python rerank_eval_fast.py training_with_sentence_transformers/output/$MODEL_DIR/${N}000/0_ColBERTTransformer ../msmarco/effi_V_${yr}.trec.tsv ../msmarco/20${yr}qrels-pass.txt output_20${yr}_colbert${NAME}_${N}k.tsv
#python rerank_eval_fast.py training_with_sentence_transformers/output/$MODEL_DIR/${N}000/0_ColBERTTransformer ../msmarco/splade_klfr_5-1_num1.2020.top1000.trec.tsv ../msmarco/20${yr}qrels-pass.txt output_20${yr}_colbert_splade_updatemrr_2k_sh5-1.0_${N}k.tsv
#python rerank_eval_fast.py training_with_sentence_transformers/output/$MODEL_DIR/${N}000/0_ColBERTTransformer ../msmarco/splade_klfr_5-1_num1.2020.top100.trec.tsv ../msmarco/20${yr}qrels-pass.txt output_20${yr}_colbert_splade-new-updatemrr_ckl-ib-0.1_${N}k.tsv

done
