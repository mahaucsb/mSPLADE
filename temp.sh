#!/bin/bash
#SBATCH -A csb176
#SBATCH --job-name="temp"
#SBATCH --output="temp.%j.%N.out"
#SBATCH --error="temp.%j.%N.err"
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --no-requeue
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=32G
#SBATCH -t 4:00:00

MODEL_DIR=colbert_splade_distill_num1_kldiv_multipos_default_alpha0.2_denoiseFalse_num20_kldiv_multipos5-lr1e-05-batch_size_8x4-2022-11-09
source activate splade
for ((n=1000; n<=100000; n=n+1000))
do 
      python rerank_eval_fast.py ../splade_cls_backup/all_tar/splade_cls/training_with_sentence_transformers/output/${MODEL_DIR}/${n}/0_ColBERTTransformer ../msmarco/2019_2020_rel0_data.tsv  ../msmarco/trec_qrels.txt output_trec_rel0_multipos_colbert_kldiv_iter${n}.tsv
     #python rerank_eval_fast.py training_with_sentence_transformers/output/colbert_splade_distill_num1_kldiv_multipos_default_alpha0.2_denoiseFalse_num20_kldiv_multipos5-lr1e-05-batch_size_8x4-2022-11-09/${n}/0_ColBERTTransformer ../msmarco/2019_2020_qrels2_data.tsv  ../msmarco/trec_qrels.txt output_trec_rel2_multipos_colbert_kldiv_iter${n}.tsv
done
