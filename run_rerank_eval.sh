#!/bin/bash
#SBATCH -A csb176
#SBATCH --job-name="rerankcol"
#SBATCH --output="rerankcol.%j.%N.out"
#SBATCH --error="rerankcol.%j.%N.err"
#SBATCH --partition=gpu-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --no-requeue
#SBATCH --gpus=1
#SBATCH -t 48:00:00

MODEL_DIR=../../../output/colbert_dense0.0_hn_denoiseFalse_marginmse_Luyu-co-condenser-marco-batch_size_128-2022-06-17_23-23-06
#MODEL_DIR=distilSplade_dense2_inbatch_0.01_0.008-splade10denoise-batchsize_64-denseweight_0.2-2022-06-12_11-40-13
#MODEL_DIR=distilSplade_dense2_0.01_0.008-splade10denoise-batchsize_64-denseweight_0.2-reverse3-2022-06-12_12-58-24

#/bin/bash -c "python rerank_eval.py training_with_sentence_transformers/output/$MODEL_DIR/281000/0_ColBERTTransformer ../msmarco/msmarco-passagetest2020-top1000.tsv ../msmarco/2020qrels-pass.txt output_20_1.run.json colbert"
#/bin/bash -c "python rerank_eval.py training_with_sentence_transformers/output/$MODEL_DIR/281000/0_ColBERTTransformer ../msmarco/msmarco-passagetest2019-top1000.tsv ../msmarco/2019qrels-pass.txt output_19_1.run.json colbert"
/bin/bash -c "python rerank_eval.py training_with_sentence_transformers/output/$MODEL_DIR/271000/0_ColBERTTransformer ../msmarco/top1000.dev ../msmarco/qrels.dev.tsv output_colbert_2.json colbert"
#/bin/bash -c "python rerank_eval.py training_with_sentence_transformers/output/$MODEL_DIR/195900/0_MLMTransformerDense ../msmarco/top1000.dev ../msmarco/qrels.dev.tsv output612.json splade_cls"
#/bin/bash -c "python rerank_eval.py training_with_sentence_transformers/output/$MODEL_DIR/220000/0_MLMTransformer ../msmarco/msmarco-passagetest2020-top1000.tsv ../msmarco/2020qrels-pass.txt output_20.run.json splade"
#/bin/bash -c "python rerank_eval.py training_with_sentence_transformers/output/$MODEL_DIR/220000/0_MLMTransformer ../msmarco/msmarco-passagetest2019-top1000.tsv ../msmarco/2019qrels-pass.txt output_19.run.json splade"
#/bin/bash -c "python rerank_eval.py training_with_sentence_transformers/output/$MODEL_DIR/279500/0_MLMTransformerDense ../msmarco/top1000.dev ../msmarco/qrels.dev.tsv output_64x1_0.01_0.088_150k_fromsplade_reverse3_cls_dev.run.json splade_cls"
##/bin/bash -c "python rerank_eval.py training_with_sentence_transformers/output/$MODEL_DIR/189000/0_MLMTransformer ../msmarco/wentai_splade_2020_top1000.tsv ../msmarco/2020qrels-pass.txt test.run.json splade"
##/bin/bash -c "python rerank_eval.py training_with_sentence_transformers/output/$MODEL_DIR/189000/0_MLMTransformer ../msmarco/wentai_splade_2019_top1000.tsv ../msmarco/2019qrels-pass.txt test.run.json splade"
##/bin/bash -c "python rerank_eval.py training_with_sentence_transformers/output/$MODEL_DIR/189000/0_MLMTransformer ../msmarco/wentai_splade_dev_top1000.tsv ../msmarco/qrels.dev.tsv output_inbatch_fromsplade_dev.run.json splade"
#/bin/bash -c "python rerank_eval.py training_with_sentence_transformers/output/$MODEL_DIR/250000/0_MLMTransformerDense ../msmarco/msmarco-passagetest2020-top1000.tsv ../msmarco/2020qrels-pass.txt output_64x1_0.01_0.088_100k_fromsplade_reverse3_cls_2020.run.json splade_cls"
#/bin/bash -c "python rerank_eval.py training_with_sentence_transformers/output/$MODEL_DIR/250000/0_MLMTransformerDense ../msmarco/msmarco-passagetest2019-top1000.tsv ../msmarco/2019qrels-pass.txt  output_64x1_0.01_0.088_200k_fromsplade_reverse3_cls_2019.run.json splade_cls"
#/bin/bash -c "python rerank_eval.py training_with_sentence_transformers/output/$MODEL_DIR/100000/0_MLMTransformerDense ../msmarco/msmarco-passagetest2020-top1000.tsv ../msmarco/2020qrels-pass.txt output_64x1_0.01_0.088_100k_fromsplade_reverse3_nocls_2020.run.json splade"
#/bin/bash -c "python rerank_eval.py training_with_sentence_transformers/output/$MODEL_DIR/100000/0_MLMTransformerDense ../msmarco/msmarco-passagetest2019-top1000.tsv ../msmarco/2019qrels-pass.txt  output_64x1_0.01_0.088_100k_fromsplade_reverse3_nocls_2019.run.json splade"
