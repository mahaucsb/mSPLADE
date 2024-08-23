#!/bin/bash
#SBATCH --job-name="eval_beir"
#SBATCH --output="eval_beir_cq_ckl.%j.%N.out"
#SBATCH --error="eval_beir_cq_ckl.%j.%N.err"
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --no-requeue
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=32G
#SBATCH -t 24:00:00


source ~/.bashrc
source activate splade

#python eval_hybrid_beir.py ../msmarco/warmup_Splade_0_MLMTransformer ../msmarco/warmup_0_ColBERTTransformer 100000
#python eval_hybrid_beir.py ../msmarco/warmup_Splade_0_MLMTransformer cross-encoder/ms-marco-MiniLM-L-6-v2 100000 
#python eval_hybrid_beir.py training_with_sentence_transformers/output/splade_distill_num1_kldiv_position_focal_mrr_diff_gamma5.0-alpha1.0_denoiseFalse_num20_kldiv_position_focal5-lambda0.0-0.0_lr1e-05-batch_size_32x4-2022-11-24/25000/0_MLMTransformer training_with_sentence_transformers/output/colbert_splade_distill_num1_kldiv_position_focal_default_gamma5.0-alpha1.5_denoiseFalse_num20_kldiv_position_focal5-lr1e-05-batch_size_8x4-2022-11-01/100000/0_ColBERTTransformer 100000
#python eval_hybrid_beir.py training_with_sentence_transformers/output/splade_distill_num1_klar-0.01-0.015_denoiseFalse_num20_5-lr1e-05-batch_size_32x4-2022-12-15//4000/0_MLMTransformer training_with_sentence_transformers/output/colbert_splade_distill_num1_klar-0.02-0.03_denoiseFalse_num20_5-lr1e-05-batch_size_8x4-2022-12-10/30000/0_ColBERTTransformer 30000
#python eval_hybrid_beir.py training_with_sentence_transformers/output/splade_distill_num1_klar-0.01-0.015_denoiseFalse_num20_5-lr1e-05-batch_size_32x4-2022-12-15/4000/0_MLMTransformer training_with_sentence_transformers/output_ckl/colbert_splade_distill_num1_kldiv_default_alpha0.2_denoiseFalse_num20_kldiv5-lr1e-05-batch_size_8x4-2022-11-25/0_ColBERTTransformer 30000

#python eval_hybrid_beir.py training_with_sentence_transformers/output/splade_distill_num1_kldiv_position_focal_mrr_diff_gamma5.0-alpha1.0_denoiseFalse_num20_kldiv_position_focal5-lambda0.0-0.0_lr1e-05-batch_size_32x4-2022-11-24/25000/0_MLMTransformer training_with_sentence_transformers/output_ckl/output_ckl/colbert_splade_distill_num1_kldiv_default_alpha0.2_denoiseFalse_num20_kldiv5-lr1e-05-batch_size_8x4-2022-11-25/0_ColBERTTransformer 30000
#python eval_hybrid_beir.py training_with_sentence_transformers/output/splade_distill_num1_kldiv_position_focal_mrr_diff_gamma5.0-alpha1.0_denoiseFalse_num20_kldiv_position_focal5-lambda0.0-0.0_lr1e-05-batch_size_32x4-2022-11-24/25000/0_MLMTransformer training_with_sentence_transformers/output_ckl/output_ckl/colbert_splade_distill_num1_kldiv_ib_ib0.05_denoiseFalse_num20_kldiv_ib5-lr1e-05-batch_size_8x4-2022-11-07/110000/0_ColBERTTransformer 
#python eval_hybrid_beir.py training_with_sentence_transformers/output/splade_distill_num1_kldiv_position_focal_mrr_diff_gamma5.0-alpha1.0_denoiseFalse_num20_kldiv_position_focal5-lambda0.0-0.0_lr1e-05-batch_size_32x4-2022-11-24/25000/0_MLMTransformer training_with_sentence_transformers/output_ckl/output_ckl/colbert_splade_distill_num1_curriculum_default_alpha0.2_denoiseFalse_num20_curriculum5-lr1e-05-batch_size_8x4-2022-11-16/90000/0_ColBERTTransformer 
#MODEL_DIR=colbert_splade_distill_kldiv_multipos_focal_focal_gamma5.0_denoiseFalse_num20_kldiv_multipos_focal_focal5-lr1e-05-batch_size_16x2-2023-05-20
#python eval_hybrid_beir_cq.py training_with_sentence_transformers/output/splade_distill_num1_kldiv_position_focal_mrr_diff_gamma5.0-alpha1.0_denoiseFalse_num20_kldiv_position_focal5-lambda0.0-0.0_lr1e-05-batch_size_32x4-2022-11-24/25000/0_MLMTransformer training_with_sentence_transformers/output/$MODEL_DIR/30000/0_ColBERTTransformer 

#MODEL_DIR=colbert_splade_distill-128_kldiv_multipos-upw0.0_denoiseFalse_num20_kldiv_multipos5-lr1e-05-batch_size_16x2-2023-05-10
MODEL_DIR=colbert_splade_distill-128_kldiv_multipos_focal_gamma4.0-alpha0.0_denoiseFalse_num20_kldiv_multipos_focal5-lr1e-05-batch_size_16x2-2023-05-10

python eval_hybrid_beir_cq.py training_with_sentence_transformers/output/splade_distill_num1_kldiv_position_focal_mrr_diff_gamma5.0-alpha1.0_denoiseFalse_num20_kldiv_position_focal5-lambda0.0-0.0_lr1e-05-batch_size_32x4-2022-11-24/25000/0_MLMTransformer training_with_sentence_transformers/output/$MODEL_DIR

