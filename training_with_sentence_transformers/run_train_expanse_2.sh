#!/bin/bash
#SBATCH -A csb176
#SBATCH --job-name="tr_splade_klmulti_0.01"
#SBATCH --output="tr_splade_klmulti_0.01.%j.%N.out"
#SBATCH --error="tr_splade_klmulti_0.01.%j.%N.err"
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --no-requeue
#SBATCH --gpus=4
#SBATCH --mem-per-gpu=32G
#SBATCH -t 15:00:00

source activate splade
#/bin/bash -c "python -m train_splade --accum_iter 4 --continues --epochs 3000000 --lambda_d 0.0 --lambda_q 0.0 --loss_type kldiv --lr 1e-05 --model_name ../../msmarco/warmup_Splade_0_MLMTransformer --num_negs_per_system=20 --nway 5 --train_batch_size 32 --training_queries ../../msmarco/train_queries_distill_splade_colbert_0.json --warmup_steps=6000"
#/bin/bash -c "python -m train_splade --model_name ../../msmarco/warmup_Splade_0_MLMTransformer --train_batch_size 32 --accum_iter 4 --epochs 600000 --warmup_steps 6000 --loss_type kldiv_focal --gamma 5.0 --num_negs_per_system 20 --lr 1e-5 --nway 5 --continues --training_queries ../../msmarco/train_queries_distill_splade_colbert_0.json --lambda_q 0.0 --lambda_d 0.0"
#/bin/bash -c "python -m train_splade --accum_iter 4 --continues --epochs 3000000 --lambda_d 0.0 --lambda_q 0.0 --beta_p 0.01 --beta_n 0.015 --loss_type klar --lr 1e-05 --model_name ../../msmarco/warmup_Splade_0_MLMTransformer --num_negs_per_system=20 --nway 5 --train_batch_size 32 --training_queries ../../msmarco/train_queries_distill_splade_colbert_0.json --warmup_steps=6000 --sample_upweight 1"

/bin/bash -c "python -m train_splade --accum_iter 4 --continues --epochs 3000000 --lambda_d 0.0 --lambda_q 0.0 --beta_p 0.01 --loss_type kllog --lr 1e-05 --model_name ../../msmarco/warmup_Splade_0_MLMTransformer --num_negs_per_system=20 --nway 5 --train_batch_size 32 --training_queries ../../msmarco/train_queries_distill_splade_colbert_0.json --warmup_steps=6000"
#/bin/bash -c "python -m train_splade --accum_iter 4 --continues --epochs 2900000 --lambda_d 0.0 --lambda_q 0.0 --beta_p 0.01 --loss_type kllog --lr 1e-05 --model_name output/distilSplade_fromcocondenser_0.0_0.0_..-..-msmarco-warmup_Splade_0_MLMTransformer-batch_size_32-2023-04-27_18-31-26/15200/0_MLMTransformer --num_negs_per_system=20 --nway 5 --train_batch_size 32 --training_queries ../../msmarco/train_queries_distill_splade_colbert_0.json --warmup_steps=0"
