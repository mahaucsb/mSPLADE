#!/bin/bash
#SBATCH --job-name="tsplade"
#SBATCH --output="trsplade_kllog0.02.%j.%N.out"
#SBATCH --error="trsplade_kllog0.02.%j.%N.err"
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --gres=gpu:4
#SBATCH -t 48:00:00

source ~/.bashrc
source activate splade
#/bin/bash -c "python -m train_splade --model_name Luyu/co-condenser-marco --train_batch_size 128 --accum_iter 1 --epochs 600000 --warmup_steps 6000 --loss_type marginmse --num_negs_per_system 20 --lr 2e-5 --nway 1 --lambda_q 0.0 --lambda_d 0.0"

#/bin/bash -c "python -m train_splade --model_name ../../msmarco/warmup_Splade_0_MLMTransformer --train_batch_size 32 --accum_iter 4 --epochs 600000 --warmup_steps 6000 --loss_type kldiv_focal --gamma 2.0 --num_negs_per_system 20 --lr 1e-5 --nway 5 --continues --training_queries ../../msmarco/train_queries_distill_splade_colbert_0.json --lambda_q 0.0 --lambda_d 0.0"
#/bin/bash -c "python -m train_splade --model_name ../../msmarco/warmup_Splade_0_MLMTransformer --train_batch_size 128 --accum_iter 1 --epochs 600000 --warmup_steps 6000 --loss_type marginmse --num_negs_per_system 20 --continues --training_queries ../../msmarco/train_queries_distill_splade_colbert_0.json --lambda_q 0.0 --lambda_d 0.0"
#/bin/bash -c "python -m train_splade --model_name ../../msmarco/warmup_Splade_0_MLMTransformer --train_batch_size 32 --accum_iter 4 --epochs 600000 --warmup_steps 6000 --loss_type kldiv_position_focal --gamma 5.0 --alpha 1.5 --num_negs_per_system 20 --lr 1e-5 --nway 5 --continues --training_queries ../../msmarco/train_queries_distill_splade_colbert_0.json --lambda_q 0.0 --lambda_d 0.0"

#/bin/bash -c "python -m train_splade_summax --model_name Luyu/co-condenser-marco --train_batch_size 128 --accum_iter 1 --epochs 600000 --warmup_steps 6000 --loss_type marginmse --num_negs_per_system 20 --lr 2e-5 --nway 1 --lambda_q 0.0 --lambda_d 0.0"
#/bin/bash -c "python -m train_splade_summax --model_name ../../msmarco/warmup_Splade_0_MLMTransformer --train_batch_size 128 --accum_iter 1 --epochs 600000 --warmup_steps 6000 --loss_type marginmse --num_negs_per_system 20 --lr 1e-5 --nway 1 --lambda_q 0.0 --lambda_d 0.0"

#/bin/bash -c "python -m train_splade --accum_iter 4 --continues --epochs 3000000 --lambda_d 0.0 --lambda_q 0.0 --beta_p 0.01 --loss_type klar --lr 1e-05 --model_name ../../msmarco/warmup_Splade_0_MLMTransformer --num_negs_per_system=20 --nway 5 --train_batch_size 32 --training_queries ../../msmarco/train_queries_distill_splade_colbert_0.json --warmup_steps=6000 --sample_upweight 1"
#/bin/bash -c "python -m train_splade --accum_iter 4 --continues --epochs 3000000 --lambda_d 0.0 --lambda_q 0.0 --loss_type kldiv_multipos --lr 1e-05 --model_name ../../msmarco/warmup_Splade_0_MLMTransformer --num_negs_per_system=20 --nway 5 --train_batch_size 32 --training_queries ../../msmarco/train_queries_distill_splade_colbert_0.json --warmup_steps=6000 --sample_upweight 1"
bin/bash -c "python -m train_splade --accum_iter 4 --continues --epochs 3000000 --lambda_d 0.0 --lambda_q 0.0 --loss_type kllog --lr 1e-05 --model_name ../../msmarco/warmup_Splade_0_MLMTransformer --num_negs_per_system=20 --nway 5 --train_batch_size 32 --training_queries ../../msmarco/train_queries_distill_splade_colbert_0.json --warmup_steps=6000 --beta_p 0.02"
