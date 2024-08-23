#!/bin/bash
#SBATCH --job-name="tr_splade_updatemrr_sh6-1.0_batch32"
#SBATCH --output="tr_splade_updatemrr_sh6-1.0_batch32.%j.%N.out"
#SBATCH --error="tr_splade_updatemrr_sh6-1.0_batch32.%j.%N.err"
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --no-requeue
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=32G
#SBATCH -t 48:00:00

source ~/.bashrc
source activate splade
#/bin/bash -c "python -m train_splade_updatemrr --accum_iter 2 --continues --epochs 600000 --lambda_d 0.0 --lambda_q 0.0 --loss_type kldiv_multipos_position_focal --gamma 4.0 --alpha 1.0 --lr 1e-05 --model_name ../../msmarco/warmup_Splade_0_MLMTransformer --num_negs_per_system=20 --nway 5 --train_batch_size 16 --training_queries ../../msmarco/train_queries_distill_splade_colbert_0.json --warmup_steps=6000 --mrr_update 2000"
/bin/bash -c "python -m train_splade_updatemrr --accum_iter 2 --continues --epochs 600000 --lambda_d 0.0 --lambda_q 0.0 --loss_type kldiv_multipos_position_focal --gamma 6.0 --alpha 1.0 --lr 1e-05 --model_name output/splade_distill_num1_updatemrr2000_denoiseFalse_num20_kldiv_multipos_position_focal5-lambda0.0-0.0_gamma6.0-1.0_lr1e-05-batch_size_16x2-2023-01-13/18000/0_MLMTransformer --num_negs_per_system=20 --nway 5 --train_batch_size 16 --training_queries ../../msmarco/train_queries_distill_splade_colbert_0.json --warmup_steps=0 --mrr_update 2000"
