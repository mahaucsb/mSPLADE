#!/bin/bash
#SBATCH -A csb176
#SBATCH --job-name="tr_splade_summax"
#SBATCH --output="tr_splade_summax.%j.%N.out"
#SBATCH --error="tr_splade_summax.%j.%N.err"
#SBATCH --partition=gpu-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --no-requeue
#SBATCH --gpus=2
#SBATCH --mem-per-gpu=32G
#SBATCH -t 48:00:00

source activate splade
#/bin/bash -c "python -m train_splade_mv --accum_iter 4 --epochs 3000000 --loss_type marginmse  --lr 1e-05 --model_name Luyu/co-condenser-marco --num_negs_per_system=20 --train_batch_size 32 --warmup_steps=6000 --lambda_q 0.08 --lambda_d 0.001"
#/bin/bash -c "python -m train_splade_mv --accum_iter 8 --epochs 3000000 --loss_type marginmse  --lr 1e-05 --model_name Luyu/co-condenser-marco --num_negs_per_system=20 --train_batch_size 16 --warmup_steps=6000 --lambda_q 0.08 --lambda_d 0.001 --token_sparse --gumbel"
/bin/bash -c "python -m train_splade_summax --model_name Luyu/co-condenser-marco --train_batch_size 32 --accum_iter 4 --epochs 600000 --warmup_steps 6000 --loss_type marginmse --num_negs_per_system 20 --lr 2e-5 --nway 1 --lambda_q 0.0 --lambda_d 0.0"
