#!/bin/bash
#SBATCH -A csb176
#SBATCH --job-name="trcol_updatemrr"
#SBATCH --output="trcol_updatemrr-ib0.001.%j.%N.out"
#SBATCH --error="trcol_updatemrr-ib0.001.%j.%N.err"
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --no-requeue
#SBATCH --gpus=2
#SBATCH --mem-per-gpu=32G
#SBATCH -t 48:00:00

source activate splade
#/bin/bash -c "python -m train_colbert_ranker_with_eval --model_name ../../msmarco/warmup_0_ColBERTTransformer/ --train_batch_size 8 --accum_iter 4 --epochs 600000 --warmup_steps 6000 --loss_type klar --num_negs_per_system 20 --lr 1e-5 --continues --nway 5 --beta_p 0.02 --beta_n 0.03"
#/bin/bash -c "python -m train_colbert_ranker_with_eval --model_name Luyu/co-condenser-marco --train_batch_size 8 --accum_iter 4 --epochs 1000000 --warmup_steps 6000 --loss_type kldiv --num_negs_per_system 20 --lr 1e-5 --nway 5"
#/bin/bash -c "python -m train_colbert_ranker_with_eval --model_name Luyu/co-condenser-marco --train_batch_size 8 --accum_iter 4 --epochs 1000000 --warmup_steps 6000 --loss_type kldiv_ib --ib_p 0.05 --num_negs_per_system 20 --lr 2e-5 --nway 5 --denoise"
#/bin/bash -c "python -m train_colbert_ranker --model_name Luyu/co-condenser-marco --train_batch_size 8 --accum_iter 4 --epochs 1000000 --warmup_steps 6000 --loss_type kldiv_multipos_focal --num_negs_per_system 20 --lr 1e-5 --nway 5 --gamma 5.0"

#/bin/bash -c "python -m train_colbert_ranker_updatemrr --model_name ../../msmarco/warmup_0_ColBERTTransformer/ --train_batch_size 16 --accum_iter 2 --epochs 600000 --warmup_steps 6000 --loss_type kldiv_multipos_position_focal --num_negs_per_system 20 --lr 1e-5 --continues --nway 5 --gamma 4.0 --alpha 1.0 --mrr_update 2000"
/bin/bash -c "python -m train_colbert_ranker_updatemrr --model_name ../../msmarco/warmup_0_ColBERTTransformer/ --train_batch_size 32 --accum_iter 1 --epochs 600000 --warmup_steps 6000 --loss_type kldiv_multipos_ib_position_focal --num_negs_per_system 20 --lr 1e-5 --continues --nway 5 --gamma 5.0 --alpha 1.0 --ib_p 0.001 --mrr_update 2000"