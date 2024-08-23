#!/bin/bash
#SBATCH --job-name="tr_col_updatemrr_2k-5.0-1.0_ib0.001"
#SBATCH --output="tr_col_updatemrr_2k-5.0-1.0_ib0.001.%j.%N.out"
#SBATCH --error="tr_col_updatemrr_2k-5.0-1.0_ib0.001.%j.%N.err"
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=32G
#SBATCH -t 48:00:00

source ~/.bashrc
source activate splade

#/bin/bash -c "python -m train_colbert_ranker_updatemrr --model_name ../../msmarco/warmup_0_ColBERTTransformer/ --train_batch_size 16 --accum_iter 2 --epochs 500000 --warmup_steps 0 --loss_type kldiv_multipos_position_focal --num_negs_per_system 20 --lr 1e-5 --continues --nway 5 --gamma 5.0 --alpha 1.0 --mrr_update 2000"
#/bin/bash -c "python -m train_colbert_ranker_updatemrr --model_name ../../msmarco/warmup_0_ColBERTTransformer/ --train_batch_size 16 --accum_iter 2 --epochs 600000 --warmup_steps 6000 --loss_type kldiv_ib_position_focal --num_negs_per_system 20 --lr 1e-5 --continues --nway 5 --gamma 5.0 --alpha 1.0 --ib_p 0.1 --mrr_update 2000"

/bin/bash -c "python -m train_colbert_ranker_updatemrr --model_name ../../msmarco/warmup_0_ColBERTTransformer/ --train_batch_size 16 --accum_iter 2 --epochs 600000 --warmup_steps 6000 --loss_type kldiv_multipos_ib_position_focal --num_negs_per_system 20 --lr 1e-5 --continues --nway 5 --gamma 5.0 --alpha 1.0 --ib_p 0.001 --mrr_update 2000"