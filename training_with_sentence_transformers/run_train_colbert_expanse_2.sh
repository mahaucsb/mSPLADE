#!/bin/bash
#SBATCH -A csb176
#SBATCH --job-name="trcol_cont_128"
#SBATCH --output="trcol_cont_128_gamma5.%j.%N.out"
#SBATCH --error="trcol_cont_128_gamma5.%j.%N.err"
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
#/bin/bash -c "python -m train_colbert_ranker --model_name output/colbert_splade_distill_num1-simlm-round2_kldiv_ib_ib0.05_denoiseFalse_num20_kldiv_ib5-lr1e-05-batch_size_8x4-2023-01-07/4000/0_ColBERTTransformer/ --train_batch_size 16 --accum_iter 2 --epochs 992000 --warmup_steps 0 --loss_type kldiv_ib --ib_p 0.05 --num_negs_per_system 20 --lr 1e-5 --nway 5 --continues"
/bin/bash -c "python -m train_colbert_ranker --model_name output/colbert_warmup_128_num20_kldiv_ib5-lr1e-05-batch_size_32x1-2023-04-15/232000/0_ColBERTTransformer/ --train_batch_size 32 --accum_iter 1 --epochs 1000000 --warmup_steps 6000 --loss_type kldiv_multipos_focal --num_negs_per_system 20 --lr 1e-5 --nway 5 --continues --gamma 5.0 --alpha 1.0"
#
#/bin/bash -c "python -m train_colbert_ranker --model_name ../../msmarco/warmup_0_ColBERTTransformer/ --train_batch_size 8 --accum_iter 4 --epochs 600000 --warmup_steps 6000 --loss_type kldiv_multipos_focal --num_negs_per_system 20 --lr 1e-5 --continues --nway 5 --gamma 5.0 --alpha 0.0 --beta 1.0"

python -m train_cq_rank --model_name output/colbert_warmup_128_num20_kldiv_ib5-lr1e-05-batch_size_32x1-2023-04-15/232000/0_ColBERTTransformer/ --train_batch_size 16 --accum_iter 2 --epochs 1000000 --warmup_steps 6000 --loss_type kldiv  --num_negs_per_system 20 --lr 1e-5 --nway 5 --dim 128