#!/bin/bash
#SBATCH -A csb180
#SBATCH --job-name="eval_ce"
#SBATCH --output="eval_ce.%j.%N.out"
#SBATCH --error="eval_ce.%j.%N.err"
#SBATCH --partition=gpu-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --no-requeue
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=32G
#SBATCH -t 8:00:00

source activate splade
#/bin/bash -c "python -m train_splade --model_name ../../msmarco/warmup_Splade_0_MLMTransformer --train_batch_size 32 --accum_iter 4 --epochs 600000 --warmup_steps 6000 --loss_type kldiv_ib --num_negs_per_system 20 --lr 1e-5 --nway 5 --continues --training_queries ../../msmarco/train_queries_distill_splade_colbert_0.json"
#/bin/bash -c "python -m train_colbert_ranker --model_name ../../msmarco/warmup_0_ColBERTTransformer --train_batch_size 8 --accum_iter 4 --epochs 600000 --warmup_steps 6000 --loss_type kldiv_position_focal --num_negs_per_system 20 --cls_weight 0.0 --lr 1e-5 --nway 16 --gamma 5.0 --alpha 1.5 --continues"

/bin/bash -c "python -m rerank_eval_teacher"