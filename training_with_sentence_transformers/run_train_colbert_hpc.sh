#!/bin/bash
#SBATCH --job-name="tr_col_focal"
#SBATCH --output="tr_col_focal.%j.%N.out"
#SBATCH --error="tr_col_focal.%j.%N.err"
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=32G
#SBATCH -t 12:00:00


source ~/.bashrc
source activate splade
#/bin/bash -c "python -m train_colbert_ranker --model_name ../../msmarco/warmup_0_ColBERTTransformer/ --train_batch_size 8 --accum_iter 4 --epochs 600000 --warmup_steps 6000 --loss_type klar --num_negs_per_system 20 --lr 1e-5 --continues --nway 5 --beta_p 0.015 --beta_n 0.0225"

#/bin/bash -c "python -m train_colbert_ranker --model_name ../../msmarco/warmup_0_ColBERTTransformer/ --train_batch_size 8 --accum_iter 4 --epochs 600000 --warmup_steps 6000 --loss_type kldiv_multipos_focal --num_negs_per_system 20 --lr 1e-5 --continues --nway 5 --gamma 5.0 --alpha 0.0 --beta 0.5"

#/bin/bash -c "python -m train_colbert_ranker --model_name Luyu/co-condenser-marco --train_batch_size 16 --accum_iter 2 --epochs 1000000 --warmup_steps 6000 --loss_type kldiv_multipos_focal --gamma 2.0 --num_negs_per_system 20 --lr 2e-5 --nway 5"


#/bin/bash -c "python -m train_colbert_ranker --model_name Luyu/co-condenser-marco --train_batch_size 16 --accum_iter 2 --epochs 1000000 --warmup_steps 6000 --loss_type kldiv --num_negs_per_system 20 --lr 2e-5 --nway 5 --negs_to_use bm25"

#/bin/bash -c "python -m train_colbert_ranker --model_name ../../msmarco/warmup_0_ColBERTTransformer/ --train_batch_size 8 --accum_iter 4 --epochs 2400000 --warmup_steps 6000 --loss_type kldiv_multipos_focal --gamma 9.0 --num_negs_per_system 50 --lr 1e-5 --nway 16 --continues"

#/bin/bash -c "python -m train_colbert_ranker --model_name ../../msmarco/warmup_0_ColBERTTransformer/ --train_batch_size 8 --accum_iter 4 --epochs 2400000 --warmup_steps 6000 --loss_type kldiv_multipos --num_negs_per_system 20 --lr 1e-5 --nway 5 --continues --sample_upweight 1.0"
#/bin/bash -c "python -m train_colbert_ranker --model_name ../../msmarco/warmup_0_ColBERTTransformer/ --train_batch_size 16 --accum_iter 2 --epochs 2400000 --warmup_steps 6000 --loss_type klar --num_negs_per_system 20 --lr 1e-5 --continues --nway 5 --sample_upweight 1.0 --beta_p 0.02"
#/bin/bash -c "python -m train_colbert_ranker --model_name ../../msmarco/warmup_0_ColBERTTransformer/ --train_batch_size 16 --accum_iter 2 --epochs 2400000 --warmup_steps 6000 --loss_type klar --num_negs_per_system 20 --lr 1e-5 --continues --nway 5 --sample_upweight 0.5 --beta_p 0.02"


#/bin/bash -c "python -m train_colbert_ranker --model_name Luyu/co-condenser-marco --train_batch_size 16 --accum_iter 2 --epochs 1000000 --warmup_steps 6000 --loss_type kldiv_multipos_focal --num_negs_per_system 20 --lr 2e-5 --nway 5 --dim 128 --gamma 1.5 --alpha 0.5"
/bin/bash -c "python -m train_colbert_ranker --model_name output/colbert_splade_wmp-128_kldiv-upw0.0_denoiseFalse_num20_kldiv5-lr2e-05-batch_size_16x2-2023-05-09/20000/0_ColBERTTransformer/ --train_batch_size 16 --accum_iter 2 --epochs 1000000 --warmup_steps 6000 --loss_type kldiv_multipos_focal --num_negs_per_system 20 --lr 1e-5 --nway 5 --dim 128 --gamma 1.5 --alpha 0.1 --continues"
#/bin/bash -c "python -m train_colbert_ranker --model_name output/colbert_splade_wmp-128_kldiv-upw0.0_denoiseFalse_num20_kldiv5-lr2e-05-batch_size_16x2-2023-05-09/100000/0_ColBERTTransformer/ --train_batch_size 16 --accum_iter 2 --epochs 1000000 --warmup_steps 6000 --loss_type kldiv_multipos --num_negs_per_system 20 --lr 1e-5 --nway 5 --dim 128 --continues"

