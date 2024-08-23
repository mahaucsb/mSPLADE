#!/bin/bash
#SBATCH -A csb176
#SBATCH --job-name="trcol_num1_128_ckl5-1.5-temp2_newnorm"
#SBATCH --output="trcol_num1_128_ckl5-1.5-temp2_newnorm.%j.%N.out"
#SBATCH --error="trcol_num1_128_ckl5-1.5-temp2_newnorm.%j.%N.err"
#SBATCH --partition=gpu-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --no-requeue
#SBATCH --gpus=1
#SBATCH --mem=128G
#SBATCH -t 12:00:00

source activate splade
#/bin/bash -c "python -m train_colbert_ranker_with_eval --model_name ../../msmarco/warmup_0_ColBERTTransformer/ --train_batch_size 8 --accum_iter 4 --epochs 600000 --warmup_steps 6000 --loss_type kldiv_multipos --num_negs_per_system 20 --lr 1e-5 --continues --nway 5"
#/bin/bash -c "python -m train_colbert_ranker_with_eval --model_name Luyu/co-condenser-marco --train_batch_size 8 --accum_iter 4 --epochs 1000000 --warmup_steps 6000 --loss_type kldiv --num_negs_per_system 20 --lr 1e-5 --nway 5"
#/bin/bash -c "python -m train_colbert_ranker --model_name output/colbert_splade_distill_wmp_kldiv_ib_ib0.05_denoiseTrue_num20_kldiv_ib5-lr2e-05-batch_size_8x4-2022-12-29/15000/0_ColBERTTransformer/ --train_batch_size 8 --accum_iter 4 --epochs 1000000 --warmup_steps 6000 --loss_type kldiv_ib --ib_p 0.05 --num_negs_per_system 20 --lr 1e-5 --nway 5 --denoise"
#/bin/bash -c "python -m train_colbert_ranker --model_name Luyu/co-condenser-marco --train_batch_size 8 --accum_iter 4 --epochs 1000000 --warmup_steps 6000 --loss_type kldiv_ib --num_negs_per_system 20 --lr 1e-5 --nway 5 --gamma 5.0"

#/bin/bash -c "python -m train_colbert_ranker --model_name ../../msmarco/warmup_0_ColBERTTransformer/ --train_batch_size 8 --accum_iter 4 --epochs 600000 --warmup_steps 6000 --loss_type kldiv_multipos_focal --num_negs_per_system 20 --lr 1e-5 --continues --nway 5 --gamma 5.0 --alpha 0.0 --beta 1.0"

#/bin/bash -c "python -m train_colbert_ranker --model_name ../../msmarco/warmup_0_ColBERTTransformer/ --train_batch_size 8 --accum_iter 4 --epochs 600000 --warmup_steps 6000 --loss_type kldiv --num_negs_per_system 20 --lr 1e-5 --continues --nway 5"
#/bin/bash -c "python -m train_colbert_ranker --model_name ../../msmarco/warmup_0_ColBERTTransformer/ --train_batch_size 8 --accum_iter 4 --epochs 600000 --warmup_steps 6000 --loss_type kldiv_ib --ib_p 0.05 --num_negs_per_system 20 --lr 1e-5 --continues --nway 5"
#/bin/bash -c "python -m train_colbert_ranker --model_name ../../msmarco/warmup_0_ColBERTTransformer/ --train_batch_size 8 --accum_iter 4 --epochs 600000 --warmup_steps 6000 --loss_type kldiv_focal --gamma 0.5 --num_negs_per_system 20 --lr 1e-5 --continues --nway 5"
#STARTING=colbert_splade_distill_num1-simlm_kldiv_ib_ib0.05_denoiseFalse_num20_kldiv_ib5-lr1e-05-batch_size_8x4-2023-01-07
#/bin/bash -c "python -m train_colbert_ranker --model_name output/colbert_splade_distill_num1-simlm_kldiv_ib_ib0.05_denoiseFalse_num20_kldiv_ib5-lr1e-05-batch_size_8x4-2023-01-07/4000/0_ColBERTTransformer/ --train_batch_size 8 --accum_iter 4 --epochs 600000 --warmup_steps 1000 --loss_type kldiv_focal --gamma 5 --num_negs_per_system 20 --lr 1e-5 --continues --nway 5"
#/bin/bash -c "python -m train_colbert_ranker --model_name output/colbert_splade_distill_num1-simlm-round4_kldiv_ib_ib0.05_denoiseFalse_num20_kldiv_ib5-lr9e-06-batch_size_16x2-2023-01-08/10000/0_ColBERTTransformer/ --train_batch_size 16 --accum_iter 2 --epochs 572000 --warmup_steps 0 --loss_type kldiv_ib --ib_p 0.05 --num_negs_per_system 20 --lr 9e-6 --continues --nway 5"


#/bin/bash -c "python -m train_colbert_ranker --model_name ../../msmarco/warmup_0_ColBERTTransformer/ --train_batch_size 16 --accum_iter 2 --epochs 600000 --warmup_steps 6000 --loss_type klar --num_negs_per_system 20 --lr 1e-5 --continues --nway 5 --beta_p 0.02 --beta_n 0.03 --sample_upweight 0.1"
#/bin/bash -c "python -m train_colbert_ranker --model_name output/colbert_warmup_128_num20_kldiv_ib5-lr1e-05-batch_size_32x1-2023-04-15/232000/0_ColBERTTransformer --train_batch_size 8 --accum_iter 4 --epochs 2400000 --warmup_steps 6000 --loss_type kldiv_multipos_focal --num_negs_per_system 20 --lr 1e-5 --continues --nway 5 --gamma 2 --alpha 0 --dim 128"
#/bin/bash -c "python -m train_colbert_ranker --model_name output/colbert_warmup_128_num20_kldiv_ib5-lr1e-05-batch_size_32x1-2023-04-15/232000/0_ColBERTTransformer --train_batch_size 8 --accum_iter 4 --epochs 2400000 --warmup_steps 6000 --loss_type klar --num_negs_per_system 20 --lr 1e-5 --continues --nway 5 --beta_p 0.01 --dim 128"
#
#/bin/bash -c "python -m train_colbert_ranker --model_name ../../msmarco/warmup_0_ColBERTTransformer/ --train_batch_size 8 --accum_iter 4 --epochs 2400000 --warmup_steps 6000 --loss_type kllog --num_negs_per_system 20 --lr 1e-5 --continues --nway 5 --sample_upweight 0 --beta_p 0.01"
#/bin/bash -c "python -m train_colbert_ranker --model_name ../../msmarco/warmup_0_ColBERTTransformer/ --train_batch_size 8 --accum_iter 4 --epochs 2400000 --warmup_steps 6000 --loss_type kldiv_multipos --num_negs_per_system 20 --lr 1e-5 --continues --nway 5 --sample_upweight 10.0"

/bin/bash -c "python -m train_colbert_ranker --model_name output/colbert_warmup_128_num20_kldiv_ib5-lr1e-05-batch_size_32x1-2023-04-15/232000/0_ColBERTTransformer --train_batch_size 8 --accum_iter 4 --epochs 2400000 --warmup_steps 6000 --loss_type ckl --num_negs_per_system 20 --lr 1e-5 --continues --nway 5 --gamma 5.0 --alpha 1.5 --dim 128 --temp 2"
#/bin/bash -c "python -m train_colbert_ranker --model_name Luyu/co-condenser-marco --train_batch_size 8 --accum_iter 4 --epochs 1000000 --warmup_steps 6000 --loss_type kldiv --num_negs_per_system 20 --lr 2e-5 --nway 5 --dim 128"
#/bin/bash -c "python -m train_colbert_ranker --model_name Luyu/co-condenser-marco --train_batch_size 8 --accum_iter 4 --epochs 1000000 --warmup_steps 6000 --loss_type ce --num_negs_per_system 20 --lr 2e-5 --nway 5"
