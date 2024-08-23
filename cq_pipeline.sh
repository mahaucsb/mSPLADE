#!/bin/bash
#SBATCH -A csb176
#SBATCH --job-name="cq_kl1616"
#SBATCH --output="cq_kl1616.%j.%N.out"
#SBATCH --error="cq_kl1616.%j.%N.err"
#SBATCH --partition=gpu-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --no-requeue
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=32G
#SBATCH -t 10:00:00

source activate splade
#MODEL_DIR=colbert_cont_128_num20_kldiv_ib5-lr1e-05-batch_size_32x1-2023-04-17
#MODEL_DIR=colbert_splade_distill_num1_updatemrr2000_kldiv_ib_position_focal_gamma5.0-alpha1.0-ibp0.1_denoiseFalse_num20_kldiv_ib_position_focal5-lr1e-05-batch_size_16x2-2023-01-20
#MODEL_DIR=colbert_splade_distill-128_kldiv_multipos_focal_gamma4.0-alpha0.0_denoiseFalse_num20_kldiv_multipos_focal5-lr1e-05-batch_size_16x2-2023-05-10
MODEL_DIR=colbert_splade_distill-128_kldiv_multipos-upw0.0_denoiseFalse_num20_kldiv_multipos5-lr1e-05-batch_size_16x2-2023-05-10
#/bin/bash -c "python inference_ColBERT.py training_with_sentence_transformers/output/$MODEL_DIR/30000/0_ColBERTTransformer training_with_sentence_transformers/output/$MODEL_DIR/index 128"
#/bin/bash -c "python preprocess_cq_embedding.py training_with_sentence_transformers/output/$MODEL_DIR/30000/0_ColBERTTransformer training_with_sentence_transformers/output/$MODEL_DIR/index"
#cd ../ContextualQuantizer/src/
#/bin/bash -c "python -m cq.train_code_learner_static --data_file ../../splade_cls/training_with_sentence_transformers/output/$MODEL_DIR/index_42k/orig_embedding.pt --embedding_size 768 --M 16 --K 256 --epochs 200000"

#/bin/bash -c "python -m cq.train_code_learner_static_new --data_file ../../splade_cls/training_with_sentence_transformers/output/$MODEL_DIR/index/orig_embedding.pt --embedding_size 128 --M 16 --K 16 --residual_size 128 --epochs 200000 --model_prefix model_kl_2layer-n128"
#/bin/bash -c "python -m cq.train_code_learner_static_new --data_file ../../splade_cls/training_with_sentence_transformers/output/$MODEL_DIR/index/orig_embedding.pt --embedding_size 128 --M 16 --K 4 --residual_size 128 --epochs 200000 --model_prefix model_kl_2layer-n128"
#/bin/bash -c "python -m cq.train_code_learner_static_new --data_file ../../splade_cls/training_with_sentence_transformers/output/$MODEL_DIR/index/orig_embedding.pt --embedding_size 128 --M 16 --K 256 --residual_size 128 --epochs 200000 --model_prefix model_kl_2layer-n128"

#cd ../../splade_cls
#/bin/bash -c "python rerank_colbert_cq.py training_with_sentence_transformers/output/$MODEL_DIR/42000/0_ColBERTTransformer ../msmarco/wentai_splade_dev_top100.tsv ../msmarco/qrels.dev.tsv ../ContextualQuantizer/src/cq_models/model_96_256/epoch_84000.pt ./training_with_sentence_transformers/output/$MODEL_DIR/index_42k output_cq_kl768_model_cyclic2e-4-1e-6_96_256.tsv"

#cd training_with_sentence_transformers
#/bin/bash -c "python -m train_cq_rank --model_name output/$MODEL_DIR/30000/0_ColBERTTransformer --train_batch_size 16 --accum_iter 2 --epochs 1000000 --warmup_steps 6000 --num_negs_per_system 20 --lr 1e-4 --nway 5 --dim 128 --cq_model ../../ContextualQuantizer/src/cq_models/model_kl_2layer-n128_16_256/epoch_180000.pt  --teacher_type default --loss_type kldiv --M 16 --K 256 --hidden 128" 
#/bin/bash -c "python -m train_cq_rank --model_name output/$MODEL_DIR/30000/0_ColBERTTransformer --train_batch_size 16 --accum_iter 2 --epochs 1000000 --warmup_steps 6000 --num_negs_per_system 20 --lr 1e-4 --nway 5 --dim 128 --cq_model ../../ContextualQuantizer/src/cq_models/model_kl_2layer-n128_16_16/epoch_194500.pt  --teacher_type default --loss_type kldiv --M 16 --K 16 --hidden 128" 
#/bin/bash -c "python -m train_cq_rank --model_name output/$MODEL_DIR/30000/0_ColBERTTransformer --train_batch_size 16 --accum_iter 2 --epochs 1000000 --warmup_steps 6000 --num_negs_per_system 20 --lr 1e-4 --nway 5 --dim 128 --cq_model ../../ContextualQuantizer/src/cq_models/model_kl_2layer-n128_16_4/epoch_164500.pt  --teacher_type default --loss_type kldiv --M 16 --K 4 --hidden 128" 


#/bin/bash -c "python -m train_cq_rank --model_name output/$MODEL_DIR/30000/0_ColBERTTransformer --train_batch_size 16 --accum_iter 2 --epochs 1000000 --warmup_steps 6000 --num_negs_per_system 20 --lr 1e-4 --nway 5 --dim 128 --cq_model ../../ContextualQuantizer/src/cq_models/model_2layer-n128_16_32/epoch_191500.pt  --teacher_type default --loss_type kldiv --M 16 --K 32 --hidden 128" 
#/bin/bash -c "python -m train_cq_rank --model_name output/$MODEL_DIR/30000/0_ColBERTTransformer --train_batch_size 16 --accum_iter 2 --epochs 1000000 --warmup_steps 6000 --num_negs_per_system 20 --lr 1e-4 --nway 5 --dim 128 --cq_model ../../ContextualQuantizer/src/cq_models/model_2layer-n128_16_16/epoch_199500.pt  --teacher_type default --loss_type kldiv --M 16 --K 16 --hidden 128" 
#/bin/bash -c "python -m train_cq_rank --model_name output/$MODEL_DIR/30000/0_ColBERTTransformer --train_batch_size 16 --accum_iter 2 --epochs 1000000 --warmup_steps 6000 --num_negs_per_system 20 --lr 1e-4 --nway 5 --dim 128 --cq_model ../../ContextualQuantizer/src/cq_models/model_2layer-n128_16_4/epoch_175500.pt  --teacher_type default --loss_type kldiv --M 16 --K 4 --hidden 128" 

M=16
K=16
hidden=128
for ((n=1000; n<=7000; n=n+1000))
do 
echo $n
/bin/bash -c "python rerank_colbert_cq_new.py training_with_sentence_transformers/output/$MODEL_DIR/30000/0_ColBERTTransformer ../msmarco/wentai_splade_dev_top100.tsv ../msmarco/qrels.dev.tsv training_with_sentence_transformers/output/colbert_cq_test_fromkl_kldiv_default_16-16-128-2023-05-23/$n/1_CodeLearner/pytorch_model.bin ./training_with_sentence_transformers/output/$MODEL_DIR/index output_cq_fromkl_klar4-128-newnorm_model_cyclic2e-4-1e-6_n${hidden}-${M}-${K}_new_kldiv_default_$n.tsv 128 ${hidden} ${M} ${K}"
yr=19
/bin/bash -c "python rerank_colbert_cq_new.py training_with_sentence_transformers/output/$MODEL_DIR/30000/0_ColBERTTransformer ../msmarco/splade_klfr_5-1_num1.2019.top1000.trec.tsv ../msmarco/20${yr}qrels-pass.txt  training_with_sentence_transformers/output/colbert_cq_test_fromkl_kldiv_default_16-16-128-2023-05-23/$n/1_CodeLearner/pytorch_model.bin ./training_with_sentence_transformers/output/$MODEL_DIR/index output_fromkl_20${yr}_cq_klar4-128-newnorm_model_cyclic2e-4-1e-6_n${hidden}-${M}-${K}_new_kldiv_default_$n.tsv 128 ${hidden} ${M} ${K}"
yr=20
/bin/bash -c "python rerank_colbert_cq_new.py training_with_sentence_transformers/output/$MODEL_DIR/30000/0_ColBERTTransformer ../msmarco/splade_klfr_5-1_num1.2020.top1000.trec.tsv ../msmarco/20${yr}qrels-pass.txt  training_with_sentence_transformers/output/colbert_cq_test_fromkl_kldiv_default_16-16-128-2023-05-23/$n/1_CodeLearner/pytorch_model.bin ./training_with_sentence_transformers/output/$MODEL_DIR/index output_fromkl_20${yr}_cq_klar4-128-newnorm_model_cyclic2e-4-1e-6_n${hidden}-${M}-${K}_new_kldiv_default_$n.tsv 128 ${hidden} ${M} ${K}"
done
