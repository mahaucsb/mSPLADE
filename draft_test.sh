MODEL_DIR=colbert_splade_distill_num1_kldiv_focal_default_gamma7.0-alpha0.2_denoiseFalse_num20_kldiv_focal5-lr1e-05-batch_size_8x4-2022-11-02
n=10
python rerank_eval_fast.py training_with_sentence_transformers/output/$MODEL_DIR/${n}0000/0_ColBERTTransformer ../msmarco/wentai_splade_dev_top100.tsv ../msmarco/qrels.dev.tsv output_dev_colbert_distill_splade_iter${n}0k_kldiv5_focal_gamma7.0_1e-5.tsv
n=11
python rerank_eval_fast.py training_with_sentence_transformers/output/$MODEL_DIR/${n}0000/0_ColBERTTransformer ../msmarco/wentai_splade_dev_top100.tsv ../msmarco/qrels.dev.tsv output_dev_colbert_distill_splade_iter${n}0k_kldiv5_focal_gamma7.0_1e-5.tsv
