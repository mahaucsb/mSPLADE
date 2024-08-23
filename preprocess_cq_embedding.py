import torch
from transformers import AutoTokenizer
from tqdm import tqdm
import os
import json
import sys
from training_with_sentence_transformers.models import ColBERTTransformer
import gzip


# MODEL_DIR=colbert_splade_distill_num1_updatemrr2000_kldiv_ib_position_focal_gamma5.0-alpha1.0-ibp0.1_denoiseFalse_num20_kldiv_ib_position_focal5-lr1e-05-batch_size_16x2-2023-01-20
#python inference_ColBERT.py training_with_sentence_transformers/output/$MODEL_DIR/42000/0_ColBERTTransformer training_with_sentence_transformers/output/$MODEL_DIR/index_42k

agg = "max"
bsize = 128
model_type_or_dir = sys.argv[1] #"colspla-prf_from_colbert_3e-6_negpersys5" #"output/0_MLMTransformer"
out_dir = sys.argv[2]

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# loading model and tokenizer
dense_weight = 0.0

tokenizer = AutoTokenizer.from_pretrained(model_type_or_dir)
tokens = ["[unused0]", "[unused1]", "[unused2]"] #[unused0] for query, [unused1] for doc, [unused2] for query expansion
tokenizer.add_tokens(tokens, special_tokens=True)



doc_tokens_mapping = dict()
with open("../msmarco/collection.tsv") as f:
    i = 0
    for line in tqdm(f):
        did, doc = line.strip().split("\t")   
        d_features = tokenizer("[unused1] " + doc, return_tensors="pt", truncation=True)['input_ids'].tolist()[0]   
        doc_tokens_mapping[i] = d_features
        i += 1
        if i > 10001:
            break

non_contextual_emb_dict = dict()

with gzip.open(os.path.join(out_dir, f"file_static_embedding.jsonl.gz")) as f:
    for line in f:
        info = json.loads(line)
        non_contextual_emb_dict[info['id']] = info['content'][0]

orig_emb = []
with gzip.open(os.path.join(out_dir, "file_0_unorm.jsonl.gz")) as f:
    for line in tqdm(f):
        try:
            info = json.loads(line)
            tokens = doc_tokens_mapping[info['id']]
            non_context = [non_contextual_emb_dict[t] for t in tokens]
            doc_emb = info['content'][0]
            orig_emb.extend([x + y for x,y in zip(doc_emb, non_context)])
        except:
            break

        if len(orig_emb) > 500000:
            break
torch.save(torch.tensor(orig_emb), os.path.join(out_dir, f"orig_embedding.pt"))