import torch
from transformers import AutoTokenizer
from tqdm import tqdm
import os
import json
import sys
from training_with_sentence_transformers.models import ColBERTTransformer
import gzip

def _split_into_batches(features, bsize):
    batches = []
    for offset in range(0, features["input_ids"].size(0), bsize):
        batches.append({key: features[key][offset:offset+bsize] for key in features.keys()})

    return batches

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
model = ColBERTTransformer(model_type_or_dir, max_seq_length=256, dim=int(sys.argv[3]))
checkpoint = torch.load(os.path.join(model_type_or_dir, "checkpoint.pt"), map_location='cpu')
model.load_state_dict(checkpoint)
    
model.eval()
model.to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_type_or_dir)
tokens = ["[unused0]", "[unused1]", "[unused2]"] #[unused0] for query, [unused1] for doc, [unused2] for query expansion
tokenizer.add_tokens(tokens, special_tokens=True)
reverse_voc = {v: k for k, v in tokenizer.vocab.items()}


fo = gzip.open(os.path.join(out_dir, f"file_static_embedding.jsonl.gz"), "w")
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
with torch.no_grad():
    for tok in tqdm(range(model.auto_model.module.config.vocab_size)):
        output_states = model.auto_model(torch.tensor([[101, 2, tok, 102]]).to(device), torch.tensor([[1,1,1,1]]).to(device), return_dict=False)

        all_layer_idx = 2
        if len(output_states) < 3: #Some models only output last_hidden_states and all_hidden_states
            all_layer_idx = 1
            
        hidden_states = output_states[all_layer_idx]
        tok_emb = model.linear(hidden_states[-1])

        outline = json.dumps({"id": int(tok), "content": tok_emb[:,2,:].tolist()}) + "\n"
        fo.write(outline.encode('utf-8'))
        fo.flush()
        

scale = 100
file_per = 100000
i = 0
end_i = 10000 #500000


fo = None
with open("../msmarco/collection.tsv") as f:
    for line in tqdm(f):
        if i > end_i:
            i += 1
            break
        if i % file_per == 0:
            if fo is not None:
                fo.close()
                #break
            fo = gzip.open(os.path.join(out_dir, f"file_{i // file_per}_unorm.jsonl.gz"), "w")

        did, doc = line.strip().split("\t")    
        
        with torch.no_grad():
            d_features = tokenizer("[unused1] " + doc, return_tensors="pt", truncation=True).to('cuda')
            doc_rep = model(d_features) # (sparse) doc rep in voc space, shape (30522,)
        d_mask = doc_rep['attention_mask'].to('cuda')
        d_mask = d_mask.unsqueeze(-1)
        d_emb = doc_rep['last_layer_embeddings']
        del doc_rep
        
        token_rep_d =  d_emb * d_mask
        del d_mask, d_emb
        #token_rep_d = torch.nn.functional.normalize(token_rep_d).detach().to('cpu').tolist()
        token_rep_d = token_rep_d.detach().to('cpu').tolist()

        outline = json.dumps({"id": int(did), "content": token_rep_d}) + "\n"
        fo.write(outline.encode('utf-8'))
        fo.flush()
        i += 1


