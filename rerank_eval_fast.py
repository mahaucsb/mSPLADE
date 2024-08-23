import torch
from transformers import AutoTokenizer
from collections import defaultdict
from tqdm import tqdm
import os
import pytrec_eval
from collections import OrderedDict, defaultdict
from statistics import mean
import json
import sys
from training_with_sentence_transformers.models import ColBERTTransformer
from training_with_sentence_transformers.losses import pairwise_dot_score
#import nvidia_smi

def _split_into_batches(features, bsize):
    batches = []
    for offset in range(0, features["input_ids"].size(0), bsize):
        batches.append({key: features[key][offset:offset+bsize] for key in features.keys()})

    return batches

#nvidia_smi.nvmlInit()
#handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)

agg = "max"
bsize = 128
model_type_or_dir = sys.argv[1] #"colspla-prf_from_colbert_3e-6_negpersys5" #"output/0_MLMTransformer"


topk = open(sys.argv[2]) #open("../msmarco/wentai_splade_dev_top1000.tsv")

VALIDATION_METRIC = 'recip_rank'   #'recip_rank' #'ndcg_cut_10' 

qrel_file = sys.argv[3] #"../msmarco/qrels.dev.tsv"

qrels = defaultdict(dict)
with open(qrel_file) as f:
    for line in f:
        try:
            qid, _, did, rel = line.strip().split("\t")
        except:
            qid, _, did, rel = line.strip().split(" ")
        if int(rel) > 0:
            qrels[qid][did] = int(rel)


dim = int(sys.argv[5])
dense_weight = 0.0
model = ColBERTTransformer(model_type_or_dir, max_seq_length=256,dim=int(sys.argv[5]))
checkpoint = torch.load(os.path.join(model_type_or_dir, "checkpoint.pt"), map_location='cpu')
model.load_state_dict(checkpoint)
    
model.eval()
model.to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_type_or_dir)
tokens = ["[unused0]", "[unused1]", "[unused2]"] #[unused0] for query, [unused1] for doc, [unused2] for query expansion
tokenizer.add_tokens(tokens, special_tokens=True)
# top k inference               

run = defaultdict(dict)

fo = open(sys.argv[4], 'w')

cur_qid = None
cur_qtext = None
dtexts = []
dids = []
for idx, line in enumerate(tqdm(topk)):
    qid, didt, qtext, dtext = line.strip().split("\t")
    qtexts = qtext.split(" [SEP] [unused2] ")
    qtext = qtexts[0]

    if len(qrels[qid]) == 0:
        continue
    if cur_qid is not None and qid != cur_qid:
        with torch.no_grad():
            q_features = tokenizer("[unused0] " + cur_qtext, return_tensors="pt").to('cuda')
            q_features = model(q_features)
            token_rep_q = torch.nn.functional.normalize(q_features['last_layer_embeddings'], p=2, dim=2)
             
            d_features = tokenizer(["[unused1] " + dtext for dtext in dtexts], return_tensors="pt", max_length=256,truncation=True,padding=True)
            d_features = _split_into_batches(d_features,bsize=bsize)
            
            all_scores = []
            for batch in d_features:
                d_batch = model(batch)
                d_mask = d_batch['attention_mask'].to('cuda')
                d_emb = d_batch['last_layer_embeddings']
                del d_batch
                d_mask = d_mask
                token_rep_d =  d_emb * d_mask.unsqueeze(-1)
                del d_emb
                # token_rep_d = torch.nn.functional.normalize(token_rep_d) BIG CHANGE
                token_rep_d = torch.nn.functional.normalize(token_rep_d, p=2, dim=2)
                token_level_score = token_rep_q @ token_rep_d.permute(0,2,1)
                iter_mask = ~d_mask.unsqueeze(1).repeat(1, token_level_score.shape[1], 1).bool()
                token_level_score[iter_mask] = -9999

                scores =  token_level_score.max(2).values.sum(1).tolist()
                del token_rep_d
                torch.cuda.empty_cache()
                all_scores.extend(scores)
                
            
            for did, score in zip(dids, all_scores):
                fo.write(f"{cur_qid}\t{did}\t{score}\n")
                fo.flush()
                run[cur_qid][did] = score

        dtexts = []
        dids = []
        
    cur_qid = qid
    cur_qtext = qtext
    dtexts.append(dtext)
    dids.append(didt)
    
fo.close()


for VALIDATION_METRIC in ['recip_rank','ndcg_cut_10', 'ndcg_cut_20', 'P_20']:
    for top_k in [5,10,20,100]:
        top_run = defaultdict(dict)
        for q in run:
            docs = sorted(run[q].items(), key=lambda x: -x[1])
            for item in docs[:top_k]:
                top_run[q][item[0]] = item[1]
        trec_eval = pytrec_eval.RelevanceEvaluator(qrels, {VALIDATION_METRIC})
        eval_scores = trec_eval.evaluate(top_run)
        print(VALIDATION_METRIC, top_k, mean([d[VALIDATION_METRIC] for d in eval_scores.values()]))


