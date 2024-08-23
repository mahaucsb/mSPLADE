import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from collections import defaultdict
from tqdm import tqdm
import os
import pytrec_eval
from collections import OrderedDict, defaultdict
from statistics import mean
import json
import sys
from training_with_sentence_transformers.models import ColBERTTransformer

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



dense_weight = 0.0
model = ColBERTTransformer(model_type_or_dir, max_seq_length=256)
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
            
            max_rep_q = torch.max(token_rep_q, dim=1).values
            mean_rep_q = torch.mean(token_rep_q, dim=1)
            d_features = tokenizer(["[unused1] " + dtext for dtext in dtexts], return_tensors="pt", max_length=256,truncation=True,padding=True)
            d_features = _split_into_batches(d_features,bsize=bsize)
            
            all_scores = []
            all_scores_max_max = []
            all_scores_max_sum = []
            all_scores_sum_sum = []
            all_scores_sum_max = []

            for batch in d_features:
                d_batch = model(batch)
                d_mask = d_batch['attention_mask'].to('cuda')
                d_emb = d_batch['last_layer_embeddings']
                del d_batch
                d_mask = d_mask.unsqueeze(-1)
                token_rep_d =  d_emb * d_mask
                del d_mask, d_emb
                token_rep_d = torch.nn.functional.normalize(token_rep_d)
                max_rep_d = torch.max(token_rep_d, dim=1).values
                mean_rep_d = torch.mean(token_rep_d, dim=1)

                scores =  (token_rep_q @ token_rep_d.permute(0,2,1)).max(2).values.sum(1).tolist()
                scores_maxmax = (max_rep_q * max_rep_d).sum(dim=-1).tolist()
                scores_maxmean = (max_rep_q * mean_rep_d).sum(dim=-1).tolist()
                scores_meanmax = (mean_rep_q * max_rep_d).sum(dim=-1).tolist()
                scores_meanmean = (mean_rep_q * mean_rep_d).sum(dim=-1).tolist()
                del token_rep_d
                torch.cuda.empty_cache()
                all_scores.extend(scores)
                all_scores_max_max.extend(scores_maxmax)
                all_scores_max_sum.extend(scores_maxmean)
                all_scores_sum_sum.extend(scores_meanmean)
                all_scores_sum_max.extend(scores_meanmax)
                #info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
                #print("{}, Memory : ({:.2f}% free): {}(total), {} (free), {} (used)".format(nvidia_smi.nvmlDeviceGetName(handle), 100*info.free/info.total, info.total, info.free, info.used))
            
        

            for did, score_max_max, score_max_sum, score_sum_max, score_sum_sum, score_multi in zip(dids, all_scores_max_max, all_scores_max_sum, all_scores_sum_max, all_scores_sum_sum, all_scores):
                fo.write(f"{cur_qid}\t{did}\t{score_max_max}\t{score_max_sum}\t{score_sum_max}\t{score_sum_sum}\t{score_multi}\n")
                fo.flush()

    
        dtexts = []
        dids = []
        
    cur_qid = qid
    cur_qtext = qtext
    dtexts.append(dtext)
    dids.append(didt)
    
fo.close()

