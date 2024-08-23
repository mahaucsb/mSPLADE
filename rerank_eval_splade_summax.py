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
import numpy as np


def _split_into_batches(features, bsize):
    batches = []
    for offset in range(0, features["input_ids"].size(0), bsize):
        batches.append({key: features[key][offset:offset+bsize] for key in features.keys()})

    return batches

#nvidia_smi.nvmlInit()
#handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)

agg = "max"

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

class Splade(torch.nn.Module):

    def __init__(self, model_type_or_dir, agg="max"):
        super().__init__()
        self.transformer = AutoModelForMaskedLM.from_pretrained(model_type_or_dir)
        assert agg in ("sum", "max")
        self.agg = agg
    
    def forward(self, **kwargs):
        out = self.transformer(**kwargs)["logits"] # output (logits) of MLM head, shape (bs, pad_len, voc_size)
        origin = torch.log(1 + torch.relu(out)) * kwargs["attention_mask"].unsqueeze(-1)
        max_values, _ = torch.max(origin, dim=1)

            # 0 masking also works with max because all activations are positive
       
        mean_values = torch.sum(origin, dim=1)
        
        return max_values, mean_values, origin


model = Splade(model_type_or_dir, agg=agg)
model.eval()
model.to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_type_or_dir)
reverse_voc = {v: k for k, v in tokenizer.vocab.items()}

query_weights_max = dict()
query_weights_sum = dict()
run = defaultdict(dict)

fo = open(sys.argv[4], 'w')

remove_q = float(sys.argv[5]) # 0.2 means remove 20% of tokens
remove_d = float(sys.argv[6])
cur_qid = None 
cur_qtext = None

dtexts = []
dids = []
qids = []
num=128
bsize=64

doc_lens_max = []
doc_lens_sum = []
q_lens_max = []
q_lens_sum = []
for idx, line in enumerate(tqdm(topk)):
    qid, didt, qtext, dtext = line.strip().split("\t")
    
    if len(qrels[qid]) == 0:
        continue

    if int(qid) not in query_weights_max:
        q_rep_max, q_rep_sum, q_multi = model(**tokenizer(qtext, return_tensors="pt").to('cuda'))
        q_rep_max = q_rep_max.squeeze()
        q_rep_sum = q_rep_sum.squeeze()
        #print("q multi shape", q_multi.shape)
        col_max = torch.nonzero(q_rep_max).squeeze().cpu().tolist()
        col_sum = torch.nonzero(q_rep_sum).squeeze().cpu().tolist()
        weights_max = q_rep_max[col_max].cpu().tolist()
        weights_sum = q_rep_sum[col_sum].cpu().tolist()

        if remove_q > 0:
            query_weight_max = sorted([(c, w) for c,w in zip(col_max, weights_max)], key = lambda x: -x[1])
            query_weight_sum = sorted([(c, w) for c,w in zip(col_sum, weights_sum)], key = lambda x: -x[1])
            n_keep_max = int(len(query_weight_max) * (1-remove_q)) + 1
            n_keep_sum = int(len(query_weight_sum) * (1-remove_q)) + 1
            query_weight_sum = query_weight_sum[:n_keep_sum]
            query_weight_max = query_weight_sum[:n_keep_max]
            query_weights_sum[int(qid)] = {k: v for k, v in query_weight_sum}
            query_weights_max[int(qid)] = {k: v for k, v in query_weight_max}
        else:
            query_weights_max[int(qid)] = {k: v for k, v in zip(col_max, weights_max)}
            query_weights_sum[int(qid)] = {k: v for k, v in zip(col_sum, weights_sum)}
        q_lens_max.append(len(query_weights_max[int(qid)]))
        q_lens_sum.append(len(query_weights_sum[int(qid)]))
        del q_rep_max, q_rep_sum

    if (idx+1) % num == 0:
        with torch.no_grad():
            d_features = tokenizer(dtexts, return_tensors="pt", max_length=512, truncation=True, padding=True)
            
            d_features = _split_into_batches(d_features,bsize=bsize)
            i, j = 0, 0
            all_scores_max_max = []
            all_scores_max_sum = []
            all_scores_sum_sum = []
            all_scores_sum_max = []
            all_scores_multi = []
            for batch in d_features:
                for k in batch:
                    batch[k] = batch[k].to("cuda")
                d_batch_max, d_batch_sum, d_batch_multi = model(**batch)
                d_batch_max = d_batch_max.cpu().tolist()
                d_batch_sum = d_batch_sum.cpu().tolist()
                dlens = torch.sum(batch['attention_mask'], dim = 1).unsqueeze(-1).tolist()
                
                #print("d_batch_multi", d_batch_multi.shape)

                all_scores_multi.extend((q_multi @ d_batch_multi.permute(0,2,1)).max(2).values.sum(1).tolist())

                for d_rep in d_batch_max:
                    d_rep = torch.tensor(d_rep)
                    d_col = torch.nonzero(d_rep).squeeze().cpu().tolist()
                    d_weights = d_rep[d_col].cpu().tolist()
                    
                    d_weight = sorted([(c, w) for c,w in zip(d_col, d_weights)], key = lambda x: -x[1])
                    n_keep = int(len(d_weights) * (1-remove_d)) + 1
                    d_weight = d_weight[:n_keep]
                    d_weights = {k: v for k, v in d_weight}
                    doc_lens_max.append(len(d_weights))
                    
                    score = 0
                    qid = qids[i]
                    i += 1
                    for k in query_weights_max[int(qid)]:
                        if k in d_weights:
                            score += d_weights[k] * query_weights_max[int(qid)][k]
                    all_scores_max_max.append(score)

                    for k in query_weights_sum[int(qid)]:
                        if k in d_weights:
                            score += d_weights[k] * query_weights_sum[int(qid)][k]
                    all_scores_sum_max.append(score)
                    
                
                for d_rep, dlen in zip(d_batch_sum, dlens):
                    
                    d_rep = torch.tensor(d_rep)
                    d_col = torch.nonzero(d_rep).squeeze().cpu().tolist()
                    d_weights = d_rep[d_col].cpu().tolist()
                    
                    d_weight = sorted([(c, w) for c,w in zip(d_col, d_weights)], key = lambda x: -x[1])
                    n_keep = int(len(d_weights) * (1-remove_d)) + 1
                    d_weight = d_weight[:n_keep]
                    d_weights = {k: v for k, v in d_weight}
                    
                    score = 0
                    qid = qids[j]
                    j += 1
                    for k in query_weights_max[int(qid)]:
                        if k in d_weights:
                            score += d_weights[k] * query_weights_max[int(qid)][k]
                    all_scores_max_sum.append(score)

                    for k in query_weights_sum[int(qid)]:
                        if k in d_weights:
                            score += d_weights[k] * query_weights_sum[int(qid)][k]
                    
                    all_scores_sum_sum.append(score / dlen[0])

                torch.cuda.empty_cache()

                
            for qid, did, score_max_max, score_max_sum, score_sum_max, score_sum_sum, score_multi in zip(qids, dids, all_scores_max_max, all_scores_max_sum, all_scores_sum_max, all_scores_sum_sum, all_scores_multi):
                fo.write(f"{qid}\t{did}\t{score_max_max}\t{score_max_sum}\t{score_sum_max}\t{score_sum_sum}\t{score_multi}\n")
                fo.flush()
                run[qid][did] = score
                
        dtexts = []
        dids = []
        qids = []
        
    qids.append(qid)
    dtexts.append(dtext)
    dids.append(didt)
    
fo.close()

print("doc length max", np.mean(doc_lens_max))
print("doc length sum", np.mean(doc_lens_sum))
print("query length max", np.mean(q_lens_max))
print("query length sum", np.mean(q_lens_sum))




