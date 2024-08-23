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
from training_with_sentence_transformers.models import MLMTransformer
from training_with_sentence_transformers.losses import pairwise_dot_score
import nvidia_smi
from sentence_transformers import  models as smodel
import torch.nn as nn
from sentence_transformers import SentenceTransformer

def _split_into_batches(features, bsize):
    batches = []
    for offset in range(0, features["input_ids"].size(0), bsize):
        batches.append({key: features[key][offset:offset+bsize] for key in features.keys()})

    return batches

nvidia_smi.nvmlInit()
handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)

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
        if self.agg == "max":
            values, _ = torch.max(torch.log(1 + torch.relu(out)) * kwargs["attention_mask"].unsqueeze(-1), dim=1)
            return values
            # 0 masking also works with max because all activations are positive
        else:
            return torch.sum(torch.log(1 + torch.relu(out)) * kwargs["attention_mask"].unsqueeze(-1), dim=1)


model0 = Splade(os.path.join(model_type_or_dir, "0_MLMTransformer"), agg=agg)
model1 = smodel.Dense(in_features=30522, out_features = 30522, activation_function = nn.Identity())

checkpoint = torch.load(os.path.join(model_type_or_dir, "1_Dense", "pytorch_model.bin"), map_location='cpu')
model1.load_state_dict(checkpoint)


model0.eval()
model0.to("cuda")

model1.eval()
model1.to("cuda")

tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_type_or_dir, "0_MLMTransformer"))
reverse_voc = {v: k for k, v in tokenizer.vocab.items()}

query_weights = dict()
run = defaultdict(dict)

fo = open(sys.argv[4], 'w')

cur_qid = None
cur_qtext = None

dtexts = []
dids = []
qids = []
num=128
bsize=64

for idx, line in enumerate(tqdm(topk)):
    qid, didt, qtext, dtext = line.strip().split("\t")
    
    if len(qrels[qid]) == 0:
        continue

    if int(qid) not in query_weights:
        q_rep = model1({"sentence_embedding": model0(**tokenizer(qtext, return_tensors="pt").to('cuda'))})['sentence_embedding'].squeeze()
        
        col = torch.nonzero(q_rep).squeeze().cpu().tolist()
        weights = q_rep[col].cpu().tolist()
        query_weights[int(qid)] = {k: v for k, v in zip(col, weights)}
        del q_rep

    if (idx+1) % num == 0:
        with torch.no_grad():
            d_features = tokenizer(dtexts, return_tensors="pt", max_length=512, truncation=True, padding=True)
            d_features = _split_into_batches(d_features,bsize=bsize)
            i = 0
            all_scores = []
            for batch in d_features:
                for k in batch:
                    batch[k] = batch[k].to("cuda")
                
                batch['sentence_embedding'] = model0(**batch)
                d_batch = batch['sentence_embedding'].cpu().tolist()
                for d_rep in d_batch:
                    score = 0
                    qid = qids[i]
                    i += 1
                    
                    for k in query_weights[int(qid)]:
                        score += d_rep[k] * query_weights[int(qid)][k]
                    all_scores.append(score)
                torch.cuda.empty_cache()
                
            for qid, did, score in zip(qids, dids, all_scores):
                fo.write(f"{qid}\t{did}\t{score}\n")
                fo.flush()
                run[qid][did] = score
                
        dtexts = []
        dids = []
        qids = []
        
    qids.append(qid)
    dtexts.append(dtext)
    dids.append(didt)
    
fo.close()


for VALIDATION_METRIC in ['recip_rank','ndcg_cut_10']:
    for top_k in [5,10,20,100]:
        top_run = defaultdict(dict)
        for q in run:
            docs = sorted(run[q].items(), key=lambda x: -x[1])
            for item in docs[:top_k]:
                top_run[q][item[0]] = item[1]
        trec_eval = pytrec_eval.RelevanceEvaluator(qrels, {VALIDATION_METRIC})
        eval_scores = trec_eval.evaluate(top_run)
        print(VALIDATION_METRIC, top_k, mean([d[VALIDATION_METRIC] for d in eval_scores.values()]))




