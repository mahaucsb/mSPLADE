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
from training_with_sentence_transformers.models import MLMTransformerDense, ColBERTTransformer
from training_with_sentence_transformers.losses import pairwise_dot_score
import nvidia_smi

nvidia_smi.nvmlInit()
handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)

agg = "max"

model_type_or_dir = sys.argv[1] #"output/0_MLMTransformer"
topk = open(sys.argv[2]) #"../msmarco/top1000.dev")

VALIDATION_METRIC = 'ndcg_cut_10'    #'recip_rank' #'ndcg_cut_10' 

qrel_file = sys.argv[3] #"../msmarco/qrels.dev.tsv"
model_type = sys.argv[5] #"splade_cls"
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
        
class SpladeDense(torch.nn.Module):

    def __init__(self, model_type_or_dir, agg="max"):
        super().__init__()
        self.transformer = MLMTransformerDense(model_type_or_dir) #AutoModelForMaskedLM.from_pretrained(model_type_or_dir)
        self.transformer.load(model_type_or_dir)
        assert agg in ("sum", "max")
        self.agg = agg
        self.transformer.config.output_hidden_states == True
    
    def forward(self, features):
        output_states = self.transformer(features)
        out_sparse = output_states['token_embeddings'] # output (logits) of MLM head, shape (bs, pad_len, voc_size)

        cls = output_states['cls']

        if self.agg == "max":
            sparse_values, _ = torch.max(torch.log(1 + torch.relu(out_sparse)) * features["attention_mask"].unsqueeze(-1), dim=1)
            
            # 0 masking also works with max because all activations are positive
        else:
            sparse_values = torch.sum(torch.log(1 + torch.relu(out_sparse)) * features["attention_mask"].unsqueeze(-1), dim=1)

        return sparse_values, cls


lineout = open(sys.argv[4]+".tsv", "w")

if model_type == "splade":
    model = Splade(model_type_or_dir, agg=agg)
    model.eval()
    model.to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_type_or_dir)
    reverse_voc = {v: k for k, v in tokenizer.vocab.items()}


    # top k inference         
    query_weights = dict()
    run = defaultdict(dict)
    idx = 0
    for idx, line in enumerate(tqdm(topk)):
        qid, did, qtext, dtext = line.strip().split("\t")
        if len(qrels[qid]) == 0:
            continue
        if int(qid) not in query_weights:
            q_rep = model(**tokenizer(qtext, return_tensors="pt").to('cuda')).squeeze() 
            col = torch.nonzero(q_rep).squeeze().cpu().tolist()
            weights = q_rep[col].cpu().tolist()
            query_weights[int(qid)] = {k: v for k, v in zip(col, weights)}
            del q_rep
        d_rep = model(**tokenizer(dtext, return_tensors="pt", max_length=512, truncation=True).to('cuda')).squeeze().cpu().tolist()
        score = 0
        for k in query_weights[int(qid)]:
            score += d_rep[k] * query_weights[int(qid)][k]

        run[qid][did] = score
        lineout.write(f"{qid}\t{did}\t{score}\n")
        lineout.flush()
        del d_rep
    
elif model_type == "splade_cls":
    dense_weight = 0.2
    model = SpladeDense(model_type_or_dir, agg=agg)
    checkpoint = torch.load(os.path.join(model_type_or_dir, "checkpoint.pt"), map_location='cpu')
    model.transformer.load_state_dict(checkpoint)
    model.eval()
    model.to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_type_or_dir)
    reverse_voc = {v: k for k, v in tokenizer.vocab.items()}


    # top k inference               
    query_weights = dict()
    query_clss = dict()
    run = defaultdict(dict)
    for idx, line in enumerate(tqdm(topk)):
        #if idx < 6359243:
        #    continue
        qid, did, qtext, dtext = line.strip().split("\t")
        if len(qrels[qid]) == 0:
            continue
        if int(qid) not in query_weights:
            features = tokenizer(qtext, return_tensors="pt").to('cuda')
            q_rep, q_cls = model(features)
            q_rep = q_rep.squeeze().cpu()
            q_cls = q_cls.cpu()
            col = torch.nonzero(q_rep).squeeze().tolist()
            weights = q_rep[col].tolist()
            query_weights[int(qid)] = {k: v for k, v in zip(col, weights)}
            query_clss[int(qid)] = q_cls.tolist()
            del q_rep, q_cls
        d_features = tokenizer(dtext, return_tensors="pt", max_length=512,truncation=True).to('cuda') 
        d_rep, d_cls = model(d_features)
        d_rep = d_rep.squeeze().cpu().tolist()
        d_cls = d_cls.cpu()
        score = 0
        for k in query_weights[int(qid)]:
            score += d_rep[k] * query_weights[int(qid)][k] * (1-dense_weight)
        score += pairwise_dot_score(query_clss[int(qid)], d_cls).tolist()[0] * dense_weight

        run[qid][did] = score
        lineout.write(f"{qid}\t{did}\t{score}\n")
        lineout.flush()
        del d_rep, d_cls

elif model_type == "colbert":
    dense_weight = 0.0
    model = ColBERTTransformer(model_type_or_dir, max_seq_length=256)
    checkpoint = torch.load(os.path.join(model_type_or_dir, "checkpoint.pt"), map_location='cpu')
    model.load_state_dict(checkpoint)
    

    model.eval()
    model.to("cuda")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_type_or_dir)
    tokens = ["[unused0]", "[unused1]", "[unused2]"] #[unused0] for query, [unused1] for doc, [unused2] for query expansion
    tokenizer.add_tokens(tokens, special_tokens=True)
    # top k inference               
    query_weights = dict()
    run = defaultdict(dict)
    
    for idx, line in enumerate(tqdm(topk)):
        qid, did, qtext, dtext = line.strip().split("\t")
        if len(qrels[qid]) == 0:
            continue
        query_weights = dict()
        #if int(qid) not in query_weights:
        #"[unused0] " + 
        q_features = tokenizer("[unused0] " + qtext, return_tensors="pt").to('cuda')
        q_features = model(q_features)
        token_rep_q = torch.nn.functional.normalize(q_features['last_layer_embeddings'], p=2, dim=2)
        q_cls = q_features['cls']
        query_weights[int(qid)] = [token_rep_q.to('cpu'), q_cls.to('cpu')]
        del q_features, q_cls, token_rep_q
        #"[unused1] " + 
        d_features = tokenizer("[unused1] " + dtext, return_tensors="pt", max_length=256,truncation=True).to('cuda') 
        d_features = model(d_features)
        d_mask = d_features['attention_mask']
        d_emb = d_features['last_layer_embeddings']
        d_cls = d_features['cls'].to('cpu')
        cls_score = pairwise_dot_score(query_weights[int(qid)][1], d_cls).tolist()[0]
        
        token_rep_d =  d_emb * d_mask.unsqueeze(-1)
        del d_features, d_mask, d_emb
        token_rep_d = torch.nn.functional.normalize(token_rep_d).to('cpu')
        
        tok_score =  (query_weights[int(qid)][0] @ token_rep_d.permute(0,2,1)).max(2).values.sum(1).tolist()[0]
        score = cls_score * dense_weight + tok_score
        
        run[qid][did] = score
        lineout.write(f"{qid}\t{did}\t{score}\n")
        lineout.flush()
        del d_cls, token_rep_d
        torch.cuda.empty_cache()
        #info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        #print("{}, Memory : ({:.2f}% free): {}(total), {} (free), {} (used)".format(nvidia_smi.nvmlDeviceGetName(handle), 100*info.free/info.total, info.total, info.free, info.used))


else:
    print(f"Unexpected model type!")
    raise
with open(sys.argv[4], "w") as fo:  #"output_10000_dev.run.json", "w") as fo:
    fo.write(json.dumps(run))

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


#trec_eval = pytrec_eval.RelevanceEvaluator(qrels, {VALIDATION_METRIC})
#eval_scores = trec_eval.evaluate(run)
#print(mean([d[VALIDATION_METRIC] for d in eval_scores.values()]))
