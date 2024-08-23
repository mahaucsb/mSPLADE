'''
#import torch
#from sentence_transformers import util
#from transformers import AutoModelForMaskedLM, AutoTokenizer
from collections import defaultdict
#from tqdm import tqdm
import os
import tarfile
import gzip
from collections import OrderedDict, defaultdict
from statistics import mean
import json
import sys
#from training_with_sentence_transformers.models import MLMTransformerDense, ColBERTTransformer
#from training_with_sentence_transformers.losses import pairwise_dot_score
import pickle

agg = "max"

model_type_or_dir = sys.argv[1] #"output/0_MLMTransformer"

data_folder = '/expanse/lustre/projects/csb176/yryang/msmarco'

qrel_file = "../msmarco/qrels.dev.tsv"
qrels = defaultdict(dict)
with open(qrel_file) as f:
    for line in f:
        try:
            qid, _, did, rel = line.strip().split("\t")
        except:
            qid, _, did, rel = line.strip().split(" ")
        if int(rel) > 0:
            qrels[qid][did] = int(rel)


# As training data we use hard-negatives that have been mined using various systems
hard_negatives_filepath = os.path.join(data_folder, 'msmarco-hard-negatives-splade.jsonl.gz')

train_queries = {}
negs_to_use = None

ce_score_margin = 3.0
num_negs_per_system = 3
'''
'''
#### Read the corpus file containing all the passages. Store them in the corpus dict
corpus = {}  # dict in the format: passage_id -> passage. Stores all existing passages
collection_filepath = os.path.join(data_folder, 'collection.tsv')
if not os.path.exists(collection_filepath):
    tar_filepath = os.path.join(data_folder, 'collection.tar.gz')
    
    with tarfile.open(tar_filepath, "r:gz") as tar:
        tar.extractall(path=data_folder)

with open(collection_filepath, 'r', encoding='utf8') as fIn:
    for line in fIn:
        pid, passage = line.strip().split("\t")
        pid = int(pid)
        corpus[pid] = passage
'''  
'''      
ce_scores_file = os.path.join(data_folder, 'cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz')

with gzip.open(ce_scores_file, 'rb') as fIn:
    ce_scores = pickle.load(fIn)

queries = {}  # dict in the format: query_id -> query. Stores all training queries
queries_filepath = os.path.join(data_folder, 'queries.train.tsv')
if not os.path.exists(queries_filepath):
    tar_filepath = os.path.join(data_folder, 'queries.tar.gz')
    
    with tarfile.open(tar_filepath, "r:gz") as tar:
        tar.extractall(path=data_folder)

with open(queries_filepath, 'r', encoding='utf8') as fIn:
    for line in fIn:
        qid, query = line.strip().split("\t")
        qid = int(qid)
        queries[qid] = query


with gzip.open(hard_negatives_filepath, 'rt') as fIn:
    for line in fIn:
        data = json.loads(line)

        #Get the positive passage ids
        pos_pids = data['pos']

        if len(pos_pids) == 0:  #Skip entries without positives passages
            continue

        pos_min_ce_score = min([ce_scores[data['qid']][pid] for pid in data['pos']])
        ce_score_threshold = pos_min_ce_score - ce_score_margin

        neg_pids = []
        
        #Get the hard negatives
        system_name = 'splade'
        
        if system_name not in data['neg']:
            continue

        system_negs = data['neg'][system_name]
        negs_added = 0
        for pid in system_negs:
            pid = int(pid)
            if ce_scores[data['qid']][pid] > ce_score_threshold:
                    continue
            if pid not in neg_pids:
                neg_pids.append(pid)
                negs_added += 1
                if negs_added >= num_negs_per_system:
                    break

        if (len(pos_pids) > 0 and len(neg_pids) > 0):
            train_queries[data['qid']] = {'qid': data['qid'], 'pos': data['pos'], 'pid': neg_pids}

open("queries_train_splade_top3.json", "w").write(json.dumps(train_queries))
'''

'''
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
        


lineout = open(sys.argv[2]+".tsv", "a")


model = Splade(model_type_or_dir, agg=agg)
model.eval()
model.to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_type_or_dir)
reverse_voc = {v: k for k, v in tokenizer.vocab.items()}

# top k inference        
doc_seen = set()
idx = 0
for qid in train_queries:
    if len(qrels[qid]) == 0:
        continue
    dids = []
    for did in train_queries[qid]['pid']:
        if did in doc_seen:
            continue
        dids.append(did)
        doc_seen.add(did)
        
    dtext = [corpus[did] for did in dids]
    d_reps = model(**tokenizer(dtext, return_tensors="pt", max_length=512, truncation=True).to('cuda')).squeeze().cpu().tolist()
    
    for did, d_rep in zip(dids, d_reps):
        lineout.write(f"{did}\t{json.dumps(d_rep)}\n")
        lineout.flush()
    del d_reps
    


'''
'''
with open("../msmarco/wentai_splade_2020.trec") as f, open("queries_2020_top3_original.trec" , "w") as fo:
    for line in f:
        if line.split("\t")[3] in ['1','2','3']:
            fo.write(line)
            fo.flush()


'''
'''
def max_tok(tok_dicts, n=10):
    maxheap = dict()
    smallest = ["a",-1]

    for tok_dict in tok_dicts:
        for k,v in tok_dict.items():
            if len(maxheap) == 0:
                maxheap[k] = v
                smallest = [k,v]
            elif len(maxheap) < n:
                maxheap[k] = v
                if v < smallest[1]:
                    smallest = [k,v]
            elif v > smallest[1]:
                del maxheap[smallest[0]]
                maxheap[k] = v
                smallest[0] = min(maxheap, key=maxheap.get)
                smallest[1] = maxheap[smallest[0]]

    return maxheap    


import json
import sys
import gzip
import os
from collections import defaultdict

qdoc = defaultdict(list)
with open("queries_2020_top3_original.trec") as f:
    for line in f:
        qid, _, did, *_ = line.split("\t")
        qdoc[int(qid)].append(int(did))

corpus = dict()
with gzip.open("../msmarco/2020_top3_docs_original.jsonl.gz") as f:
    for line in f:
        doc_dict = json.loads(line)
        id = doc_dict['id']
        toks = doc_dict["vector"]
        corpus[id] = max_tok([toks], 30)

fo = open("2020_top3_query_top30.json", "w")
for q in qdoc:
    fo.write(f"{q}\t{json.dumps(max_tok([corpus[did] for  did in qdoc[q]], 30))}\n")
'''

import json
q_toks = dict()
with open("2020_top3_query_top30.json") as f:
    for line in f:
        qid, vec = line.split("\t")
        vecs = json.loads(vec)
        vecs = sorted([(k,v) for k,v in vecs.items()], key=lambda x: -x[1])
        q_toks[qid] = vecs
print(len(q_toks))
with open("../msmarco/msmarco-passagetest2020-top1000.tsv") as f, open("../msmarco/msmarco-passagetest2020-top1000_spladedoc3_10.tsv", "w") as fo:
    for line in f:
        qid, did, qtext, dtext = line[:-1].split("\t")
        if qid not in q_toks:
            continue
        else:
            qtext = f"{qtext} [SEP] [unused2] {' '.join([x[0] for x in q_toks[qid][:10]])}"
            fo.write(f"{qid}\t{did}\t{qtext}\t{dtext}\n")
'''
with open("../msmarco/queries.dev.tsv") as f, open("../msmarco/queries.dev.spladedoc3_10.tsv", "w") as fo:
    for line in f:
        qid, qtext = line[:-1].split("\t")
        if qid not in q_toks:
            fo.write(line)
        else:
            qtext = f"{qtext} [SEP] [unused2] {' '.join([x[0] for x in q_toks[qid][:10]])}"
            fo.write(f"{qid}\t{qtext}\n")

'''