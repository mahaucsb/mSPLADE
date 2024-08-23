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
from training_with_sentence_transformers.models import MLMTransformerDense, CETransformerSeq
from training_with_sentence_transformers.losses import pairwise_dot_score
#import nvidia_smi
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def _split_into_batches(features, bsize):
    batches = []
    for offset in range(0, features["input_ids"].size(0), bsize):
        batches.append({key: features[key][offset:offset+bsize] for key in features.keys()})

    return batches

#nvidia_smi.nvmlInit()
#handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)

agg = "max"
bsize = 128
num = 512
model_type_or_dir = sys.argv[1] #"colspla-prf_from_colbert_3e-6_negpersys5" #"output/0_MLMTransformer"

model = AutoModelForSequenceClassification.from_pretrained('cross-encoder/ms-marco-MiniLM-L-6-v2')

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
model = CETransformerSeq(model_type_or_dir, max_seq_length=256)
#checkpoint = torch.load(os.path.join(model_type_or_dir, "checkpoint.pt"), map_location='cpu')
#model.load_state_dict(checkpoint)
    
model.eval()
model.to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_type_or_dir)
tokens = ["[unused0]", "[unused1]", "[unused2]"] #[unused0] for query, [unused1] for doc, [unused2] for query expansion
tokenizer.add_tokens(tokens, special_tokens=True)
# top k inference               

run = defaultdict(dict)

fo = open(sys.argv[4], 'w')
#fo_prf = open(sys.argv[4] + ".prf", 'w')
cur_qid = None
cur_qtext = None
#cur_qtext_prf = None
dtexts = []
qtexts = []
#qtexts_prf = []
dids = []
qids = []
for idx, line in enumerate(tqdm(topk)):
    qid, didt, qtext, dtext = line.strip().split("\t")
    qtext_info = qtext.split(" [SEP] [unused2] ")
    qtext = qtext_info[0]
    #if len(qtext_info) == 2:
    #    qtext_prf = qtext_info[1]
    #else:
    #    qtext_prf = None
    if len(qrels[qid]) == 0:
        continue
    if (idx+1) % num == 0:
        with torch.no_grad():
            d_features = tokenizer(qtexts, dtexts, return_tensors="pt", max_length=512,truncation=True,padding=True)
            d_features = _split_into_batches(d_features,bsize=bsize)
            
            all_scores = []
            #all_scores_prf = []
            for batch in d_features:
                out = model(batch)
                scores = out.logits.to('cpu')
                del out
                scores = scores[:,0].tolist()
                
                torch.cuda.empty_cache()
                
                torch.cuda.empty_cache()
                all_scores.extend(scores)
                
       
        for qid, did, score in zip(qids, dids, all_scores):
            fo.write(f"{qid}\t{did}\t{score}\n")
            
            fo.flush()

            run[qid][did] = score
                
        dtexts = []
        dids = []
        qtexts = []
        qids = []
        

    qtexts.append(qtext)
    dtexts.append(dtext)
    dids.append(didt)
    qids.append(qid)
    
    
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



