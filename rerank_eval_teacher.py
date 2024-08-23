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
from sentence_transformers import CrossEncoder



from training_with_sentence_transformers.losses import pairwise_dot_score


def _split_into_batches(features, bsize):
    batches = []
    for offset in range(0, features["input_ids"].size(0), bsize):
        batches.append({key: features[key][offset:offset+bsize] for key in features.keys()})

    return batches


agg = "max"
bsize = 128
num = 512

topk = open("../msmarco/wentai_splade_dev_top100.tsv")
#topk = open("../msmarco/wentai_splade_2020_top1000.tsv")

VALIDATION_METRIC = 'recip_rank'   #'recip_rank' #'ndcg_cut_10' 

qrel_file = "../msmarco/qrels.dev.tsv"
#qrel_file = "../msmarco/2020qrels-pass.txt"

qrels = defaultdict(dict)
with open(qrel_file) as f:
    for line in f:
        try:
            qid, _, did, rel = line.strip().split("\t")
        except:
            qid, _, did, rel = line.strip().split(" ")
        if int(rel) > 0:
            qrels[qid][did] = int(rel)



model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=256)
run = defaultdict(dict)

fo = open("../msmarco/output_dev_teacher.tsv", 'w')

for idx, line in enumerate(tqdm(topk)):
    qid, did, qtext, dtext = line.strip().split("\t")
    score = model.predict([(qtext, dtext)])
    run[qid][did] = score[0]
    fo.write(f"{qid}\t{did}\t{score[0]}\n")
    fo.flush()
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



