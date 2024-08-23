

import pytrec_eval
import sys
from statistics import mean
from collections import defaultdict
qrel_file = sys.argv[1] 

filename = sys.argv[2]


qrels = defaultdict(dict)
with open(qrel_file) as f:
    for line in f:
        try:
            qid, _, did, rel = line.strip().split("\t")
        except:
            qid, _, did, rel = line.strip().split(" ")
        if int(rel) > 0:
            qrels[qid][did] = int(rel)


run = defaultdict(dict)

with open(filename) as f:
    for line in f:
        qid, did, score = line.strip().split("\t")
        run[qid][did] = float(score)
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
