

import pytrec_eval
import sys
from statistics import mean
from collections import defaultdict
filename = sys.argv[1]

qrel_file = sys.argv[2] 

qrels = defaultdict(dict)
with open(qrel_file) as f:
    for line in f:
        try:
            qid, _, did, rel = line.strip().split("\t")
        except:
            qid, _, did, rel = line.strip().split(" ")
        if int(rel) > 0:
            qrels[qid][did] = int(rel)


p1 = float(sys.argv[3])
p2 = float(sys.argv[4])
p3 = float(sys.argv[5])
p4 = float(sys.argv[6]) 
p5 = float(sys.argv[7]) 
run = defaultdict(dict)

with open(filename) as f:
    for line in f:
        qid, did, s1, s2, s3, s4, s5 = line.strip().split("\t")
        run[qid][did] = p1 * float(s1) + p2 * float(s2) + p3 * float(s3) + p4 * float(s4) + p5 * float(s5)
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
