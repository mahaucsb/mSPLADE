from collections import defaultdict
import pytrec_eval
from statistics import mean
import scipy.stats as stats

combine_type = 'none'

VALIDATION_METRICS = {'ndcg_cut_10', 'ndcg_cut_5','ndcg_cut_20', 'recall_1000'}     #'recip_rank' #'ndcg_cut_10' 
top_k = 10
for k1 in [60]:
    for k2 in [240]:
        for yr in ['19','20']:
        
            qrel_file = f"../msmarco/20{yr}qrels-pass.txt" #"../msmarco/qrels.dev.tsv"  f"../msmarco/20{yr}qrels-pass.txt"
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
          
            with open(f"../msmarco/retrieval_trec/wentai_splade_20{yr}.trec") as f: #f"../msmarco/wentai_splade_20{yr}.trec", "../msmarco/wentai_splade_dev_top1000.trec"  f"../msmarco/splade-remove/test{yr}.remove60.1000", f"../msmarco/yifan_bm25_top1000.psg.20{yr}.trec.trec"
                for line in f:
                    qid, _, did, rank, _, _ = line.split("\t")
                    run[qid][did] = 1.0/(int(rank) + k1)
            
            run_retriever = {q: list(run[q].items()) for q in run}
            
            file_name = f"../msmarco/retrieval_trec/run.msmarco-passage.ance.bf.20{yr}.tsv" #"output_{yr}_colbert_yifan_bm25_000.run.json"
            if combine_type == "weight":
                sub_run_prf = defaultdict(dict)
                with open(file_name + ".prf") as f: 
                    for line in f:
                        qid, did, score = line.split("\t")
                        sub_run_prf[qid][did] = float(score)
                       
                sub_run = defaultdict(list)
                with open(file_name) as f:
                    for line in f:
                        qid, did, score = line.split("\t")
                        sub_run[qid].append((did, float(score) + 0.1 * sub_run_prf[qid][did]))
                for q in sub_run:
                    sub_run[q] = sorted(sub_run[q], key=lambda x: -x[1])


            elif combine_type in ["none", "rrf"]:
                sub_run = defaultdict(list)
                with open(file_name) as f: 
                    for line in f:
                        qid, did, score = line.split("\t")
                        sub_run[qid].append((did, float(score)))

                for q in sub_run:
                    sub_run[q] = sorted(sub_run[q], key=lambda x: x[1])
                if combine_type == "rrf":
                    sub_run_prf = defaultdict(list)
                    with open(file_name + ".prf") as f: #f"output_{yr}_colspla_from_colbert_3e-6_negpersys20_200k.run.json.tsv"
                        for line in f:
                            qid, did, score = line.split("\t")
                            sub_run_prf[qid].append((did, float(score)))

                    for q in sub_run_prf:
                        sub_run_prf[q] = sorted(sub_run_prf[q], key=lambda x: -x[1])

            
            for q in sub_run:
                x1 = [x[0] for x in run_retriever[q]]
                x2 = [x[0] for x in sub_run[q]]
                tau, p_value = stats.kendalltau(x1[:10], x2[:10])

                for idx, item in enumerate(sub_run[q]):
                    if item[0] in run[q]:
                        run[q][item[0]] += 1.0/(int(idx+1) + k2)
                    else:
                        run[q][item[0]] = 1.0/(int(idx+1) + k2)

                if combine_type in ["none", "weight"]:
                    threshold = -1
                else:
                    threshold = 1
                if tau < threshold:
                    k3 = 240
                    for idx, item in enumerate(sub_run_prf[q]):
                        if item[0] in run[q]:
                            run[q][item[0]] += 1.0/(int(idx+1) + k3)
                        else:
                            run[q][item[0]] = 1.0/(int(idx+1) + k3)
                else:
                    continue
                

            trec_eval = pytrec_eval.RelevanceEvaluator(qrels, VALIDATION_METRICS)
            eval_scores = trec_eval.evaluate(run)
            for VALIDATION_METRIC in VALIDATION_METRICS:
                print(k1,k2, yr, VALIDATION_METRIC, mean([d[VALIDATION_METRIC] for d in eval_scores.values()]))


VALIDATION_METRICS = {'recip_rank','recall_1000'}     #'recip_rank' #'ndcg_cut_10' 
top_k = 10
for k1 in [60]:
    for k2 in [240]:
        for yr in ['dev']:
        
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

            run = defaultdict(dict)
            
            with open("../msmarco/retrieval_trec/wentai_splade_dev_top1000.trec") as f: #"../msmarco/yifan_bm25_top1000.psg.dev.trec.trec") as f: 
                for line in f:
                    qid, _, did, rank, _, _ = line.split("\t")
                    run[qid][did] = 1.0/(int(rank) + k1)
            
            run_retriever = {q: list(run[q].items()) for q in run}
            
            file_name = "../msmarco/retrieval_trec/run.msmarco-passage.ance.bf.dev.tsv"
            
            if combine_type == "weight":
                sub_run_prf = defaultdict(dict)
                with open(file_name + ".prf") as f: 
                    for line in f:
                        qid, did, score = line.split("\t")
                        sub_run_prf[qid][did] = float(score)
                       
                sub_run = defaultdict(list)
                with open(file_name) as f:
                    for line in f:
                        qid, did, score = line.split("\t")
                        sub_run[qid].append((did, float(score) + 0.1 * sub_run_prf[qid][did]))
                for q in sub_run:
                    sub_run[q] = sorted(sub_run[q], key=lambda x: -x[1])


            elif combine_type in ["none", "rrf"]:
                sub_run = defaultdict(list)
                with open(file_name) as f: #f"output_{yr}_colspla_from_colbert_3e-6_negpersys20_200k.run.json.tsv"
                    for line in f:
                        qid, did, score = line.split("\t")
                        sub_run[qid].append((did, float(score)))

                for q in sub_run:
                    sub_run[q] = sorted(sub_run[q], key=lambda x: x[1])
                if combine_type == "rrf":
                    sub_run_prf = defaultdict(list)
                    with open(file_name + ".prf") as f: #f"output_{yr}_colspla_from_colbert_3e-6_negpersys20_200k.run.json.tsv"
                        for line in f:
                            qid, did, score = line.split("\t")
                            sub_run_prf[qid].append((did, float(score)))

                    for q in sub_run_prf:
                        sub_run_prf[q] = sorted(sub_run_prf[q], key=lambda x: -x[1])

                        
            for q in sub_run:
                x1 = [x[0] for x in run_retriever[q]]
                x2 = [x[0] for x in sub_run[q]]
                tau, p_value = stats.kendalltau(x1[:10], x2[:10])

                for idx, item in enumerate(sub_run[q]):
                    if item[0] in run[q]:
                        run[q][item[0]] += 1.0/(int(idx+1) + k2)
                    else:
                        run[q][item[0]] = 1.0/(int(idx+1) + k2)

                if combine_type in ["none", "weight"]:
                    threshold = -1
                else:
                    threshold = 1
                if tau < threshold:
                    k3 = 240
                    for idx, item in enumerate(sub_run_prf[q]):
                        if item[0] in run[q]:
                            run[q][item[0]] += 1.0/(int(idx+1) + k3)
                        else:
                            run[q][item[0]] = 1.0/(int(idx+1) + k3)
                else:
                    continue
            
            for VALIDATION_METRIC in VALIDATION_METRICS:
                if VALIDATION_METRIC == "recip_rank":
                    top_run = defaultdict(dict)
                    for q in run:
                        docs = sorted(run[q].items(), key=lambda x: -x[1])
                        for item in docs[:top_k]:
                            top_run[q][item[0]] = item[1]

                    trec_eval = pytrec_eval.RelevanceEvaluator(qrels, {VALIDATION_METRIC})
                    eval_scores = trec_eval.evaluate(top_run)
                    print(k1,k2, yr, mean([d[VALIDATION_METRIC] for d in eval_scores.values()]))

                else:
                    top_run = defaultdict(dict)
                    for q in run:
                        docs = sorted(run[q].items(), key=lambda x: -x[1])
                        for item in docs:
                            top_run[q][item[0]] = item[1]

                    trec_eval = pytrec_eval.RelevanceEvaluator(qrels, {VALIDATION_METRIC})
                    eval_scores = trec_eval.evaluate(top_run)
                    print(k1,k2, yr, mean([d[VALIDATION_METRIC] for d in eval_scores.values()]))




