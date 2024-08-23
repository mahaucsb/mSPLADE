from collections import defaultdict
import pytrec_eval
from statistics import mean
import scipy.stats as stats


top_k=10
VALIDATION_METRIC = 'recip_rank'     #'recip_rank' #'ndcg_cut_10' 
top_k = 10
for k1 in [60]:
    for k2 in [60]:
        for yr in ['dev']:
        
            qrel_file = f"../msmarco/qrels.dev.tsv" #"../msmarco/qrels.dev.tsv"  f"../msmarco/20{yr}qrels-pass.txt"
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
          
            with open(f"../msmarco/wentai_splade_dev_top1000.trec") as f: #f"../msmarco/wentai_splade_20{yr}.trec", "../msmarco/wentai_splade_dev_top1000.trec"  
                for line in f:
                    qid, _, did, rank, _, _ = line.split("\t")
                    run[qid][did] = 1.0/(int(rank) + k1)
            
            run_retriever = {q: list(run[q].items()) for q in run}
            
            file_name = f"results/output_{yr}_fromwarmup_marginkldiv5_position_dynamic_num0_170k.tsv"
            
            sub_run = defaultdict(list)
            with open(file_name) as f: #f"output_{yr}_colspla_from_colbert_3e-6_negpersys20_200k.run.json.tsv"
                for line in f:
                    qid, did, score = line.split("\t")
                    sub_run[qid].append((did, float(score)))


                    
            sub_run_ce = defaultdict(dict)
            sub_run_ce_order = defaultdict(list)
            with open(f"output_dev_fromwarmup_marginkldiv5_position_num1_50k.tsv") as f: 
                for line in f:
                    qid, did, score = line.split("\t")
                    sub_run_ce[qid][did] = float(score)
                       
            for q in sub_run_ce:
                sub_run_ce_order[q] = sorted([[k,v] for k,v in sub_run_ce[q].items()], key=lambda x: -x[1])
            
            for q in sub_run_ce:
                for idx, item in enumerate(sub_run_ce_order[q]):
                    sub_run_ce[q][item[0]] = idx
            
            for q in sub_run:
                sub_run[q] = sorted(sub_run[q], key=lambda x: -x[1])
            
            
            for q in sub_run:
                x1 = [x[0] for x in run_retriever[q]]
                x2 = [x[0] for x in sub_run[q]]
                tau, p_value = stats.kendalltau(x1[:10], x2[:10])

                for idx, item in enumerate(sub_run[q]):
                    if item[0] in run[q]:
                        run[q][item[0]] += 1.0/(int(idx+1) + k2)
                    else:
                        run[q][item[0]] = 1.0/(int(idx+1) + k2)
                
                
                orders = sorted([[k,v] for k,v in run[q].items()], key=lambda x: -x[1])
                
                k3 = 60
                topk = 1000
                for idx, item in enumerate(orders):
                    if idx < topk:
                        run[q][item[0]] += 1.0/(int(sub_run_ce[q][item[0]] +1) + k3)
                    else:
                        break
               
            top_run = defaultdict(dict)
            for q in run:
                docs = sorted(run[q].items(), key=lambda x: -x[1])
                for item in docs[:top_k]:
                    top_run[q][item[0]] = item[1]


            trec_eval = pytrec_eval.RelevanceEvaluator(qrels, {VALIDATION_METRIC})
            eval_scores = trec_eval.evaluate(top_run)
            print(k1,k2, yr, mean([d[VALIDATION_METRIC] for d in eval_scores.values()]))

