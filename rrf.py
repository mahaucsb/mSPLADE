from collections import defaultdict
import pytrec_eval
from statistics import mean
import scipy.stats as stats

combine_type = 'rrf'

VALIDATION_METRIC = 'ndcg_cut_10'     #'recip_rank' #'ndcg_cut_10' 
top_k = 1000
alpha = 0.05
for k1 in [120, 60]:
    for k2 in [120, 60]:
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

            run_temp = defaultdict(list)
            run_retriever = defaultdict(list)
            run = defaultdict(dict)
            #f"../msmarco/index_splade_num1_marginmse.psg.{yr}.trec.trec"
            
            #with open(f"../msmarco/wentai_splade_20{yr}.trec") as f: # f"../msmarco/splade_klfr_5-1_num1.20{yr}.trec.trec" f"../msmarco/wentai_splade_20{yr}.trec",   f"../msmarco/splade-remove/test{yr}.remove60.1000"  f"../msmarco/yifan_bm25_top1000.psg.20{yr}.trec.trec"
            #    for line in f:
            #        qid, _, did, rank, _, _ = line.split("\t")
            #        run[qid][did] = 1.0/(int(rank) + k1)
            
            #run_retriever = {q: list(run[q].items()) for q in run}
        
            
            retrieval_file = f"results/output_20{yr}_splade_updatemrr5-1.0_iter16000_16x2_%remove.tsv"
            with open(retrieval_file) as f: 
                for line in f:
                    qid, did, score = line.split("\t")
                    #qid, _, did, _, score, _ = line.split("\t")
                    run_temp[qid].append((did, float(score)))

            for q in run_temp:
                run_temp[q] = sorted(run_temp[q], key=lambda x: -x[1])
                run_retriever[q] = [(x[0], 1.0/(int(i + 1) + k1)) for i, x in enumerate(run_temp[q])]
                run[q] = {k:v for k,v in run_retriever[q]}
            #file_name = f"results/output_{yr}_colbert_distill_splade_iter100k_kldiv_focal_gamma4.0_1e-5.tsv"

            #file_name = f"output_fromkl_20{yr}_cq_klar4-128-newnorm_model_cyclic2e-4-1e-6_n128-16-4_new_kldiv_default_3000.tsv" #f"output_{yr}_colbert_iter90k_curriculum_l1_32x4_0.0.tsv"
            #file_name = f"results/output_20{yr}_colbert_dim128_ckl4-1.0_30k.tsv"
            file_name = f"output_20{yr}_colbert_distill_klar4_newnorm_128_45k.tsv" #f"output_{yr}_colbert_iter90k_curriculum_l1_32x4_0.0.tsv" output_20{yr}_colbert_splade_updatemrr_2k_sh5-1.0_42k.tsv
            
            if combine_type == "weight":
                sub_run_temp = defaultdict(dict)
                with open(retrieval_file) as f: 
                    for line in f:
                        qid, did, score = line.split("\t")
                        sub_run_temp[qid][did] = float(score)
                       
                sub_run = defaultdict(list)
                with open(file_name) as f:
                    for line in f:
                        qid, did, score = line.split("\t")
                        if did not in sub_run_temp[qid]:
                            continue
                        sub_run[qid].append((did, float(score) + alpha * sub_run_temp[qid][did]))
                
                for q in sub_run:
                    sub_run[q] = sorted(sub_run[q], key=lambda x: -x[1])

            
            elif combine_type in ["none", "rrf"]:
                sub_run = defaultdict(list)
                with open(file_name) as f: 
                    for line in f:
                        qid, did, score = line.split("\t")
                        sub_run[qid].append((did, float(score)))

                for q in sub_run:
                    sub_run[q] = sorted(sub_run[q], key=lambda x: -x[1])
                if combine_type == "rrf":
                    sub_run_temp = defaultdict(list)
                    with open(file_name) as f: #f"output_{yr}_colspla_from_colbert_3e-6_negpersys20_200k.run.json.tsv"
                        for line in f:
                            qid, did, score = line.split("\t")
                            sub_run_temp[qid].append((did, float(score)))

                    for q in sub_run_temp:
                        sub_run_temp[q] = sorted(sub_run_temp[q], key=lambda x: -x[1])

            
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
                    for idx, item in enumerate(sub_run_temp[q]):
                        if item[0] in run[q]:
                            run[q][item[0]] += 1.0/(int(idx+1) + k3)
                        else:
                            run[q][item[0]] = 1.0/(int(idx+1) + k3)
                else:
                    continue
                
            
            trec_eval = pytrec_eval.RelevanceEvaluator(qrels, {VALIDATION_METRIC})
            eval_scores = trec_eval.evaluate(run)
            print(k1,k2, yr, mean([d[VALIDATION_METRIC] for d in eval_scores.values()]))



VALIDATION_METRIC = 'recip_rank'     #'recip_rank' #'ndcg_cut_10' 
top_k = 10
for k1 in [120, 60]:  # retieval
    for k2 in [120, 60]:
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

            run_temp = defaultdict(list)
            run_retriever = defaultdict(list)
            run = defaultdict(dict)
            #f"../msmarco/index_splade_num1_marginmse.psg.{yr}.trec.trec"
            
            retrieval_file = f"results/output_dev_splade_updatemrr5-1.0_iter16000_16x2_%remove.tsv"
            with open(retrieval_file) as f: 
                for line in f:
                    qid, did, score = line.split("\t")
                    #qid, _, did, _, score, _ = line.split("\t")
                    run_temp[qid].append((did, float(score)))

            for q in run_temp:
                run_temp[q] = sorted(run_temp[q], key=lambda x: -x[1])
                run_retriever[q] = [(x[0], 1.0/(int(i + 1) + k1)) for i, x in enumerate(run_temp[q])]
                run[q] = {k:v for k,v in run_retriever[q]}


            #file_name = f"output_cq_fromkl_klar4-128-newnorm_model_cyclic2e-4-1e-6_n128-16-4_new_kldiv_default_3000.tsv"
            #file_name = f"results/output_dev_colbert_dim128_ckl4-1.0_30k.tsv"
            file_name = f"output_dev_colbert_distill_klar4_newnorm_128_45k.tsv"#"output_dev_colbert_splade_updatemrr_2k_sh5-1.0_42k.tsv"
            

            if combine_type == "weight":
                sub_run_temp = defaultdict(dict)
                with open(retrieval_file) as f: 
                    for line in f:
                        qid, did, score = line.split("\t")
                        sub_run_temp[qid][did] = float(score)
                       
                sub_run = defaultdict(list)
                with open(file_name) as f:
                    for line in f:
                        qid, did, score = line.split("\t")
                        if did not in sub_run_temp[qid]:
                            continue
                        sub_run[qid].append((did, float(score) + alpha * sub_run_temp[qid][did]))
                for q in sub_run:
                    sub_run[q] = sorted(sub_run[q], key=lambda x: -x[1])


            elif combine_type in ["none", "rrf"]:
                sub_run = defaultdict(list)
                with open(file_name) as f: #f"output_{yr}_colspla_from_colbert_3e-6_negpersys20_200k.run.json.tsv"
                    for line in f:
                        try:
                            qid, did, score = line.split("\t")
                            sub_run[qid].append((did, float(score)))
                        except:
                            print(line.split("\t"))

                for q in sub_run:
                    sub_run[q] = sorted(sub_run[q], key=lambda x: -x[1])
            
            if combine_type == "rrf":
                sub_run_temp = defaultdict(list)
                with open(file_name) as f: #f"output_{yr}_colspla_from_colbert_3e-6_negpersys20_200k.run.json.tsv"
                    for line in f:
                        qid, did, score = line.split("\t")
                        sub_run_temp[qid].append((did, float(score)))

                for q in sub_run_temp:
                    sub_run_temp[q] = sorted(sub_run_temp[q], key=lambda x: -x[1])

                        
            for q in sub_run:
                x1 = [x[0] for x in run_retriever[q]]
                x2 = [x[0] for x in sub_run[q]]
                tau, p_value = stats.kendalltau(x1[:10], x2[:10])

                for idx, item in enumerate(sub_run[q]):
                    if item[0] in run[q]:
                        run[q][item[0]] += 1.0/(int(idx+1) + k2)
                    else:
                        run[q][item[0]] = 1.0/(int(idx+1) + k2)

                if combine_type in ["rrf"]:
                    k3 = 240
                    for idx, item in enumerate(sub_run_temp[q]):
                        if item[0] in run[q]:
                            run[q][item[0]] += 1.0/(int(idx+1) + k3)
                        else:
                            run[q][item[0]] = 1.0/(int(idx+1) + k3)
                    
            
            top_run = defaultdict(dict)
            for q in run:
                docs = sorted(run[q].items(), key=lambda x: -x[1])
                for item in docs[:top_k]:
                    top_run[q][item[0]] = item[1]

            trec_eval = pytrec_eval.RelevanceEvaluator(qrels, {VALIDATION_METRIC})
            eval_scores = trec_eval.evaluate(top_run)
            print(k1,k2, yr, mean([d[VALIDATION_METRIC] for d in eval_scores.values()]))



