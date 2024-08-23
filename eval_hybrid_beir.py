from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir import util, LoggingHandler
import sys
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.lexical import BM25Search as BM25
from beir_models import Splade, BEIRSpladeModel,BEIRColBERT, BEIRTeacher
from transformers import AutoModelForMaskedLM, AutoTokenizer
from beir.reranking import Rerank
import numpy as np
from collections import defaultdict
import json
import os

k1=120 #rerank k
k2=60 # retrieval k

top_k = 100

def rankfuse(ranklist1, ranklist2, k1=60,k2=60,top_k=100):
    top_run = defaultdict(dict)
    for q in ranklist1:
        res = sorted(ranklist1[q].items(), key=lambda x: -x[1])[:top_k]
        for idx, item in enumerate(res):
            top_run[q][item[0]] = 1.0/(int(idx+1) + k1)

    for q in ranklist2:
        res = sorted(ranklist2[q].items(), key=lambda x: -x[1])[:top_k]
        for idx, item in enumerate(res):
            if item[0] in top_run[q]:
                top_run[q][item[0]] += 1.0/(int(idx+1) + k2)
            else:
                top_run[q][item[0]] = 1.0/(int(idx+1) + k2)

    return top_run

splade_dir = sys.argv[1]
colbert_dir = sys.argv[2]



retrieval_model = Splade(splade_dir)
retrieval_model.eval()
retrieval_tokenizer = AutoTokenizer.from_pretrained(splade_dir)
beir_splade = BEIRSpladeModel(retrieval_model, retrieval_tokenizer)

dres = DRES(beir_splade)
retriever = EvaluateRetrieval(dres, score_function="dot")

#beir_colbert = BEIRColBERT(f"training_with_sentence_transformers/output/{colbert_dir}/{epoch}/0_ColBERTTransformer",qlen=32,dlen=256)

averages = []

#fo = open(f"beir_results/beir_spalde_colbert_baseline_ndcg10_{splade_dir.replace('/', '-')}_{colbert_dir.replace('/', '-')}{epoch}.txt", "a")

fo = open(f"beir_results/beir_klar_splade_colbert-kldiv-num1_ndcg10_k.txt", "a")
#fo = open(f"beir_results/beir_focal_splade_colbert_ndcg10_k.txt", "a")

qlen=32
dlen=256
for dataset in ["quora", "fever", "nq", "climate-fever","hotpotqa", "trec-covid", "webis-touche2020", "fiqa", "nfcorpus", "arguana", "scifact", "scidocs","dbpedia-entity"]: # "quora", "fever", "nq", fever, 4xlarge: "quora", "nq","climate-fever","hotpotqa" small: "trec-covid", "webis-touche2020", "fiqa", "nfcorpus", "arguana", "scifact", "scidocs","dbpedia-entity"
    print("start:", dataset)

    if dataset in ['arguana']:
        qlen = 180
        dlen = 512
    elif dataset in ['quora']:
        dlen = 32
    elif dataset in ['hotpotqa']:
        qlen = 128
    elif dataset in ['nfcorpus']:
        dlen = 512
    elif dataset in ['trec-covid']:
        qlen = 64
    

    beir_colbert = BEIRColBERT(colbert_dir,qlen=qlen,dlen=dlen)
    reranker = Rerank(beir_colbert, batch_size=32)

    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
    out_dir = "dataset/{}".format(dataset)
    data_path = util.download_and_unzip(url, out_dir)
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
    
    if os.path.exists(f"beir_results/splade_sh_{dataset}.json"):
        retrieval_results = json.loads(open(f"beir_results/splade_sh_{dataset}.json").readline())
    else:
        print("retrieving using splade...")
        retrieval_results = retriever.retrieve(corpus, queries)
    
        with open(f"beir_results/splade_sh_{dataset}.json", "w") as f1:
            f1.write(json.dumps(retrieval_results))
    
    #retrieval_results = json.loads(open(f"beir_results/bm25_{dataset}.json").readline())
    
    ndcg_retrieval, _map, recall, precision = EvaluateRetrieval.evaluate(qrels, retrieval_results, [10])
    rerank_results = reranker.rerank(corpus, queries, retrieval_results, top_k=100)
    ndcg_rerank, _map, recall, precision = EvaluateRetrieval.evaluate(qrels, rerank_results, [10])
    for k in [(60,60),(60,120),(60,180),(60,240),(240,60),(180,60),(120,60)]:
        run = rankfuse(rerank_results, retrieval_results, k1=k[0], k2=k[1])
        ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(qrels, run, [10])
        print(dataset, ndcg)
        averages.append(ndcg['NDCG@10'])
        fo.write(f"{dataset}\t{ndcg_retrieval['NDCG@10']}\t{ndcg_rerank['NDCG@10']}\t{ndcg['NDCG@10']}\t{k[0]}\t{k[1]}\n")
        fo.flush()
fo.write(f"average\t{np.mean(averages)}\n")
fo.close()


