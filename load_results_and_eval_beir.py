from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir import util, LoggingHandler
import sys
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.retrieval.evaluation import EvaluateRetrieval

from beir_models import Splade, BEIRSpladeModel,BEIRColBERT
from transformers import AutoModelForMaskedLM, AutoTokenizer
from beir.reranking import Rerank
import numpy as np
from collections import defaultdict
import json

k1=60
k2=60
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

colbert_dir = sys.argv[1]
epoch = sys.argv[2]

qlen = 128
dlen = 256
beir_colbert = BEIRColBERT(f"training_with_sentence_transformers/output/{colbert_dir}/{epoch}/0_ColBERTTransformer",\
                          qlen=qlen, dlen=dlen)
reranker = Rerank(beir_colbert, batch_size=32)
averages = []


for dataset in ["hotpotqa"]: #""arguana", "scifact", "scidocs", "climate-fever","dbpedia-entity", "fever","hotpotqa", "quora", "trec-covid", "webis-touche2020", "fiqa", "nfcorpus", "nq"
    print("start:", dataset)
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
    out_dir = "dataset/{}".format(dataset)
    data_path = util.download_and_unzip(url, out_dir)
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
    
    retrieval_results = json.loads(open(f"beir_results/splade_{dataset}.json").readline())

    ndcg_retrieval, _map, recall, precision = EvaluateRetrieval.evaluate(qrels, retrieval_results, [10])
    rerank_results = reranker.rerank(corpus, queries, retrieval_results, top_k=100)
    ndcg_rerank, _map, recall, precision = EvaluateRetrieval.evaluate(qrels, rerank_results, [10])
    run = rankfuse(rerank_results, retrieval_results)
    ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(qrels, run, [10])
    print(dataset, ndcg)
    averages.append(ndcg['NDCG@10'])
    print(qlen, dlen)
    print(f"{dataset}\t{ndcg_retrieval['NDCG@10']}\t{ndcg_rerank['NDCG@10']}\t{ndcg['NDCG@10']}\n")
    
