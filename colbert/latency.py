import os
import random

from colbert.utils.parser import Arguments
from colbert.utils.runs import Run

from colbert.evaluation.loaders import load_colbert, load_topK, load_qrels
from colbert.evaluation.loaders import load_queries, load_topK_pids, load_collection
from colbert.evaluation.ranking import evaluate
from colbert.evaluation.metrics import evaluate_recall
import torch
import os
import random
import time
import torch
import torch.nn as nn

from itertools import accumulate
from math import ceil

from colbert.utils.runs import Run
from colbert.utils.utils import print_message

from colbert.evaluation.metrics import Metrics
from colbert.evaluation.ranking_logger import RankingLogger
from colbert.modeling.inference import ModelInference

from colbert.evaluation.slow import *
import numpy as np
import faiss
import random
import time



def main():
    random.seed(12345)

    parser = Arguments(description='Exhaustive (slow, not index-based) evaluation of re-ranking with ColBERT.')

    parser.add_model_parameters()
    parser.add_model_inference_parameters()
    parser.add_reranking_input()

    parser.add_argument('--depth', dest='depth', required=False, default=None, type=int)
    parser.add_argument('--docids', default = [], type = list)

    args = parser.parse()
    
    if args.quantizer in ['opq', 'pq', 'rq', 'lsq']:
        args.collection = load_collection_code(args.collection, "code")
    elif args.use_codebook is not None:
        args.collection = load_collection_code(args.collection, "neural")
    else:
        args.collection = load_collection_code(args.collection)  
        
    if args.quantizer is not None:

        ################## OPQ ###################
        if args.dtype == 'doc':
            embeddings = torch.load("/home/ec2-user/efs/WordEmbedding2Code/data/index_MSMARCO.doc.L2.32x300k_0_emb_glove_embeddings.pt", map_location='cpu')
        else:
            embeddings = torch.load("/home/ec2-user/efs/compositional_code_learning/data/unnormalize_embedding_colbert_cos_glove_embeddings.pt", map_location='cpu')
        #https://github.com/facebookresearch/faiss/issues/411
        d = args.hidden_dim # data dimension
        m = args.code_book_len   # code size (bytes)
        k = args.cluster_num
        kbit = int(np.log(k)/np.log(2))

        x = embeddings.detach().numpy().astype('float32')#[:400000]

        if args.quantizer == "opq":
            pq = faiss.ProductQuantizer(d, m, kbit)
            quantizer = faiss.OPQMatrix(d, m)
            quantizer.pq = pq
            quantizer.train(x)
        elif args.quantizer == "pq":
            pq = faiss.ProductQuantizer(d, m, kbit)
            quantizer = pq
            quantizer.train(x)
        elif args.quantizer == "rq":
            quantizer = faiss.ResidualQuantizer(d,m,kbit)
            quantizer.train(x)
        elif args.quantizer == 'lsq':
            quantizer = faiss.LocalSearchQuantizer(d,m,kbit)
            quantizer.train(x)
        del x
        ###########################################
        
    

    with Run.context():
        args.colbert, args.checkpoint = load_colbert(args)
        args.qrels = load_qrels(args.qrels)
        #if args.cbe == 0 and args.use_codebook is not None:
        #    args.colbert.codebook.load_state_dict(torch.load(args.use_codebook, map_location='cpu'))
        
        
        args.queries = load_queries(args.queries)
        args.topK_pids, args.qrels = load_topK_pids(args.topK, args.qrels)
        args.inference = ModelInference(args.colbert, amp=args.amp)
        t = time.time()
        if args.quantizer in ['opq', 'pq', 'rq', 'lsq']:
            rerank(args, quantizer)
        else:
            rerank(args)
        print("time is:", time.time()-t)
            
def load_collection_code(collection_file, load_type='emb'):
    collections = []
    k=4
    values = list(range(k))
    #PQ
    if load_type == 'code':
        for i in range(100):
            collections.append(random.choices(values, k=55 * 16))
    #CQ
    elif load_type == 'neural':
        
        for i in range(100):
            v = torch.randint(0, k, [55]).to(DEVICE)
            collections.append(v)
        '''
        for i in range(10000):
            collections.append(random.choices(values, k=55 * 16))
        '''
    #ColBERT
    else:
        for i in range(100):
            collections.append(torch.rand(55, 128).to(DEVICE))
    return collections

def rerank(args, quantizer = None):
    
    qrels, queries, topK_pids = args.qrels, args.queries, args.topK_pids
    
    collection = args.collection
    depth = args.depth
    # PQ, OPQ
    def qid2code(qid):
        #return np.array([collection[pid % 10000] for pid in topK_pids[qid][:depth]]).reshape(-1, 55, 16).astype('uint8')
        return torch.tensor([collection[pid % 100] for pid in topK_pids[qid][:depth]]).reshape(-1, 55, 16)
    # CQ
    def qid2neural(qid):
        #return torch.tensor([collection[pid % 10000] for pid in topK_pids[qid][:depth]]).reshape(-1, 16) #880/5
        ids = torch.cat([collection[pid % 100] for pid in topK_pids[qid][:depth]], 0) 
        return ids
    # ColBERT
    def qid2emb(qid):
        return torch.stack([collection[pid % 100] for pid in topK_pids[qid][:depth]]).float()

    ranking_logger = RankingLogger(Run.path, qrels=qrels)

    args.milliseconds = []
 
    with ranking_logger.context('ranking.tsv', also_save_annotations=(qrels is not None)) as rlogger:
        with torch.no_grad():
            keys = sorted(list(topK_pids.keys()))
            #random.shuffle(keys)

            for query_idx, qid in enumerate(keys):
                query = queries[qid]
                print_message(query_idx, qid, query, '\n')

                if args.quantizer is not None:
                    ranking = slow_rerank_latency(args, query, topK_pids[qid], qid2code(qid), quantizer)
                elif args.use_codebook is not None:
                    ranking = slow_rerank_latency(args, query, topK_pids[qid], qid2neural(qid))
                else:
                    ranking = slow_rerank_latency(args, query, topK_pids[qid], qid2emb(qid))
                
                #rlogger.log(qid, ranking, [0, 1])


                #print_message("#> checkpoint['batch'] =", args.checkpoint['batch'], '\n')
                #print("rlogger.filename =", rlogger.filename)

                
                #print("\n\n")

        #print("\n\n")
        #print(args.milliseconds[1:])
        #print('Avg Latency =', sum(args.milliseconds[1:]) / (len(args.milliseconds[1:])+0.01))
        #print("\n\n")

    #print('\n\n')


    
if __name__ == "__main__":
    
    main()
    

   
