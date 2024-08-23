import os
import random

from colbert.utils.parser import Arguments
from colbert.utils.runs import Run

from colbert.evaluation.loaders import load_colbert, load_topK, load_qrels
from colbert.evaluation.loaders import load_queries, load_topK_pids, load_collection
from colbert.evaluation.metrics import evaluate_recall
from colbert.modeling.inference import ModelInference
import os
import random
import time
import torch
import torch.nn as nn
import numpy as np

from transformers import AdamW
from colbert.utils.runs import Run
from colbert.utils.amp import MixedPrecisionManager

from colbert.training.lazy_batcher import LazyBatcher
from colbert.training.eager_batcher import EagerBatcher
from colbert.parameters import DEVICE

from colbert.modeling.colbert import ColBERT
from colbert.utils.utils import print_message
from colbert.training.utils import print_progress, manage_checkpoints
import inspect

def main():
    random.seed(12345)

    parser = Arguments(description='Exhaustive (slow, not index-based) evaluation of re-ranking with ColBERT.')

    parser.add_model_parameters()
    parser.add_model_inference_parameters()
    parser.add_reranking_input()

    parser.add_argument('--depth', dest='depth', required=False, default=None, type=int)

    args = parser.parse()

    if args.distributed:
        assert args.bsize % args.nranks == 0, (args.bsize, args.nranks)
        assert args.accumsteps == 1
        args.bsize = args.bsize // args.nranks

        print("Using args.bsize =", args.bsize, "(per process) and args.accumsteps =", args.accumsteps)

    print(inspect.getsource(ColBERT.query))
    args.colbert = ColBERT.from_pretrained('bert-base-uncased',
                                      query_maxlen=args.query_maxlen,
                                      doc_maxlen=args.doc_maxlen,
                                      dim=args.dim,
                                      similarity_metric=args.similarity,
                                      mask_punctuation=args.mask_punctuation)

    query = 'how much does it cost to change a jetblue flight'
    passages = ['6.4k Views. Because JetBlue does not overbook their flights like many carriers, a cancelled flight increases the potential that the seat be left unsold at time of flight. JetBlue will generally require a change/cancel fee of $100 to offset the potential losses incurred.JetBlue does offer refundable fares for customers that require flexibility in their schedule and will also look at individual customer concerns on a case by case basis..4k Views. Because JetBlue does not overbook their flights like many carriers, a cancelled flight increases the potential that the seat be left unsold at time of flight. JetBlue will generally require a change/cancel fee of $100 to offset the potential losses incurred.']
    args.colbert = args.colbert.to(DEVICE)
    args.colbert.eval()
    #args.qrels = load_qrels(args.qrels)
    #args.queries = load_queries(args.queries)
    #args.collection = load_collection(args.collection)
    #args.topK_pids, args.qrels = load_topK_pids(args.topK, args.qrels)
    
    args.inference = ModelInference(args.colbert, amp=args.amp)
    
    input_ids, attention_mask = args.inference.query_tokenizer.tensorize([query])
    print(input_ids)
    print(attention_mask)
    Q = args.colbert.bert(input_ids.to(DEVICE), attention_mask=attention_mask.to(DEVICE))[0]
    #Q = args.inference.queryFromText([query])
    
    input_ids, attention_mask = args.inference.doc_tokenizer.tensorize(passages)
    D = args.colbert.bert(input_ids.to(DEVICE), attention_mask=attention_mask.to(DEVICE))[0]
    #print(args.inference.doc_tokenizer.tensorize(passages))
    #D_ = args.inference.docFromText(passages, bsize=args.bsize)
    
    #scores = args.colbert.score(Q, D_).cpu()
    
    print(Q)
    #print(D_)
    #print(scores)
    print(Q.shape)
    print(D)
    print(D.shape)


if __name__ == "__main__":
    main()

