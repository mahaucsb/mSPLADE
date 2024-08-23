#FROM Sentence-BERT(https://github.com/UKPLab/sentence-transformers/blob/afee883a17ab039120783fd0cffe09ea979233cf/examples/training/ms_marco/train_bi-encoder_margin-mse.py) with minimal changes.
#Original License Apache2, NOTE: Trained MSMARCO models are NonCommercial (from dataset License)

import sys
import json
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, LoggingHandler, util, evaluation, InputExample
from sbert_traincq import SentenceTransformerCQ
import models
import logging
from datetime import datetime
import gzip
import os
import tarfile
import tqdm
from torch.utils.data import Dataset
import random
from shutil import copyfile
import pickle
import argparse
import losses
import torch
from collections import defaultdict
from data import MSMARCODataset
import cq_models
#from evaluate_training import evaluate_trainining
#import transformers

#from colbert_model import DETeacher

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

parser = argparse.ArgumentParser()
parser.add_argument("--train_batch_size", default=64, type=int)
parser.add_argument("--max_seq_length", default=256, type=int)
parser.add_argument("--model_name", default="distilbert-base-uncased", type=str)
parser.add_argument("--max_passages", default=0, type=int)
parser.add_argument("--epochs", default=30, type=int)
parser.add_argument("--negs_to_use", default=None, help="From which systems should negatives be used ? Multiple systems seperated by comma. None = all")
parser.add_argument("--warmup_steps", default=1000, type=int)
parser.add_argument("--lr", default=2e-5, type=float)
parser.add_argument("--num_negs_per_system", default=5, type=int)
parser.add_argument("--use_all_queries", default=False, action="store_true")
parser.add_argument("--accum_iter", default=1,type=int)
parser.add_argument("--ce_score_margin", default=3.0, type=float)
parser.add_argument("--denoise", default=False, action="store_true")
parser.add_argument("--deteacher", default=False, action="store_true")
parser.add_argument("--nway", default=1, type=int)
parser.add_argument("--alpha", default=0.0, type=float)
parser.add_argument("--gamma", default=1.0, type=float)
parser.add_argument("--weight_option", default='default', type=str)
parser.add_argument("--ib_p", default=0.2, type=float)
parser.add_argument("--beta_p", default = 0.0, type=float, help="For kldiv + added penaulty (klar), this is the weight on positive penalty.")
parser.add_argument("--beta", default = 0.0, type = float, help="the fixed penaulty for negative doc on kldiv_multipos_focal")
parser.add_argument("--sample_upweight", default = 0.0, type=float, help="For kldiv + added penaulty (klar), this is the sample upweight on good teacher (1+sample_upweight * p/n ratio).")
parser.add_argument("--dim", default = 768, type=int, help="colbert hidden layer")
parser.add_argument("--cq_model", type=str)
parser.add_argument("--loss_type", type=str, default = 'kldiv')
parser.add_argument("--teacher_type", type=str, default = 'ce')
parser.add_argument("--M", type=int, default = 16)
parser.add_argument("--K", type=int, default = 256)
parser.add_argument("--hidden", type=int, default = 128)

args = parser.parse_args()

logging.info(str(args))

train_batch_size = args.train_batch_size  # Increasing the train batch size generally improves the model performance, but requires more GPU memory
model_name = args.model_name
max_passages = args.max_passages
ce_score_margin = args.ce_score_margin
max_seq_length = args.max_seq_length  # Max length for passages. Increasing it implies more GPU memory needed
num_negs_per_system = args.num_negs_per_system  # We used different systems to mine hard negatives. Number of hard negatives to add from each system
num_epochs = args.epochs  # Number of epochs we want to train
#train_query_file = "../../msmarco/training_queries/train_queries_distill_splade_colbert_0.json"
train_query_file = "../../msmarco/train_queries_distill_splade_colbert_0.json"

# Load our embedding model
logging.info("Create new SBERT model")
word_embedding_model = models.ColBERTTransformerCQ(model_name, max_seq_length=max_seq_length, dim = args.dim)
print(len(word_embedding_model.tokenizer))
tokens = ["[unused0]", "[unused1]", "[unused2]"] #[unused0] for query, [unused1] for doc, [unused2] for query expansion
word_embedding_model.tokenizer.add_tokens(tokens, special_tokens=True)
print(len(word_embedding_model.tokenizer))
code_learner = cq_models.CodeLearner(emb_size = args.dim, M = args.M, K = args.K, hidden_size = args.hidden)

checkpoint = torch.load(os.path.join(model_name, "checkpoint.pt"), map_location='cpu')
word_embedding_model.load_state_dict(checkpoint)
word_embedding_model.to('cuda')
word_embedding_model.save_nocontext_embedding()

code_learner.load_state_dict(torch.load(args.cq_model, map_location='cpu'))
code_learner.to('cuda')
model = SentenceTransformerCQ(modules=[word_embedding_model, code_learner])
model_save_path = f'output/colbert_cq_test_fromkl_{args.loss_type}_{args.teacher_type}_{args.M}-{args.K}-{args.hidden}-{datetime.now().strftime("%Y-%m-%d")}'

# Write self to patha
os.makedirs(model_save_path, exist_ok=True)

train_script_path = os.path.join(model_save_path, 'train_script.py')
copyfile(__file__, train_script_path)
with open(train_script_path, 'a') as fOut:
    fOut.write("\n\n# Script was called via:\n#python " + " ".join(sys.argv))

### Now we read the MS MARCO dataset

data_folder = '../../msmarco'
#data_folder = '/home/ec2-user/ebs/msmarco'


#### Read the corpus file containing all the passages. Store them in the corpus dict
corpus = {}  # dict in the format: passage_id -> passage. Stores all existing passages
collection_filepath = os.path.join(data_folder, 'collection.tsv')
if not os.path.exists(collection_filepath):
    tar_filepath = os.path.join(data_folder, 'collection.tar.gz')
    if not os.path.exists(tar_filepath):
        logging.info("Download collection.tar.gz")
        util.http_get('https://msmarco.blob.core.windows.net/msmarcoranking/collection.tar.gz', tar_filepath)

    with tarfile.open(tar_filepath, "r:gz") as tar:
        tar.extractall(path=data_folder)

logging.info("Read corpus: collection.tsv")
with open(collection_filepath, 'r', encoding='utf8') as fIn:
    for line in fIn:
        pid, passage = line.strip().split("\t")
        pid = int(pid)
        corpus[pid] = passage

### Read the train queries, store in queries dict

queries = {}  # dict in the format: query_id -> query. Stores all training queries
queries_filepath = os.path.join(data_folder, 'queries.train.tsv')
'''
if not os.path.exists(queries_filepath):
    tar_filepath = os.path.join(data_folder, 'queries.tar.gz')
    if not os.path.exists(tar_filepath):
        logging.info("Download queries.tar.gz")
        util.http_get('https://msmarco.blob.core.windows.net/msmarcoranking/queries.tar.gz', tar_filepath)

    with tarfile.open(tar_filepath, "r:gz") as tar:
        tar.extractall(path=data_folder)
'''
with open(queries_filepath, 'r', encoding='utf8') as fIn:
    for line in fIn:
        qid, query = line.strip().split("\t")
        qid = int(qid)
        queries[qid] = query



ce_scores_file = os.path.join(data_folder, 'cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz')
if not os.path.exists(ce_scores_file):
    logging.info("Download cross-encoder scores file")
    util.http_get('https://huggingface.co/datasets/sentence-transformers/msmarco-hard-negatives/resolve/main/cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz', ce_scores_file)
with gzip.open(ce_scores_file, 'rb') as fIn:
    ce_scores = pickle.load(fIn)

#ce_scores_file = "../../msmarco/simlm_cross-encoder_scores.json"
#logging.info("Load CrossEncoder scores dict")
#with gzip.open(ce_scores_file, 'rb') as fIn:
#    ce_scores = pickle.load(fIn)

# As training data we use hard-negatives that have been mined using various systems
hard_negatives_filepath = os.path.join(data_folder, 'msmarco-hard-negatives-splade.jsonl.gz')
if not os.path.exists(hard_negatives_filepath):
    logging.info("Download cross-encoder scores file")
    util.http_get('https://huggingface.co/datasets/sentence-transformers/msmarco-hard-negatives/resolve/main/msmarco-hard-negatives.jsonl.gz', hard_negatives_filepath)

logging.info("Read hard negatives train file")

train_queries = dict()
#with open("output/colbert_dynamic_fromwarmup_num50_marginkldiv_position5-batch_size_16-2022-07-13_22-18-35/num1/train_queries.json") as f:
with open(train_query_file) as f: #training_queries_splade_max_156000_ceclean.json
    for line in f:
        qid = line.split("\t")[0]
        train_queries[qid] = json.loads(line.split("\t")[1])
        train_queries[qid]['query'] = queries[int(qid)]
        #train_queries[line.split("\t")[0]]['neg'] = train_queries[line.split("\t")[0]]['neg'][:args.num_negs_per_system]
        pos_min_ce_score = min([ce_scores[int(qid)][int(pid[1])] for pid in train_queries[qid]['pos']])
        ce_score_threshold = pos_min_ce_score - ce_score_margin
        if args.denoise:
            train_queries[qid]['neg'] = [x for x in train_queries[qid]['neg'] if x[0] <= args.num_negs_per_system and ce_scores[int(qid)][int(x[1])] < ce_score_threshold]
        else:
            train_queries[qid]['neg'] = [x for x in train_queries[qid]['neg'] if x[0] <= args.num_negs_per_system]
        if len(train_queries[line.split("\t")[0]]['neg']) == 0:
            del train_queries[line.split("\t")[0]]
            continue


train_loss = losses.DistillLossCQColBERT(model=model, teacher = args.teacher_type, loss_type = args.loss_type)

# For training the SentenceTransformer model, we need a dataset, a dataloader, and a loss used for training.
train_dataset = MSMARCODataset(queries=train_queries, corpus=corpus, ce_scores=ce_scores, loss_type='kldiv', num_neg=args.nway, topk = 20)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size, drop_last=True)
logging.info("Start model fitting...")

model.fit(train_objectives=[(train_dataloader, train_loss)],
            epochs=num_epochs,
            warmup_steps=args.warmup_steps,
            use_amp=True,
            checkpoint_path=model_save_path,
            checkpoint_save_steps=100,
            optimizer_params = {'lr': args.lr},
            accum_iter = args.accum_iter,
            save_loss = True)
            #schedulers = [scheduler],
            #optimizers = [optimizer])

    
