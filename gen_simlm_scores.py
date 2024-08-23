import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BatchEncoding, PreTrainedTokenizerFast
from transformers.modeling_outputs import SequenceClassifierOutput
import json
import tqdm
train_query_file = "../msmarco/train_queries_distill_splade_colbert_0.json"

queries = {}  # dict in the format: query_id -> query. Stores all training queries
queries_filepath = '../msmarco/queries.train.tsv'

with open(queries_filepath, 'r', encoding='utf8') as fIn:
    for line in fIn:
        qid, query = line.strip().split("\t")
        qid = int(qid)
        queries[qid] = query

corpus = {}  # dict in the format: passage_id -> passage. Stores all existing passages
collection_filepath = '../msmarco/collection.tsv'

with open(collection_filepath, 'r', encoding='utf8') as fIn:
    for line in fIn:
        pid, passage = line.strip().split("\t")
        pid = int(pid)
        corpus[pid] = passage

def encode(tokenizer: PreTrainedTokenizerFast,
           query, passage, title: str = '-') -> BatchEncoding:
    return tokenizer(query,
                     text_pair=['{}: {}'.format(title, p) for p in passage],
                     max_length=192,
                     padding=True,
                     truncation=True,
                     return_tensors='pt')

tokenizer = AutoTokenizer.from_pretrained('intfloat/simlm-msmarco-reranker')
model = AutoModelForSequenceClassification.from_pretrained('intfloat/simlm-msmarco-reranker')
model.eval()

n = 32
start_i = 90001
fo = open(f"test_{start_i}.json", "w")
linen = 0
with open(train_query_file) as f:
    for line in tqdm.tqdm(f):
        linen += 1
        if linen < start_i:
            continue
        qid = line.split("\t")[0]
        train_query = json.loads(line.split("\t")[1])
        top_dids = sorted(train_query['neg'], key = lambda x: x[0])[:30]
        for pair in train_query['pos']:
            if pair not in top_dids:
                top_dids.append(pair)


        with torch.no_grad():
            scores = dict()
            i = 0
            docs = []
            dids = []
            for did in top_dids:
                if i < n:
                    docs.append(corpus[int(did[1])])
                    dids.append(int(did[1]))
                else:
                    batch_dict = encode(tokenizer, [queries[int(qid)]] * len(docs),  docs)
                    docs = [corpus[int(did[1])]]
                    i = 0
                
                    outputs = model(**batch_dict, return_dict=True)
                    preds = outputs.logits.tolist()
                    
                    for d,s in zip(dids, preds):
                        scores[d] = s[0]
                    dids = [int(did[1])]
                i += 1

            batch_dict = encode(tokenizer, [queries[int(qid)]] * len(docs),  docs)
            outputs = model(**batch_dict, return_dict=True)
            preds = outputs.logits.tolist()
            for d,s in zip(dids, preds):
                scores[d] = s[0]

        fo.write(f"{qid}\t{json.dumps(scores)}\n")
        fo.flush()
fo.close()