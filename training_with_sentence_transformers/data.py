from torch.utils.data import Dataset
import random
from sentence_transformers import InputExample
import torch
import numpy as np
import tqdm
# We create a custom MS MARCO dataset that returns triplets (query, positive, negative)
# on-the-fly based on the information from the mined-hard-negatives jsonl file.
ce_threshold = -3

def _split_into_batches(features, bsize):
    batches = []
    for offset in range(0, features["input_ids"].size(0), bsize):
        batches.append({key: features[key][offset:offset+bsize] for key in features.keys()})

    return batches
    

class MSMARCODataset(Dataset):
    def __init__(self, queries, corpus, ce_scores, num_neg = 1, loss_type = "marginmse", topk=20, model_type = "colbert", reeval = False, curmodel = None, tokenizer = None, temp=1):
        self.queries = queries
        self.queries_ids = list(queries.keys())
        self.corpus = corpus
        self.ce_scores = ce_scores
        self.num_neg = num_neg
        self.loss_type = loss_type
        if self.loss_type == "marginmse":
            assert(self.num_neg == 1)

        self.model_type = model_type
        for qid in self.queries:
            self.queries[qid]['neg'] = self.queries[qid]['neg'][:topk]
            random.shuffle(self.queries[qid]['neg'])
        self.iter_num = 0 
        self.temp = temp
        if reeval:
            curmodel.eval()
            with torch.no_grad():
                for qid in tqdm.tqdm(self.queries_ids):
                    query = self.queries[qid]
                    if self.model_type == "colbert":
                        q_features = tokenizer("[unused0] " + query['query'], return_tensors="pt").to('cuda')
                        q_features = curmodel(q_features)
                        token_rep_q = torch.nn.functional.normalize(q_features['last_layer_embeddings'], p=2, dim=2)
                    else:
                        q_features = tokenizer(query['query'], return_tensors="pt").to('cuda')
                        q_rep = curmodel[0].auto_model(**q_features)["logits"]
                        q_rep, _ = torch.max(torch.log(1 + torch.relu(q_rep)) * q_features["attention_mask"].unsqueeze(-1), dim=1)
                        q_rep = q_rep.squeeze() 
                        col = torch.nonzero(q_rep).squeeze().cpu().tolist()
                        weights = q_rep[col].cpu().tolist()
                        q_features = {k: v for k, v in zip(col, weights)}

                    dids = []
                    dtexts = []
                    
                    for did in self.queries[qid]['pos']:
                        dids.append(did[1])
                        dtexts.append(self.corpus[did[1]])
                    
                    for did in self.queries[qid]['neg']:
                        dids.append(did[1])
                        dtexts.append(self.corpus[did[1]])
                    if self.model_type == "colbert":
                        d_features = tokenizer(["[unused1] " + dtext for dtext in dtexts], return_tensors="pt", max_length=256,truncation=True,padding=True)
                    else:
                        d_features = tokenizer(dtexts, return_tensors="pt", max_length=256,truncation=True,padding=True)

                    d_features = _split_into_batches(d_features,bsize=32)
                    
                    d_scores = []
                    for batch in d_features:
                        for k in batch:
                            batch[k] = batch[k].to("cuda")
                        if self.model_type == "colbert":
                            d_batch = curmodel(batch)
                            d_mask = d_batch['attention_mask']
                            d_emb = d_batch['last_layer_embeddings']
                            del d_batch
                            d_mask = d_mask.unsqueeze(-1)
                            token_rep_d =  d_emb * d_mask
                            del d_mask, d_emb
                            token_rep_d = torch.nn.functional.normalize(token_rep_d)
                            scores =  (token_rep_q @ token_rep_d.permute(0,2,1)).max(2).values.sum(1).tolist()
                            torch.cuda.empty_cache()
                            d_scores.extend(scores)
                        else:
                            d_batch = curmodel[0].auto_model(**batch)["logits"]
                            d_batch, _ = torch.max(torch.log(1 + torch.relu(d_batch)) * batch["attention_mask"].unsqueeze(-1), dim=1)
                            d_batch = d_batch.squeeze().cpu().tolist()
                            #d_batch = curmodel(**batch).cpu().tolist()
                            for d_rep in d_batch:
                                d_rep = torch.tensor(d_rep)
                                d_col = torch.nonzero(d_rep).squeeze().cpu().tolist()
                                d_weights = d_rep[d_col].cpu().tolist()
                                
                                d_weight = sorted([(c, w) for c,w in zip(d_col, d_weights)], key = lambda x: -x[1])
                                d_weights = {k: v for k, v in d_weight}
                                score = 0
                                for k in q_features:
                                    if k in d_weights:
                                        score += d_weights[k] * q_features[k]
                                d_scores.append(score)
                                
                    sorted_dids = [x for x, _ in sorted(zip(dids, d_scores), key=lambda x: -x[1])]

                    new_pos = []
                    for did in self.queries[qid]['pos']:
                        did[0] = sorted_dids.index(did[1]) + 1
                        new_pos.append(did)
                    self.queries[qid]['pos']  = new_pos 
                    pos_ids = [x[1] for x in new_pos]

                    new_neg = []
                    for did in self.queries[qid]['neg']:
                        did[0] = sorted_dids.index(did[1]) + 1
                        if did[1] not in pos_ids:
                            new_neg.append(did)
                    self.queries[qid]['neg']  = new_neg
                    random.shuffle(self.queries[qid]['neg'])
                    
                    del new_pos, new_neg

            print("finish updating data...")

    def __getitem__(self, item):
        self.iter_num += 1
        query = self.queries[self.queries_ids[item]]
        if self.model_type == "colbert":
            query_text = "[unused0] " + query['query']
        else:
            query_text = query['query']

        qid = query['qid']

        if self.loss_type == "curriculum":
            pos_label = []
        if len(query['pos']) > 0:

            if self.loss_type not in ["kldiv_multipos_focal_focal", "kldiv_multipos", "ckl", "kldiv_multipos_ib", "kldiv_multipos_position", 'kldiv_multipos_position_focal', 'kldiv_multipos_ib_position_focal', 'klar', 'convexsh', 'kllog']:
                pos_id = query['pos'].pop(0)   #Pop positive and add at end
                if self.model_type == "colbert":
                    pos_text = "[unused1] " + self.corpus[pos_id[1]]
                else:
                    pos_text = self.corpus[pos_id[1]]
                query['pos'].append(pos_id)
                if self.loss_type == "curriculum":
                    pos_label.append(1/(pos_id[2]+1))
            else:
                pos_ids = query['pos'][:3]
                if self.model_type == "colbert":
                    pos_texts = ["[unused1] " + self.corpus[pos_id[1]] for pos_id in pos_ids]
                else:
                    pos_texts = [self.corpus[pos_id[1]] for pos_id in pos_ids]
                if self.loss_type == "curriculum":
                    pos_label.append(1/(pos_id[2]+1))
        

        elif self.loss_type not in ["kldiv_multipos_focal_focal", "kldiv_multipos", "ckl", "kldiv_multipos_ib", "kldiv_multipos_position", 'kldiv_multipos_position_focal','klar', 'convexsh', 'kllog']:   #We only have negatives, use two negs
            pos_id = query['neg'].pop(0)    #Pop negative and add at end
            if self.model_type == "colbert":
                pos_text = "[unused1] " + self.corpus[pos_id[1]]
            else:
                pos_text = self.corpus[pos_id[1]]
            query['neg'].append(pos_id)
        
        


        if self.loss_type in ["kldiv_multipos_focal_focal","kldiv_multipos", "ckl", "kldiv_multipos_ib", "kldiv_multipos_position", 'kldiv_multipos_position_focal','kldiv_multipos_ib_position_focal', 'klar', 'convexsh', 'kllog']:
            pos_score = [self.ce_scores[qid][pos_id[1]] for pos_id in pos_ids]
            pos_idx = [pos_id[0] for pos_id in pos_ids]
            pos_ce_idx = [pos_id[2] for pos_id in pos_ids]
            n_neg = self.num_neg + 1 - len(pos_ids)
        else:
            pos_score = self.ce_scores[qid][pos_id[1]]
            pos_idx = pos_id[0]
            pos_ce_idx = pos_id[2]
            n_neg = self.num_neg
            #Get negative passage
        
        #Get negative passage
        neg_texts = []
        neg_scores = []
        neg_idx = []
        neg_ce_idx = []
        
        if self.loss_type == "curriculum":
            neg_group = [0, 0, 0]
            neg_group_label = []

        for i in range(n_neg):
            neg_id = query['neg'].pop(0)    #Pop negative and add at end
            if self.model_type == "colbert":
                neg_text = "[unused1] " + self.corpus[neg_id[1]]
            else:
                neg_text = self.corpus[neg_id[1]]

            if self.loss_type == "curriculum":
                if sum(neg_group) == n_neg:
                    break

                if neg_id[2] < 10 and neg_group[0] < 2:
                    neg_score = self.ce_scores[qid][neg_id[1]]
                    neg_group[0] += 1
                    neg_group_label.append(1/(neg_id[2] + 1))
                elif neg_id[2] < 20 and neg_group[1] < 2:
                    neg_score = self.ce_scores[qid][neg_id[1]]
                    neg_group[1] += 1
                    neg_group_label.append(0.0)
                elif neg_group[2] < 1:
                    neg_score = self.ce_scores[qid][neg_id[1]]
                    neg_group[2] += 1
                    neg_group_label.append(-1)
                else:
                    i -= 1
                    continue

            elif neg_id[1] in self.ce_scores[qid]:
                neg_score = self.ce_scores[qid][neg_id[1]]
            else:
                i -= 1
                continue

            
            query['neg'].append(neg_id)
            neg_texts.append(neg_text)
            neg_scores.append(neg_score)
            neg_idx.append(neg_id[0])
            neg_ce_idx.append(neg_id[2])

        if self.loss_type == "curriculum" and len(neg_texts) < n_neg:
            while len(neg_texts) < n_neg:
                neg_id = query['neg'].pop(0)    #Pop negative and add at end
                if neg_id[0] not in neg_idx:
                    neg_group_label.append(-1)
                    query['neg'].append(neg_id)
                    neg_texts.append(neg_text)
                    neg_scores.append(neg_score)
                    neg_idx.append(neg_id[0])
                    neg_ce_idx.append(neg_id[2])
                
        
        if self.loss_type == "marginmse":
            return InputExample(texts=[query_text, pos_text, neg_texts[0]], label=pos_score-neg_scores[0])
        elif self.loss_type == "marginmse_ib":
            return InputExample(texts=[query_text, pos_text, neg_texts[0]], label=[pos_score,neg_scores[0]])
        elif self.loss_type in ["kldiv", "kldiv_focal", "kldiv_ib"]:
            target_score = torch.tensor([pos_score] + neg_scores)
            target_score = torch.nn.functional.log_softmax(target_score / self.temp)
            return InputExample(texts=[query_text, pos_text] + neg_texts, label=target_score.tolist()) # length of label is number of texts

        elif self.loss_type in ["kldiv_multipos", "ckl", "kldiv_multipos_ib", "klar", 'kllog', "kldiv_multipos_focal_focal"]:
            target_score = torch.tensor(pos_score + neg_scores)
            target_score = torch.nn.functional.log_softmax(target_score  / self.temp)
            return InputExample(texts=[query_text] + pos_texts + neg_texts, label=target_score.tolist() + [1] * len(pos_ids) + [0] * n_neg) # length of label is number of texts
        elif self.loss_type == "marginkldiv":
            target_score = torch.tensor([pos_score  - neg_score for neg_score in neg_scores])
            target_score = torch.nn.functional.log_softmax(target_score)
            return InputExample(texts=[query_text, pos_text] + neg_texts, label=target_score.tolist()) # length of label is number 
        elif self.loss_type == "marginmse_position":
            return InputExample(texts=[query_text, pos_text, neg_texts[0]], label=[pos_score-neg_scores[0], pos_idx, neg_idx[0]])
        elif self.loss_type in ["wce", "ce"]:
            return InputExample(texts=[query_text, pos_text] + neg_texts, label=[pos_idx] + neg_idx)
        elif self.loss_type == "marginkldiv_position":
            ce_diffs = [neg_score - pos_score for neg_score in neg_scores]
            target_score = torch.tensor(ce_diffs)
            target_score = torch.nn.functional.log_softmax(target_score)
            
            return InputExample(texts=[query_text, pos_text] + neg_texts, label=target_score.tolist() + [pos_idx] + neg_idx) # length of label is number 
        elif self.loss_type in ["kldiv_position", 'kldiv_position_focal', 'kldiv_ib_position_focal']:
            target_score = torch.tensor([pos_score] + neg_scores)
            target_score = torch.nn.functional.log_softmax(target_score)
            return InputExample(texts=[query_text, pos_text] + neg_texts, label=target_score.tolist() + [pos_idx] + neg_idx) # length of label is number 
        elif self.loss_type in ["kldiv_multipos_position", 'kldiv_multipos_position_focal','kldiv_multipos_ib_position_focal', 'kldiv_multipos_ib_position_focal', 'convexsh']:
            target_score = torch.tensor(pos_score + neg_scores)
            target_score = torch.nn.functional.log_softmax(target_score)
            return InputExample(texts=[query_text] + pos_texts + neg_texts, label=target_score.tolist() + pos_idx + neg_idx + [1] * len(pos_ids) + [0] * n_neg) # length of label is number 
        elif self.loss_type == "kldiv_position_reverse":
            target_score = -torch.tensor([pos_score] + neg_scores)
            target_score = torch.nn.functional.log_softmax(target_score)
            return InputExample(texts=[query_text, pos_text] + neg_texts, label=target_score.tolist() + [pos_idx] + neg_idx) # length of label is number 
        elif self.loss_type == "curriculum":
            labels = pos_label + neg_group_label
            texts =  [pos_text] + neg_texts
            idxes = [pos_idx] + neg_idx
            sample_data = sorted([(x,y,z) for x,y,z in zip(labels, texts, idxes)], key = lambda x: -x[0])

            return InputExample(texts=[query_text] + [x[1] for x in sample_data], label=[x[0] for x in sample_data] + [1/(x[2]+1) for x in sample_data]) # length of label is number 
        else:
            raise("Unrecogized loss type!")
            return 

    def __len__(self):
        return len(self.queries)


class MSMARCODatasetCE(Dataset):
    def __init__(self, queries, corpus, ce_scores, num_neg = 1, topk=20, loss_type = "marginkldiv_position"):
        self.queries = queries
        self.queries_ids = list(queries.keys())
        self.corpus = corpus
        self.ce_scores = ce_scores
        self.num_neg = num_neg
        self.loss_type = loss_type
       
      
        for qid in self.queries:
            self.queries[qid]['neg'] = self.queries[qid]['neg'][:topk]
            random.shuffle(self.queries[qid]['neg'])
        self.iter_num = 0 

    def __getitem__(self, item):
        self.iter_num += 1
        query = self.queries[self.queries_ids[item]]
        query_text = "[unused0] " + query['query']
        
        qid = query['qid']

        if len(query['pos']) > 0:
            pos_id = query['pos'].pop(0)   #Pop positive and add at end
            pos_text = "[unused1] " + self.corpus[pos_id[1]]
            query['pos'].append(pos_id)
        else:   #We only have negatives, use two negs
            pos_id = query['neg'].pop(0)    #Pop negative and add at end
            pos_text = "[unused1] " + self.corpus[pos_id[1]]
            query['neg'].append(pos_id)
        
        pos_score = self.ce_scores[qid][pos_id[1]]
        pos_idx = pos_id[0]
        pos_ce_idx = pos_id[2]
        #Get a negative passage
        neg_texts = []
        neg_scores = []
        neg_idx = []
        neg_ce_idx = []

        for i in range(self.num_neg):
            neg_id = query['neg'].pop(0)    #Pop negative and add at end
            neg_text = "[unused1] " + self.corpus[neg_id[1]]
            query['neg'].append(neg_id)
            neg_texts.append(neg_text)
            neg_score = self.ce_scores[qid][neg_id[1]]
            neg_scores.append(neg_score)
            neg_idx.append(neg_id[0])
            neg_ce_idx.append(neg_id[2])
            
        
        neg_texts = [f"{query_text} [SEP] {neg_text}" for neg_text in neg_texts]
        pos_text = f"{query_text} [SEP] {pos_text}" 
        #alpha = 0.2
        #weights = [alpha/(1+np.exp(neg_i - pos_idx)) + 1 for neg_i in neg_idx]
        
        if self.loss_type == "crossentropy":   
            return InputExample(texts=[pos_text] + neg_texts) # length of label is number 
        elif self.loss_type == "marginkldiv_position":
            ce_diffs = [pos_score  - neg_score for neg_score in neg_scores]
            target_score = torch.tensor(ce_diffs)
            target_score = torch.nn.functional.log_softmax(target_score)
            return InputExample(texts=[pos_text] + neg_texts, label=target_score.tolist() + [pos_idx] + neg_idx)
        

    def __len__(self):
        return len(self.queries)



