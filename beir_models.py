import logging
from typing import List, Dict, Union

import numpy as np
import torch
from numpy import ndarray
from torch import Tensor
from tqdm.autonotebook import trange
from transformers import AutoModelForMaskedLM,AutoTokenizer

from tqdm import tqdm
import os
import pytrec_eval
from collections import OrderedDict, defaultdict
from statistics import mean
import json
import sys
from training_with_sentence_transformers.models import  ColBERTTransformer
from training_with_sentence_transformers.losses import pairwise_dot_score
from cq_models_new import Code_Learner
from typing import Tuple
from sentence_transformers import CrossEncoder

def _split_into_batches(features, bsize):
    batches = []
    for offset in range(0, features["input_ids"].size(0), bsize):
        batches.append({key: features[key][offset:offset+bsize] for key in features.keys()})

    return batches

agg = "max"
bsize = 32

try:
    import sentence_transformers
    from sentence_transformers.util import batch_to_device
except ImportError:
    print("Import Error: could not load sentence_transformers... proceeding")
logger = logging.getLogger(__name__)


class BEIRSpladeModel:
    def __init__(self, model, tokenizer, max_length=256):
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.model = model

    # Write your own encoding query function (Returns: Query embeddings as numpy array)
    def encode_queries(self, queries: List[str], batch_size: int, **kwargs) -> np.ndarray:
        X = self.model.encode_sentence_bert(self.tokenizer, queries, is_q=True, maxlen=self.max_length)
        return X

    # Write your own encoding corpus function (Returns: Document embeddings as numpy array)  
    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs) -> np.ndarray:
        sentences = [(doc["title"] + ' ' + doc["text"]).strip() for doc in corpus]
        return self.model.encode_sentence_bert(self.tokenizer, sentences, maxlen=512)


class Splade(torch.nn.Module):

    def __init__(self, model_type_or_dir):
        super().__init__()
        self.transformer = AutoModelForMaskedLM.from_pretrained(model_type_or_dir)

    def forward(self, **kwargs):
        out = self.transformer(**kwargs)["logits"]  # output (logits) of MLM head, shape (bs, pad_len, voc_size)
        return torch.max(torch.log(1 + torch.relu(out)) * kwargs["attention_mask"].unsqueeze(-1), dim=1).values

    def _text_length(self, text: Union[List[int], List[List[int]]]):
        """helper function to get the length for the input text. Text can be either
        a list of ints (which means a single text as input), or a tuple of list of ints
        (representing several text inputs to the model).
        """

        if isinstance(text, dict):  # {key: value} case
            return len(next(iter(text.values())))
        elif not hasattr(text, '__len__'):  # Object has no len() method
            return 1
        elif len(text) == 0 or isinstance(text[0], int):  # Empty string or list of ints
            return len(text)
        else:
            return sum([len(t) for t in text])  # Sum of length of individual strings

    def encode_sentence_bert(self, tokenizer, sentences: Union[str, List[str], List[int]],
                             batch_size: int = 32,
                             show_progress_bar: bool = None,
                             output_value: str = 'sentence_embedding',
                             convert_to_numpy: bool = True,
                             convert_to_tensor: bool = False,
                             device: str = None,
                             normalize_embeddings: bool = False,
                             maxlen: int = 512,
                             is_q: bool = False) -> Union[List[Tensor], ndarray, Tensor]:
        """
        Computes sentence embeddings
        :param sentences: the sentences to embed
        :param batch_size: the batch size used for the computation
        :param show_progress_bar: Output a progress bar when encode sentences
        :param output_value:  Default sentence_embedding, to get sentence embeddings. Can be set to token_embeddings to get wordpiece token embeddings.
        :param convert_to_numpy: If true, the output is a list of numpy vectors. Else, it is a list of pytorch tensors.
        :param convert_to_tensor: If true, you get one large tensor as return. Overwrites any setting from convert_to_numpy
        :param device: Which torch.device to use for the computation
        :param normalize_embeddings: If set to true, returned vectors will have length 1. In that case, the faster dot-product (util.dot_score) instead of cosine similarity can be used.
        :return:
           By default, a list of tensors is returned. If convert_to_tensor, a stacked tensor is returned. If convert_to_numpy, a numpy matrix is returned.
        """
        self.eval()
        if show_progress_bar is None:
            show_progress_bar = True

        if convert_to_tensor:
            convert_to_numpy = False

        if output_value == 'token_embeddings':
            convert_to_tensor = False
            convert_to_numpy = False

        input_was_string = False
        if isinstance(sentences, str) or not hasattr(sentences, '__len__'):
            # Cast an individual sentence to a list with length 1
            sentences = [sentences]
            input_was_string = True

        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.to(device)

        all_embeddings = []
        length_sorted_idx = np.argsort([-self._text_length(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

        for start_index in trange(0, len(sentences), batch_size, desc="Batches", disable=not show_progress_bar):
            sentences_batch = sentences_sorted[start_index:start_index + batch_size]
            # features = tokenizer(sentences_batch)
            # print(sentences_batch)
            features = tokenizer(sentences_batch,
                                 add_special_tokens=True,
                                 padding="longest",  # pad to max sequence length in batch
                                 truncation="only_first",  # truncates to self.max_length
                                 max_length=maxlen,
                                 return_attention_mask=True,
                                 return_tensors="pt")
            # print(features)
            features = batch_to_device(features, device)

            with torch.no_grad():
                out_features = self.forward(**features)
                if output_value == 'token_embeddings':
                    embeddings = []
                    for token_emb, attention in zip(out_features[output_value], out_features['attention_mask']):
                        last_mask_id = len(attention) - 1
                        while last_mask_id > 0 and attention[last_mask_id].item() == 0:
                            last_mask_id -= 1
                        embeddings.append(token_emb[0:last_mask_id + 1])
                else:  # Sentence embeddings
                    # embeddings = out_features[output_value]
                    embeddings = out_features
                    embeddings = embeddings.detach()
                    if normalize_embeddings:
                        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                    # fixes for #522 and #487 to avoid oom problems on gpu with large datasets
                    if convert_to_numpy:
                        embeddings = embeddings.cpu()
                all_embeddings.extend(embeddings)
        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]
        if convert_to_tensor:
            all_embeddings = torch.stack(all_embeddings)
        elif convert_to_numpy:
            all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])
        if input_was_string:
            all_embeddings = all_embeddings[0]
        return all_embeddings

    
class BEIRColBERT:
    def __init__(self, model_path=None, qlen=32, dlen=256, **kwargs):
        self.qlen = qlen
        self.dlen = dlen
        self.model = ColBERTTransformer(model_path, max_seq_length=self.dlen)
        checkpoint = torch.load(os.path.join(model_path, "checkpoint.pt"), map_location='cpu')
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokens = ["[unused0]", "[unused1]", "[unused2]"] 
        self.tokenizer.add_tokens(tokens, special_tokens=True)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model.to(self.device)
        
    
    # Write your own score function, which takes in query-document text pairs and returns the similarity scores
    def predict(self, sentences: List[Tuple[str,str]], batch_size: int, **kwags) -> List[float]:
        query_dict = defaultdict(list)
        for idx, item in enumerate(sentences):
            query_dict[item[0]].append([item[1], idx])
        result_scores = []
        with torch.no_grad():
            for q in tqdm(query_dict):
                q_features = self.tokenizer("[unused0] " + q, return_tensors="pt", max_length = self.qlen, truncation = True).to(self.device)
                q_features = self.model(q_features)
                token_rep_q = torch.nn.functional.normalize(q_features['last_layer_embeddings'], p=2, dim=2)

                d_features = self.tokenizer(["[unused0] " + d[0] for d in query_dict[q]], return_tensors="pt", max_length=self.dlen,truncation=True,padding=True)
                d_features = _split_into_batches(d_features,bsize=bsize)

                all_scores = []
                for batch in d_features:
                    d_batch = self.model(batch)
                    d_mask = d_batch['attention_mask'].to(self.device)
                    d_emb = d_batch['last_layer_embeddings']
                    del d_batch
                    d_mask = d_mask.unsqueeze(-1)
                    token_rep_d =  d_emb * d_mask
                    del d_mask, d_emb
                    token_rep_d = torch.nn.functional.normalize(token_rep_d)
                    scores =  (token_rep_q @ token_rep_d.permute(0,2,1)).max(2).values.sum(1).tolist()
                    del token_rep_d
                    torch.cuda.empty_cache()
                    all_scores.extend(scores)
                    
                for d, score in zip(query_dict[q], all_scores):
                    result_scores.append([d[1], score])
        
        result_scores = sorted(result_scores, key = lambda x: x[0])
        
        return [x[1] for x in result_scores]


class BEIRTeacher:
    def __init__(self, model_path=None, qlen=32, dlen=256, **kwargs):
        self.qlen = qlen
        self.dlen = dlen
        self.model = CrossEncoder(model_path, max_length=512) 

    
    # Write your own score function, which takes in query-document text pairs and returns the similarity scores
    def predict(self, sentences: List[Tuple[str,str]], batch_size: int, **kwags) -> List[float]:
        return self.model.predict(sentences, batch_size = batch_size)


class BEIRColBERTCQ:
    def __init__(self, model_path=None, qlen=32, dlen=256, cq_model_dir=None, non_contextual_emb_dict = None, **kwargs):
        self.qlen = qlen
        self.dlen = dlen
        self.model = ColBERTTransformer(model_path, max_seq_length=self.dlen, dim = 128)
        checkpoint = torch.load(os.path.join(model_path, "checkpoint.pt"), map_location='cpu')
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokens = ["[unused0]", "[unused1]", "[unused2]"] 
        self.tokenizer.add_tokens(tokens, special_tokens=True)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model.to(self.device)
        self.code_learner = Code_Learner(emb_size=128, M=16, K=256, hidden_size = 128)
        self.code_learner.load_state_dict(torch.load(cq_model_dir))
        self.code_learner.to(self.device)
        self.non_contextual_emb_dict = non_contextual_emb_dict
        
    
    # Write your own score function, which takes in query-document text pairs and returns the similarity scores
    def predict(self, sentences: List[Tuple[str,str]], batch_size: int, **kwags) -> List[float]:
        query_dict = defaultdict(list)
        for idx, item in enumerate(sentences):
            query_dict[item[0]].append([item[1], idx])
        result_scores = []
        with torch.no_grad():
            for q in tqdm(query_dict):
                q_features = self.tokenizer("[unused0] " + q, return_tensors="pt", max_length = self.qlen, truncation = True).to(self.device)
                q_features = self.model(q_features)
                token_rep_q = torch.nn.functional.normalize(q_features['last_layer_embeddings'], p=2, dim=2)

                d_features = self.tokenizer(["[unused1] " + d[0] for d in query_dict[q]], return_tensors="pt", max_length=self.dlen,truncation=True,padding=True)
                d_features = _split_into_batches(d_features,bsize=bsize)

                all_scores = []
                for batch in d_features:
                    d_batch = self.model(batch)
                    d_mask = d_batch['attention_mask'].to(self.device)
                    d_emb = d_batch['last_layer_embeddings']
                    d_len = d_emb.shape[1]
                    del d_batch
                    d_mask = d_mask
                    token_rep_d =  d_emb * d_mask.unsqueeze(-1)
                    del d_emb
                    non_context_embs_d = []
                
                    for d in batch['input_ids'].tolist():
                        non_context_embs_d.append([self.non_contextual_emb_dict[t] for t in d] + [self.non_contextual_emb_dict[self.tokenizer.pad_token_id]] * (d_len - len(d)))
                    cq_input = torch.cat([token_rep_d, torch.tensor(non_context_embs_d).to("cuda")], dim = -1).reshape(-1, token_rep_d.shape[-1]*2)
                    token_d = self.code_learner(cq_input, training = False)
                    token_d = token_d.reshape(token_rep_d.shape[0], token_rep_d.shape[1], -1)
                
                    token_d = torch.nn.functional.normalize(token_d, p=2, dim=2)

                    token_level_score = token_rep_q @ token_d.permute(0,2,1)
                    iter_mask = ~d_mask.unsqueeze(1).repeat(1, token_level_score.shape[1], 1).bool()
                    token_level_score[iter_mask] = -9999

                    scores =  token_level_score.max(2).values.sum(1).tolist()
                    del token_rep_d
                    torch.cuda.empty_cache()
                    all_scores.extend(scores)
                    
                for d, score in zip(query_dict[q], all_scores):
                    result_scores.append([d[1], score])
        
        result_scores = sorted(result_scores, key = lambda x: x[0])
        
        return [x[1] for x in result_scores]



