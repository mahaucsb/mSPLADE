#FROM Sentence-BERT(https://github.com/UKPLab/sentence-transformers/blob/afee883a17ab039120783fd0cffe09ea979233cf/examples/training/ms_marco/train_bi-encoder_margin-mse.py) with minimal changes.
#Original License APACHE2

from torch import nn
from transformers import AutoModel, BertModel, AutoModelForMaskedLM, AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
import json
from typing import List, Dict, Optional, Union, Tuple
import os
import torch
from torch import Tensor
from transformers.models.bert.modeling_bert import BertPredictionHeadTransform
import warnings
import tqdm

def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
    # type: (Tensor, float, bool, float, int) -> Tensor
    r"""
    Samples from the Gumbel-Softmax distribution (`Link 1`_  `Link 2`_) and optionally discretizes.

    Args:
      logits: `[..., num_features]` unnormalized log probabilities
      tau: non-negative scalar temperature
      hard: if ``True``, the returned samples will be discretized as one-hot vectors,
            but will be differentiated as if it is the soft sample in autograd
      dim (int): A dimension along which softmax will be computed. Default: -1.

    Returns:
      Sampled tensor of same shape as `logits` from the Gumbel-Softmax distribution.
      If ``hard=True``, the returned samples will be one-hot, otherwise they will
      be probability distributions that sum to 1 across `dim`.

    .. note::
      This function is here for legacy reasons, may be removed from nn.Functional in the future.

    .. note::
      The main trick for `hard` is to do  `y_hard - y_soft.detach() + y_soft`

      It achieves two things:
      - makes the output value exactly one-hot
      (since we add then subtract y_soft value)
      - makes the gradient equal to y_soft gradient
      (since we strip all other gradients)

    Examples::
        >>> logits = torch.randn(20, 32)
        >>> # Sample soft categorical using reparametrization trick:
        >>> F.gumbel_softmax(logits, tau=1, hard=False)
        >>> # Sample hard categorical using "Straight-through" trick:
        >>> F.gumbel_softmax(logits, tau=1, hard=True)

    .. _Link 1:
        https://arxiv.org/abs/1611.00712
    .. _Link 2:
        https://arxiv.org/abs/1611.01144
    """
    if eps != 1e-10:
        warnings.warn("`eps` parameter is deprecated and has no effect.")

    gumbels = -torch.empty_like(logits).exponential_().log()  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret

class Splade_Pooling(nn.Module):
    def __init__(self, word_embedding_dimension: int, pool_method = "max"):
        super(Splade_Pooling, self).__init__()
        self.word_embedding_dimension = word_embedding_dimension
        self.config_keys = ["word_embedding_dimension"]
        self.pooling_method = pool_method
        if self.pooling_method == "max_and_mean_smart":
            self.linear_comb = nn.Linear(2, 1, False)

    def __repr__(self):
        return "Pooling Splade({})"

    def get_pooling_mode_str(self) -> str:
        return "Splade"

    def forward(self, features: Dict[str, Tensor]):
        token_embeddings = features['token_embeddings']
        attention_mask = features['attention_mask']

        ## Pooling strategy
        reps = torch.log(1 + torch.relu(token_embeddings)) * attention_mask.unsqueeze(-1)
        if self.pooling_method == "max_and_sum":
            max_embedding = torch.max(reps, dim=1).values
            sum_embedding = torch.sum(reps, dim=1)
            features.update({'max_embedding': max_embedding, 'sum_embedding': sum_embedding})
            return features
        elif self.pooling_method == "max_and_mean_smart":
            max_embedding = torch.max(reps, dim=1).values
            sum_embedding = torch.sum(reps, dim=1)  
            mean_embedding =  sum_embedding / (torch.sum(attention_mask, dim = 1).unsqueeze(-1)) 
            features.update({'max_embedding': max_embedding, 'mean_embedding': mean_embedding})
            return features
        elif self.pooling_method == "max":
            sentence_embedding = torch.max(reps, dim=1).values
        elif self.pooling_method == "sum":
            sentence_embedding = torch.sum(reps, dim=1)
        elif self.pooling_method == "none":
            sentence_embedding = reps
        elif self.pooling_method == "mean":
            sum_embedding = torch.sum(reps, dim=1)
            sentence_embedding =  sum_embedding / (torch.sum(attention_mask, dim = 1).unsqueeze(-1)) 
        features.update({'sentence_embedding': sentence_embedding})
        return features

    def get_sentence_embedding_dimension(self):
        return self.word_embedding_dimension

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path):
        with open(os.path.join(output_path, 'config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

    @staticmethod
    def load(input_path):
        with open(os.path.join(input_path, 'config.json')) as fIn:
            config = json.load(fIn)

        return Splade_Pooling(**config)


class MLMTransformer(nn.Module):
    """Huggingface AutoModel to generate token embeddings.
    Loads the correct class, e.g. BERT / RoBERTa etc.

    :param model_name_or_path: Huggingface models name (https://huggingface.co/models)
    :param max_seq_length: Truncate any inputs longer than max_seq_length
    :param model_args: Arguments (key, value pairs) passed to the Huggingface Transformers model
    :param cache_dir: Cache dir for Huggingface Transformers to store/load models
    :param tokenizer_args: Arguments (key, value pairs) passed to the Huggingface Tokenizer model
    :param do_lower_case: If true, lowercases the input (independent if the model is cased or not)
    :param tokenizer_name_or_path: Name or path of the tokenizer. When None, then model_name_or_path is used
    """
    def __init__(self, model_name_or_path: str, max_seq_length: Optional[int] = 256,
                 model_args: Dict = {}, cache_dir: Optional[str] = None,
                 tokenizer_args: Dict = {}, do_lower_case: bool = False,
                 tokenizer_name_or_path : str = None, pooling_method: str = "max"):
        super(MLMTransformer, self).__init__()
        self.config_keys = ['max_seq_length', 'do_lower_case']
        self.do_lower_case = do_lower_case

        self.config = AutoConfig.from_pretrained(model_name_or_path, **model_args, cache_dir=cache_dir)
        self.auto_model = torch.nn.DataParallel(AutoModelForMaskedLM.from_pretrained(model_name_or_path, config=self.config, cache_dir=cache_dir))
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path if tokenizer_name_or_path is not None else model_name_or_path, cache_dir=cache_dir, **tokenizer_args)
        self.pooling = torch.nn.DataParallel(Splade_Pooling(self.get_word_embedding_dimension(), pool_method = pooling_method)) 
        
        # No max_seq_length set. Try to infer from model
        if max_seq_length is None:
            if hasattr(self.auto_model, "config") and hasattr(self.auto_model.config, "max_position_embeddings") and hasattr(self.tokenizer, "model_max_length"):
                max_seq_length = min(self.auto_model.config.max_position_embeddings, self.tokenizer.model_max_length)

        self.max_seq_length = max_seq_length

        if tokenizer_name_or_path is not None:
            self.auto_model.config.tokenizer_class = self.tokenizer.__class__.__name__

    def __repr__(self):
        return "MLMTransformer({}) with Transformer model: {} ".format(self.get_config_dict(), self.auto_model.__class__.__name__)

    def forward(self, features):
        """Returns token_embeddings, cls_token"""
        trans_features = {'input_ids': features['input_ids'], 'attention_mask': features['attention_mask']}
        if 'token_type_ids' in features:
            trans_features['token_type_ids'] = features['token_type_ids']

        output_states = self.auto_model(**trans_features, return_dict=False)
        output_tokens = output_states[0]

        features.update({'token_embeddings': output_tokens, 'attention_mask': features['attention_mask']})

        if self.auto_model.module.config.output_hidden_states:
            all_layer_idx = 2
            if len(output_states) < 3: #Some models only output last_hidden_states and all_hidden_states
                all_layer_idx = 1

            hidden_states = output_states[all_layer_idx]
            features.update({'all_layer_embeddings': hidden_states})

        features = self.pooling(features)

        return features

    def get_word_embedding_dimension(self) -> int:
            return self.auto_model.module.config.vocab_size
        
    def tokenize(self, texts: Union[List[str], List[Dict], List[Tuple[str, str]]]):
        """
        Tokenizes a text and maps tokens to token-ids
        """
        output = {}
        if isinstance(texts[0], str):
            to_tokenize = [texts]
        elif isinstance(texts[0], dict):
            to_tokenize = []
            output['text_keys'] = []
            for lookup in texts:
                text_key, text = next(iter(lookup.items()))
                to_tokenize.append(text)
                output['text_keys'].append(text_key)
            to_tokenize = [to_tokenize]
        else:
            batch1, batch2 = [], []
            for text_tuple in texts:
                batch1.append(text_tuple[0])
                batch2.append(text_tuple[1])
            to_tokenize = [batch1, batch2]

        #strip
        to_tokenize = [[str(s).strip() for s in col] for col in to_tokenize]

        #Lowercase
        if self.do_lower_case:
            to_tokenize = [[s.lower() for s in col] for col in to_tokenize]

        output.update(self.tokenizer(*to_tokenize, padding=True, truncation='longest_first', return_tensors="pt", max_length=self.max_seq_length))
        return output

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path: str):
        self.auto_model.module.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        with open(os.path.join(output_path, 'sentence_bert_config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

    @staticmethod
    def load(input_path: str):
        #Old classes used other config names than 'sentence_bert_config.json'
        for config_name in ['sentence_bert_config.json', 'sentence_roberta_config.json', 'sentence_distilbert_config.json', 'sentence_camembert_config.json', 'sentence_albert_config.json', 'sentence_xlm-roberta_config.json', 'sentence_xlnet_config.json']:
            sbert_config_path = os.path.join(input_path, config_name)
            if os.path.exists(sbert_config_path):
                break

        with open(sbert_config_path) as fIn:
            config = json.load(fIn)
        return MLMTransformer(model_name_or_path=input_path, **config)

class MVMLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size * 3, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size * 3))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states

class MVMLMTransformer(nn.Module):
    """Huggingface AutoModel to generate token embeddings.
    Loads the correct class, e.g. BERT / RoBERTa etc.

    :param model_name_or_path: Huggingface models name (https://huggingface.co/models)
    :param max_seq_length: Truncate any inputs longer than max_seq_length
    :param model_args: Arguments (key, value pairs) passed to the Huggingface Transformers model
    :param cache_dir: Cache dir for Huggingface Transformers to store/load models
    :param tokenizer_args: Arguments (key, value pairs) passed to the Huggingface Tokenizer model
    :param do_lower_case: If true, lowercases the input (independent if the model is cased or not)
    :param tokenizer_name_or_path: Name or path of the tokenizer. When None, then model_name_or_path is used
    """
    def __init__(self, model_name_or_path: str, max_seq_length: Optional[int] = 256,
                 model_args: Dict = {}, cache_dir: Optional[str] = None,
                 tokenizer_args: Dict = {}, do_lower_case: bool = False,
                 tokenizer_name_or_path : str = None, token_sparse: bool = False, gumbel: bool=False):
        super(MVMLMTransformer, self).__init__()
        self.config_keys = ['max_seq_length', 'do_lower_case', 'token_sparse']
        self.do_lower_case = do_lower_case
        self.token_sparse = token_sparse
        self.config = AutoConfig.from_pretrained(model_name_or_path, **model_args, cache_dir=cache_dir)
        self.bert = torch.nn.DataParallel(BertModel.from_pretrained(model_name_or_path, config=self.config, cache_dir=cache_dir))
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path if tokenizer_name_or_path is not None else model_name_or_path, cache_dir=cache_dir, **tokenizer_args)
        self.pooling = torch.nn.DataParallel(Splade_Pooling(self.get_word_embedding_dimension())) 
        self.cls = torch.nn.DataParallel(MVMLMPredictionHead(self.config))
        self.gumbel = gumbel
        
        # No max_seq_length set. Try to infer from model
        if max_seq_length is None:
            if hasattr(self.bert, "config") and hasattr(self.bert.config, "max_position_embeddings") and hasattr(self.tokenizer, "model_max_length"):
                max_seq_length = min(self.bert.config.max_position_embeddings, self.tokenizer.model_max_length)

        self.max_seq_length = max_seq_length

        if tokenizer_name_or_path is not None:
            self.bert.config.tokenizer_class = self.tokenizer.__class__.__name__

    def __repr__(self):
        return "MLMTransformer({}) with Transformer model: {} ".format(self.get_config_dict(), self.bert.__class__.__name__)

    def forward(self, features):
        """Returns token_embeddings, cls_token"""
        trans_features = {'input_ids': features['input_ids'], 'attention_mask': features['attention_mask']}
        if 'token_type_ids' in features:
            trans_features['token_type_ids'] = features['token_type_ids']

        output_states = self.bert(**trans_features)
        output_tokens = output_states[0]

        prediction_scores = self.cls(output_tokens)
        if self.token_sparse:
            prediction_scores = prediction_scores.view(output_tokens.shape[0], output_tokens.shape[1], 3, -1)

            if self.gumbel:
                token_select = gumbel_softmax(prediction_scores, tau=1, hard=True, dim = 2)
                prediction_p = prediction_scores.softmax(dim = 2).log() * token_select
                features.update({'token_select_logp': prediction_p})
            else:
                token_select = prediction_scores.max(dim=2,keepdim = True)[1]
                token_select = torch.zeros_like(prediction_scores).scatter_(2, token_select, 1.0)
                prediction_p = prediction_scores.softmax(dim = 2).log() * token_select
                features.update({'token_select_logp': prediction_p})

            prediction_scores = prediction_scores * token_select
            prediction_scores = prediction_scores.view(output_tokens.shape[0], output_tokens.shape[1], -1)

        features.update({'token_embeddings': prediction_scores, 'attention_mask': features['attention_mask']})

        features = self.pooling(features)

        return features

    def get_word_embedding_dimension(self) -> int:
            return self.auto_model.module.config.vocab_size
        
    def tokenize(self, texts: Union[List[str], List[Dict], List[Tuple[str, str]]]):
        """
        Tokenizes a text and maps tokens to token-ids
        """
        output = {}
        if isinstance(texts[0], str):
            to_tokenize = [texts]
        elif isinstance(texts[0], dict):
            to_tokenize = []
            output['text_keys'] = []
            for lookup in texts:
                text_key, text = next(iter(lookup.items()))
                to_tokenize.append(text)
                output['text_keys'].append(text_key)
            to_tokenize = [to_tokenize]
        else:
            batch1, batch2 = [], []
            for text_tuple in texts:
                batch1.append(text_tuple[0])
                batch2.append(text_tuple[1])
            to_tokenize = [batch1, batch2]

        #strip
        to_tokenize = [[str(s).strip() for s in col] for col in to_tokenize]

        #Lowercase
        if self.do_lower_case:
            to_tokenize = [[s.lower() for s in col] for col in to_tokenize]

        output.update(self.tokenizer(*to_tokenize, padding=True, truncation='longest_first', return_tensors="pt", max_length=self.max_seq_length))
        return output

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path: str):
        self.bert.module.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        torch.save(self.state_dict(), os.path.join(output_path, "checkpoint.pt"))
        
        with open(os.path.join(output_path, 'sentence_bert_config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

    @staticmethod
    def load(input_path: str):
        #Old classes used other config names than 'sentence_bert_config.json'
        for config_name in ['sentence_bert_config.json', 'sentence_roberta_config.json', 'sentence_distilbert_config.json', 'sentence_camembert_config.json', 'sentence_albert_config.json', 'sentence_xlm-roberta_config.json', 'sentence_xlnet_config.json']:
            sbert_config_path = os.path.join(input_path, config_name)
            if os.path.exists(sbert_config_path):
                break

        with open(sbert_config_path) as fIn:
            config = json.load(fIn)
        return MLMTransformer(model_name_or_path=input_path, **config)


def pad_mask(input_tensor, tokid, maxlen = 32):
    if input_tensor.shape[1] > maxlen:
        return input_tensor
    paddings = tokid * torch.ones([input_tensor.shape[0], maxlen - input_tensor.shape[1]]).to(input_tensor.device)
    return torch.cat([input_tensor, paddings], dim=1).to(input_tensor.dtype)

class ColBERTTransformer(MLMTransformer):
    def __init__(self, model_name_or_path: str, max_seq_length: Optional[int] = None,
                model_args: Dict = {}, cache_dir: Optional[str] = None,
                tokenizer_args: Dict = {}, do_lower_case: bool = False,
                tokenizer_name_or_path : str = None, dim = 768):
        super(ColBERTTransformer, self).__init__(model_name_or_path, max_seq_length,
                model_args, cache_dir,tokenizer_args, do_lower_case,tokenizer_name_or_path)
        self.linear = nn.Linear(self.config.hidden_size, dim, bias=True)
        self.acti = nn.GELU()
        self.output = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim, eps=self.config.layer_norm_eps)
    
    def forward(self, features, padding = True):
        """Returns token_embeddings, cls_token"""
        self.auto_model.module.config.output_hidden_states = True
        
        trans_features = {'input_ids': features['input_ids'], 'attention_mask': features['attention_mask']}
        
        if padding: 
            trans_features['input_ids'] = pad_mask(trans_features['input_ids'], self.tokenizer.mask_token_id)
            trans_features['attention_mask'] = pad_mask(trans_features['attention_mask'], 0)

        if 'token_type_ids' in features:
            trans_features['token_type_ids'] = features['token_type_ids']
            if padding:
                trans_features['token_type_ids'] =  pad_mask(trans_features['token_type_ids'], 0)
        
        output_states = self.auto_model(**trans_features, return_dict=False)
        
        features.update({'attention_mask': trans_features['attention_mask']})

        
        all_layer_idx = 2
        if len(output_states) < 3: #Some models only output last_hidden_states and all_hidden_states
            all_layer_idx = 1
            
        hidden_states = output_states[all_layer_idx]
        features.update({'all_layer_embeddings': hidden_states})

        features['cls'] = self.output(self.norm(self.acti(self.linear(features['all_layer_embeddings'][-1][:,0,:]))))
        features['last_layer_embeddings'] = self.linear(features['all_layer_embeddings'][-1])

        return features

    def save(self, output_path: str):
        self.auto_model.module.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        
        torch.save(self.state_dict(), os.path.join(output_path, "checkpoint.pt"))

        with open(os.path.join(output_path, 'sentence_bert_config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

    def save_nocontext_embedding(self):
        with torch.no_grad():
            for tok in tqdm.tqdm(range(self.auto_model.module.config.vocab_size)):
                output_states = self.auto_model(torch.tensor([[101, tok, 102]]).to(self.auto_model.device), torch.tensor([[1,1,1]]).to(self.auto_model.device), return_dict=False)

                all_layer_idx = 2
                if len(output_states) < 3: #Some models only output last_hidden_states and all_hidden_states
                    all_layer_idx = 1
                    
                hidden_states = output_states[all_layer_idx]
                tok_emb = self.linear(hidden_states[-1])

                self.non_contextual_embedding.weight[tok] = tok_emb[:,1,:]
          

class CETransformer(MLMTransformer):
    def __init__(self, model_name_or_path: str, max_seq_length: Optional[int] = None,
                model_args: Dict = {}, cache_dir: Optional[str] = None,
                tokenizer_args: Dict = {}, do_lower_case: bool = False,
                tokenizer_name_or_path : str = None, dim = 768):
        super(CETransformer, self).__init__(model_name_or_path, max_seq_length,
                model_args, cache_dir,tokenizer_args, do_lower_case,tokenizer_name_or_path)
        self.linear = nn.Linear(self.config.hidden_size, dim,bias=True)
        self.acti = nn.GELU()
        self.output = nn.Linear(dim, 1)
        self.norm = nn.LayerNorm(self.config.hidden_size, eps=self.config.layer_norm_eps)
    
    def forward(self, features):
        """Returns token_embeddings, cls_token"""
        self.auto_model.module.config.output_hidden_states = True
        
        trans_features = {'input_ids': features['input_ids'], 'attention_mask': features['attention_mask']}
        
       
        if 'token_type_ids' in features:
            trans_features['token_type_ids'] = features['token_type_ids']
            
        output_states = self.auto_model(**trans_features, return_dict=False)
        
        features.update({'attention_mask': trans_features['attention_mask']})

        
        all_layer_idx = 2
        if len(output_states) < 3: #Some models only output last_hidden_states and all_hidden_states
            all_layer_idx = 1
            
        hidden_states = output_states[all_layer_idx]
        features.update({'all_layer_embeddings': hidden_states})

        features['cls'] = self.output(self.norm(self.acti(self.linear(features['all_layer_embeddings'][-1][:,0,:]))))
        return features

    def save(self, output_path: str):
        self.auto_model.module.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        
        torch.save(self.state_dict(), os.path.join(output_path, "checkpoint.pt"))

        with open(os.path.join(output_path, 'sentence_bert_config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)


class CETransformerSeq(nn.Module):
    def __init__(self, model_name_or_path: str, max_seq_length: Optional[int] = None,
                model_args: Dict = {}, cache_dir: Optional[str] = None,
                tokenizer_args: Dict = {}, do_lower_case: bool = False,
                tokenizer_name_or_path : str = None, dim = 768):
        super(CETransformerSeq, self).__init__()
        self.config_keys = ['max_seq_length', 'do_lower_case']
        self.do_lower_case = do_lower_case

        self.config = AutoConfig.from_pretrained(model_name_or_path, **model_args, cache_dir=cache_dir)
        self.config.num_labels = 1
        self.auto_model = torch.nn.DataParallel(AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=self.config, cache_dir=cache_dir))
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path if tokenizer_name_or_path is not None else model_name_or_path, cache_dir=cache_dir, **tokenizer_args)
        
        # No max_seq_length set. Try to infer from model
        if max_seq_length is None:
            if hasattr(self.auto_model, "config") and hasattr(self.auto_model.config, "max_position_embeddings") and hasattr(self.tokenizer, "model_max_length"):
                max_seq_length = min(self.auto_model.config.max_position_embeddings, self.tokenizer.model_max_length)

        self.max_seq_length = max_seq_length

        if tokenizer_name_or_path is not None:
            self.auto_model.config.tokenizer_class = self.tokenizer.__class__.__name__

    def __repr__(self):
        return "CETransformer({}) with Transformer model: {} ".format(self.get_config_dict(), self.auto_model.__class__.__name__)
    
    def forward(self, features):
        """Returns token_embeddings, cls_token"""
        self.auto_model.module.config.output_hidden_states = True
        
        trans_features = {'input_ids': features['input_ids'], 'attention_mask': features['attention_mask']}
        
       
        if 'token_type_ids' in features:
            trans_features['token_type_ids'] = features['token_type_ids']
            
        output_states = self.auto_model(**trans_features, return_dict=True)
        features.update({'attention_mask': trans_features['attention_mask']})

        
        all_layer_idx = 2
        if len(output_states) < 3: #Some models only output last_hidden_states and all_hidden_states
            all_layer_idx = 1
        #print(output_states[all_layer_idx][-1][:,0,:].shape)
        features['cls'] = output_states.logits
        return features

    def save(self, output_path: str):
        self.auto_model.module.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        
        torch.save(self.state_dict(), os.path.join(output_path, "checkpoint.pt"))

        with open(os.path.join(output_path, 'sentence_bert_config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)


    def get_word_embedding_dimension(self) -> int:
            return self.auto_model.module.config.vocab_size
        
    def tokenize(self, texts: Union[List[str], List[Dict], List[Tuple[str, str]]]):
        """
        Tokenizes a text and maps tokens to token-ids
        """
        output = {}
        if isinstance(texts[0], str):
            to_tokenize = [texts]
        elif isinstance(texts[0], dict):
            to_tokenize = []
            output['text_keys'] = []
            for lookup in texts:
                text_key, text = next(iter(lookup.items()))
                to_tokenize.append(text)
                output['text_keys'].append(text_key)
            to_tokenize = [to_tokenize]
        else:
            batch1, batch2 = [], []
            for text_tuple in texts:
                batch1.append(text_tuple[0])
                batch2.append(text_tuple[1])
            to_tokenize = [batch1, batch2]

        #strip
        to_tokenize = [[str(s).strip() for s in col] for col in to_tokenize]

        #Lowercase
        if self.do_lower_case:
            to_tokenize = [[s.lower() for s in col] for col in to_tokenize]

        output.update(self.tokenizer(*to_tokenize, padding=True, truncation='longest_first', return_tensors="pt", max_length=self.max_seq_length))
        return output

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}


    @staticmethod
    def load(input_path: str):
        #Old classes used other config names than 'sentence_bert_config.json'
        for config_name in ['sentence_bert_config.json', 'sentence_roberta_config.json', 'sentence_distilbert_config.json', 'sentence_camembert_config.json', 'sentence_albert_config.json', 'sentence_xlm-roberta_config.json', 'sentence_xlnet_config.json']:
            sbert_config_path = os.path.join(input_path, config_name)
            if os.path.exists(sbert_config_path):
                break

        with open(sbert_config_path) as fIn:
            config = json.load(fIn)
        return CETransformerSeq(model_name_or_path=input_path, **config)

 
class ColBERTTransformerCQ(MLMTransformer):
    def __init__(self, model_name_or_path: str, max_seq_length: Optional[int] = None,
                model_args: Dict = {}, cache_dir: Optional[str] = None,
                tokenizer_args: Dict = {}, do_lower_case: bool = False,
                tokenizer_name_or_path : str = None, dim = 768):
        super(ColBERTTransformerCQ, self).__init__(model_name_or_path, max_seq_length,
                model_args, cache_dir,tokenizer_args, do_lower_case,tokenizer_name_or_path)
        self.linear = nn.Linear(self.config.hidden_size, dim, bias=True)
        self.acti = nn.GELU()
        self.output = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim, eps=self.config.layer_norm_eps)
    
    def forward(self, features, padding = True):
        """Returns token_embeddings, cls_token"""
        self.auto_model.module.config.output_hidden_states = True
        
        trans_features = {'input_ids': features['input_ids'], 'attention_mask': features['attention_mask']}
        
        if padding: 
            trans_features['input_ids'] = pad_mask(trans_features['input_ids'], self.tokenizer.mask_token_id)
            trans_features['attention_mask'] = pad_mask(trans_features['attention_mask'], 0)

        if 'token_type_ids' in features:
            trans_features['token_type_ids'] = features['token_type_ids']
            if padding:
                trans_features['token_type_ids'] =  pad_mask(trans_features['token_type_ids'], 0)
        
        output_states = self.auto_model(**trans_features, return_dict=False)
        
        features.update({'attention_mask': trans_features['attention_mask']})

        
        all_layer_idx = 2
        if len(output_states) < 3: #Some models only output last_hidden_states and all_hidden_states
            all_layer_idx = 1
            
        hidden_states = output_states[all_layer_idx]
        features.update({'all_layer_embeddings': hidden_states})

        features['cls'] = self.output(self.norm(self.acti(self.linear(features['all_layer_embeddings'][-1][:,0,:]))))
        features['last_layer_embeddings'] = self.linear(features['all_layer_embeddings'][-1])
        token_rep_d =  features['last_layer_embeddings'] * features['attention_mask'].unsqueeze(-1)
        non_context_embs_d = []
        d_len = features['last_layer_embeddings'].shape[1]    
        for d in features['input_ids'].tolist():
            non_context_embs_d.append([self.non_contextual_emb_dict[t] for t in d] + [self.non_contextual_emb_dict[self.tokenizer.pad_token_id]] * (d_len - len(d)))

        #print("non_context_embs_d", non_context_embs_d) 
        #print("token_rep_d", token_rep_d.shape) #16,32,128

        cq_input = torch.cat([token_rep_d, torch.tensor(non_context_embs_d).to("cuda")], dim = -1)
        features.update({'cq_embedding': cq_input})
        return features

    def save(self, output_path: str):
        self.auto_model.module.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        
        torch.save(self.state_dict(), os.path.join(output_path, "checkpoint.pt"))

        with open(os.path.join(output_path, 'sentence_bert_config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

    def save_nocontext_embedding(self):
        self.non_contextual_emb_dict = dict()
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        with torch.no_grad():
            for tok in tqdm.tqdm(range(self.auto_model.module.config.vocab_size)):
                output_states = self.auto_model(input_ids = torch.tensor([[101, 2, tok, 102]]).to(device), attention_mask = torch.tensor([[1,1,1,1]]).to(device), return_dict=False)

                all_layer_idx = 2
                if len(output_states) < 3: #Some models only output last_hidden_states and all_hidden_states
                    all_layer_idx = 1
                    
                hidden_states = output_states[all_layer_idx]
                tok_emb = self.linear(hidden_states[-1])

                self.non_contextual_emb_dict[tok] = tok_emb[:,2,:].tolist()[0]
          
