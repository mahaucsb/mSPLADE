#FROM Sentence-BERT: (https://github.com/UKPLab/sentence-transformers/blob/afee883a17ab039120783fd0cffe09ea979233cf/sentence_transformers/losses/MultipleNegativesRankingLoss.py) (https://github.com/UKPLab/sentence-transformers/blob/afee883a17ab039120783fd0cffe09ea979233cf/sentence_transformers/losses/MarginMSELoss.py) (https://github.com/UKPLab/sentence-transformers/blob/afee883a17ab039120783fd0cffe09ea979233cf/sentence_transformers/util.py) with minimal changes.
#Original License APACHE2

from multiprocessing import reduction
from tkinter import E
from xml.etree.ElementPath import prepare_descendant
import torch
from torch import nn, Tensor
from typing import Iterable, Dict
import torch.nn.functional as F
import numpy as np

def pairwise_dot_score(a: Tensor, b: Tensor):
    """
   Computes the pairwise dot-product dot_prod(a[i], b[i])
   :return: Vector with res[i] = dot_prod(a[i], b[i])
   """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)
    
    return (a * b).sum(dim=-1)


def mse_scale(output, bench, target):
    """
   Computes the weighted mse loss
   """
    loss = nn.functional.mse_loss(output, target,reduce=None)
    print(loss)
    scale = -torch.sign(bench) * torch.sign(target) + 2
    scale = scale.detach()
    return torch.mean(loss * scale)
    

def dot_score(a: Tensor, b: Tensor):
    """
    Computes the dot-product dot_prod(a[i], b[j]) for all i and j
    :return: Matrix with res[i][j]  = dot_prod(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    return torch.mm(a, b.transpose(0, 1))

class FLOPS:
    """constraint from Minimizing FLOPs to Learn Efficient Sparse Representations
    https://arxiv.org/abs/2004.05665
    """

    def __call__(self, batch_rep):
        return torch.sum(torch.mean(torch.abs(batch_rep), dim=0) ** 2)

class L1:
    def __call__(self, batch_rep):
        return torch.sum(torch.abs(batch_rep), dim=-1).mean()

class UNIFORM: 
    def __call__(self, x, t=2):
        return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

class MarginMSELossSplade(nn.Module):
    """
    Compute the MSE loss between the |sim(Query, Pos) - sim(Query, Neg)| and |gold_sim(Q, Pos) - gold_sim(Query, Neg)|
    By default, sim() is the dot-product
    For more details, please refer to https://arxiv.org/abs/2010.02666
    """
    def __init__(self, model, similarity_fct = pairwise_dot_score, lambda_d=8e-2, lambda_q=1e-1, lambda_uni = 1e-2, uni_mse = False, logp_3v = 0.0):
        """
        :param model: SentenceTransformerModel
        :param similarity_fct:  Which similarity function to use
        """
        super(MarginMSELossSplade, self).__init__()
        self.model = model
        self.similarity_fct = similarity_fct
        self.loss_fct = nn.MSELoss()
        self.lambda_d = lambda_d
        self.lambda_q = lambda_q
        self.labmda_uni = lambda_uni
        self.FLOPS = FLOPS()
        self.L1=L1()
        self.uniform_mse = uni_mse
        self.uni = UNIFORM()
        self.logp_3v = logp_3v

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        # sentence_features: query, positive passage, negative passage
        preds = [self.model(sentence_feature) for sentence_feature in sentence_features]
        reps = [pred['sentence_embedding'] for pred in preds]
        embeddings_query = reps[0]
        embeddings_pos = reps[1]
        embeddings_neg = reps[2]

        scores_pos = self.similarity_fct(embeddings_query, embeddings_pos)
        scores_neg = self.similarity_fct(embeddings_query, embeddings_neg)
        margin_pred = scores_pos - scores_neg

        #flops_doc = self.lambda_d*(self.FLOPS(embeddings_pos) + self.FLOPS(embeddings_neg))
        flops_doc = self.lambda_d*self.FLOPS(torch.cat(reps, 0))
        #flops_query = self.lambda_q*self.FLOPS(embeddings_query)
        flops_query = self.lambda_q * self.L1(embeddings_query)
        
        if self.uniform_mse:
            uniform_dist = self.labmda_uni * (self.uni(embeddings_query) - self.uni(embeddings_neg)) ** 2
            return self.loss_fct(margin_pred, labels) + flops_doc + flops_query + uniform_dist

        ranking_loss = self.loss_fct(margin_pred, labels) 
        print("loss magnitude",ranking_loss, flops_doc, flops_query)
        print("loss argp_log", self.logp_3v)

        if self.logp_3v > 0:
            logp_loss = self.logp_3v * torch.cat([pred['token_select_logp'].sum(dim=2) for pred in preds], 1).mean()
            print("loss magnitude",ranking_loss, flops_doc, flops_query, logp_loss)
            return ranking_loss + flops_doc + flops_query + logp_loss
        return ranking_loss + flops_doc + flops_query


class MarginMSELossSpladeSumAndMax(nn.Module):
    """
    Compute the MSE loss between the |sim(Query, Pos) - sim(Query, Neg)| and |gold_sim(Q, Pos) - gold_sim(Query, Neg)|
    By default, sim() is the dot-product
    For more details, please refer to https://arxiv.org/abs/2010.02666
    """
    def __init__(self, model, similarity_fct = pairwise_dot_score, lambda_d=8e-2, lambda_q=1e-1):
        """
        :param model: SentenceTransformerModel
        :param similarity_fct:  Which similarity function to use
        """
        super(MarginMSELossSpladeSumAndMax, self).__init__()
        self.model = model
        self.loss_fct = nn.MSELoss()
        self.lambda_d = lambda_d
        self.lambda_q = lambda_q
        self.FLOPS = FLOPS()
        self.L1=L1()
        self.similarity_fct = similarity_fct

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        # sentence_features: query, positive passage, negative passage
        preds = [self.model(sentence_feature) for sentence_feature in sentence_features]
        max_reps = [pred['max_embedding'] for pred in preds]
        sum_reps = [pred['sum_embedding'] for pred in preds]
        
        embeddings_pos_max = max_reps[1]
        embeddings_neg_max = max_reps[2]

        embeddings_query_sum = sum_reps[0]

        summax_scores_pos = self.similarity_fct(embeddings_query_sum, embeddings_pos_max)
        summax_scores_neg = self.similarity_fct(embeddings_query_sum, embeddings_neg_max)
    
        #margin_pred = maxmax_scores_pos - maxmax_scores_neg + 0.02 * (maxsum_scores_pos - maxsum_scores_neg) + 0.2 * (summax_scores_pos - summax_scores_neg) + 0.004 * (sumsum_scores_pos - sumsum_scores_neg)
        margin_pred = summax_scores_pos - summax_scores_neg
        ranking_loss = self.loss_fct(margin_pred, labels) 
        
        return ranking_loss


class MarginMSELossSpladeSumAndMeanSmart(nn.Module):
    """
    Compute the MSE loss between the |sim(Query, Pos) - sim(Query, Neg)| and |gold_sim(Q, Pos) - gold_sim(Query, Neg)|
    By default, sim() is the dot-product
    For more details, please refer to https://arxiv.org/abs/2010.02666
    """
    def __init__(self, model, similarity_fct = pairwise_dot_score, lambda_d=8e-2, lambda_q=1e-1):
        """
        :param model: SentenceTransformerModel
        :param similarity_fct:  Which similarity function to use
        """
        super(MarginMSELossSpladeSumAndMeanSmart, self).__init__()
        self.model = model
        self.loss_fct = nn.MSELoss()
        self.lambda_d = lambda_d
        self.lambda_q = lambda_q
        self.FLOPS = FLOPS()
        self.L1=L1()
        self.similarity_fct = similarity_fct

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        # sentence_features: query, positive passage, negative passage
        preds = [self.model(sentence_feature) for sentence_feature in sentence_features]
        max_reps = [pred['max_embedding'] for pred in preds]
        mean_reps = [pred['mean_embedding'] for pred in preds]
        
        embeddings_query_max = max_reps[0]
        embeddings_pos_max = max_reps[1]
        embeddings_neg_max = max_reps[2]

        embeddings_query_mean = mean_reps[0]
        embeddings_pos_mean = mean_reps[1]
        embeddings_neg_mean = mean_reps[2]

        maxmax_scores_pos = self.similarity_fct(embeddings_query_max, embeddings_pos_max)
        maxmax_scores_neg = self.similarity_fct(embeddings_query_max, embeddings_neg_max)

        maxsum_scores_pos = self.similarity_fct(embeddings_query_max, embeddings_pos_mean)
        maxsum_scores_neg = self.similarity_fct(embeddings_query_max, embeddings_neg_mean)

        summax_scores_pos = self.similarity_fct(embeddings_query_mean, embeddings_pos_max)
        summax_scores_neg = self.similarity_fct(embeddings_query_mean, embeddings_neg_max)

        sumsum_scores_pos = self.similarity_fct(embeddings_query_mean, embeddings_pos_mean)
        sumsum_scores_neg  = self.similarity_fct(embeddings_query_mean, embeddings_neg_mean)


        #print("loss maxsum_scores_pos", maxsum_scores_pos) 1xB
        #margin_pred = maxmax_scores_pos - maxmax_scores_neg + 0.02 * (maxsum_scores_pos - maxsum_scores_neg) + 0.2 * (summax_scores_pos - summax_scores_neg) + 0.004 * (sumsum_scores_pos - sumsum_scores_neg)
        margin_pred = self.model[0].pooling.module.linear_comb(torch.stack([maxmax_scores_pos - maxmax_scores_neg, summax_scores_pos - summax_scores_neg], dim=1))
  
        ranking_loss = self.loss_fct(margin_pred, labels) 
        return ranking_loss



class MultipleNegativesRankingLossSplade(nn.Module):
    def __init__(self, model, scale: float = 1.0, similarity_fct = dot_score, lambda_d=0.0008, lambda_q=0.0006):
        super(MultipleNegativesRankingLossSplade, self).__init__()
        self.model = model
        self.scale = scale
        self.similarity_fct = similarity_fct
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.lambda_d = lambda_d
        self.lambda_q = lambda_q
        self.FLOPS = FLOPS()

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        embeddings_a = reps[0]
        embeddings_b = torch.cat(reps[1:])

        scores = self.similarity_fct(embeddings_a, embeddings_b) * self.scale
        labels = torch.tensor(range(len(scores)), dtype=torch.long, device=scores.device)  # Example a[i] should match with b[i]

        flops_doc = self.lambda_d*(self.FLOPS(embeddings_b))
        flops_query = self.lambda_q*(self.FLOPS(embeddings_a))

        return self.cross_entropy_loss(scores, labels) + flops_doc + flops_query

    def get_config_dict(self):
        return {'scale': self.scale, 'similarity_fct': self.similarity_fct.__name__, "lambda_q": self.lambda_q, "lambda_d": self.lambda_d}

class KLDivLossSplade(nn.Module):
    """
    Compute the KL div loss 
    By default, sim() is the dot-product
    For more details, please refer to https://arxiv.org/abs/2010.02666
    """
    def __init__(self, model, similarity_fct = pairwise_dot_score, lambda_d=8e-2, lambda_q=1e-1, scaled = False, weight_option="default", focal=False, gamma = 2.0, alpha = 0.2, multipos = False, sample_upweight = 0):
        """
        :param model: SentenceTransformerModel
        :param similarity_fct:  Which similarity function to use
        """
        super(KLDivLossSplade, self).__init__()
        self.model = model
        self.similarity_fct = similarity_fct
        self.lambda_d = lambda_d
        self.lambda_q = lambda_q
        self.FLOPS = FLOPS()
        self.L1=L1()
        self.scaled = scaled
        self.focal = focal
        self.multipos = multipos

        self.loss_fct = torch.nn.KLDivLoss(reduction='none', log_target=True)
        
        self.alpha = alpha
        self.weight_option = weight_option
        self.gamma = gamma
        self.sample_upweight = sample_upweight

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        # sentence_features: query, positive passage, negative passage
        results = [self.model(sentence_feature) for sentence_feature in sentence_features]
        # sparse
        reps = [result['sentence_embedding'] for result in results]
        embeddings_query = reps[0]
        embeddings_docs = reps[1:]

        scores = torch.stack([self.similarity_fct(embeddings_query, embeddings_doc) for embeddings_doc in embeddings_docs], dim=1)
        p_scores = torch.nn.functional.softmax(scores, dim=-1)
        log_scores = torch.log(p_scores)

        flops_doc = self.lambda_d * self.FLOPS(torch.cat(embeddings_docs,0))
        flops_query = self.lambda_q * self.L1(embeddings_query)
        
        if self.scaled == True:
            if self.multipos:
                nway = int(labels.shape[1]/3)
            else:
                nway = int(labels.shape[1]/2)

            losses = self.loss_fct(log_scores, labels[:,:nway])
            if not self.focal:
                if self.weight_option == "default":
                    weights = torch.stack([self.alpha/(1+torch.exp(labels[:,i]-labels[:,nway])) + 1 for i in range(nway, labels.shape[1])], 1)
                elif self.weight_option == "mrr_diff":
                    weights =  torch.stack([self.alpha * (1/labels[:,i]-1/labels[:,nway]) + 1 for i in range(nway, labels.shape[1])], 1)
                loss_vector = losses * weights
            elif self.multipos:
                wmasks = labels[:,2*nway:]
                weights =  self.gamma - self.alpha * torch.stack([(1/labels[:,i]-1/labels[:,nway]) for i in range(nway, 2*nway)], 1)
                
                loss_vector = losses * wmasks * (1-p_scores) ** weights + losses * (1 - wmasks) * (p_scores) ** weights
            else:
                wmasks = torch.zeros_like(losses)
                wmasks[:,0] = 1
                weights =  self.gamma - self.alpha * torch.stack([(1/labels[:,i]-1/labels[:,nway]) for i in range(nway, labels.shape[1])], 1)
                
                loss_vector = losses * wmasks * (1-p_scores) ** weights + losses * (1 - wmasks) * (p_scores) ** weights
                #loss_vector = losses * wmasks * (1-p_scores) ** weights + losses * (1 - wmasks) * (1-p_scores) ** weights
               
        else:
            if self.multipos:
                nway = int(labels.shape[1]/2)
                loss_vector = self.loss_fct(log_scores, labels[:, :nway])
            else:
                loss_vector = self.loss_fct(log_scores, labels)

            if self.focal:
                if self.multipos:
                    wmasks = labels[:,nway:]
                else:
                    wmasks = torch.zeros_like(loss_vector)
                    wmasks[:,0] = 1
                loss_vector = loss_vector * wmasks * (1-p_scores) ** self.gamma + loss_vector * (1 - wmasks) * (p_scores) ** self.gamma

            if self.sample_upweight != 0:
                if self.multipos:
                    wmasks = labels[:,nway:]
                    label = labels[:, :nway]
                else:
                    wmasks = torch.zeros_like(loss_vector)
                    wmasks[:,0] = 1
                    label = labels
                
                teacher_pos_score = torch.min(wmasks * label, dim = 1).values
                teacher_neg_score = torch.max((1 - wmasks) * label, dim = 1).values
                teacher_pos_neg_ratio = teacher_pos_score / teacher_neg_score

                sample_weight = self.sample_upweight/(self.sample_upweight+torch.exp(-teacher_pos_neg_ratio.unsqueeze(1))) # (1 + self.sample_upweight) * teacher_pos_neg_ratio.unsqueeze(1)
                return torch.mean(loss_vector * sample_weight) + flops_doc + flops_query
        return torch.mean(loss_vector) + flops_doc + flops_query


class KLAddRegLossSplade(nn.Module):
        
    def __init__(self, model, beta_p = 0.0, similarity_fct = pairwise_dot_score, sample_upweight = 0):
        super(KLAddRegLossSplade, self).__init__()
        self.model = model
        self.loss_fct = torch.nn.KLDivLoss(reduction='none', log_target=True)
        self.beta_p = beta_p
        self.similarity_fct = similarity_fct
        self.sample_upweight = sample_upweight

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        results = [self.model(sentence_feature) for sentence_feature in sentence_features]
        # sparse
        reps = [result['sentence_embedding'] for result in results]
        embeddings_query = reps[0]
        embeddings_docs = reps[1:]

        scores = torch.stack([self.similarity_fct(embeddings_query, embeddings_doc) for embeddings_doc in embeddings_docs], dim=1)
        p_scores = torch.nn.functional.softmax(scores, dim=-1)
        log_scores = torch.log(p_scores)

        nway = int(labels.shape[1]/2)
        losses = self.loss_fct(log_scores, labels[:,:nway])
        
        wmasks = labels[:,nway:]
        if self.sample_upweight != 0:
            teacher_pos_score = torch.min(wmasks * labels[:,:nway], dim = 1).values
            teacher_neg_score = torch.max((1 - wmasks) * labels[:,:nway], dim = 1).values
            teacher_pos_neg_ratio = teacher_pos_score / teacher_neg_score

            loss_pos =  torch.sum(p_scores * log_scores * wmasks)
            loss_neg =  torch.sum(p_scores * (1 - wmasks))

            sample_weight = self.sample_upweight/(self.sample_upweight+torch.exp(-teacher_pos_neg_ratio.unsqueeze(1))) # (1 + self.sample_upweight) * teacher_pos_neg_ratio.unsqueeze(1)
            
            return torch.mean(losses * sample_weight) + self.beta_p * loss_pos + self.beta_p/np.log(2) * loss_neg
        else:
            loss_pos =  torch.sum(p_scores * log_scores * wmasks)
            loss_neg =  torch.sum(p_scores * (1 - wmasks))
            
            return torch.mean(losses) + self.beta_p * loss_pos + self.beta_p/np.log(2) * loss_neg




class KLLogRegLossSplade(nn.Module):
        
    def __init__(self, model, beta_p = 0.0, similarity_fct = pairwise_dot_score, sample_upweight = 0):
        super(KLLogRegLossSplade, self).__init__()
        self.model = model
        self.loss_fct = torch.nn.KLDivLoss(reduction='none', log_target=True)
        self.beta_p = beta_p
        self.similarity_fct = similarity_fct
        self.sample_upweight = sample_upweight

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        results = [self.model(sentence_feature) for sentence_feature in sentence_features]
        # sparse
        reps = [result['sentence_embedding'] for result in results]
        embeddings_query = reps[0]
        embeddings_docs = reps[1:]

        scores = torch.stack([self.similarity_fct(embeddings_query, embeddings_doc) for embeddings_doc in embeddings_docs], dim=1)
        p_scores = torch.nn.functional.softmax(scores, dim=-1)
        log_scores = torch.log(p_scores)

        nway = int(labels.shape[1]/2)
        losses = self.loss_fct(log_scores, labels[:,:nway])
        
        wmasks = labels[:,nway:]
        if self.sample_upweight != 0:
            teacher_pos_score = torch.min(wmasks * labels[:,:nway], dim = 1).values
            teacher_neg_score = torch.max((1 - wmasks) * labels[:,:nway], dim = 1).values
            teacher_pos_neg_ratio = teacher_pos_score / teacher_neg_score

            loss_pos =  -torch.sum(log_scores * wmasks)

            sample_weight = self.sample_upweight/(self.sample_upweight+torch.exp(-teacher_pos_neg_ratio.unsqueeze(1))) # (1 + self.sample_upweight) * teacher_pos_neg_ratio.unsqueeze(1)
            return torch.mean(losses * sample_weight) + self.beta_p * loss_pos
        else:
            loss_pos =  -torch.sum(log_scores * wmasks)
            
            return torch.mean(losses) + self.beta_p * loss_pos 



class KLDivLossSpladeInBatch(nn.Module):
    def __init__(self, model, similarity_fct = dot_score, lambda_d=8e-2, lambda_q=1e-1, inbatch_p = 0.0):
        """
        :param model: SentenceTransformerModel
        :param similarity_fct:  Which similarity function to use
        """
        super(KLDivLossSpladeInBatch, self).__init__()
        self.model = model
        self.similarity_fct = similarity_fct
        self.loss_fct = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)
        self.loss_inbatch = nn.CrossEntropyLoss()
        self.lambda_d = lambda_d
        self.lambda_q = lambda_q
        self.FLOPS = FLOPS()
        self.L1 = L1()
        self.de_teacher = None
        self.inbatch_p = inbatch_p
    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        # sentence_features: query, positive passage, negative passage
        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        embeddings_query = reps[0]
        embeddings_pos = reps[1]
        embeddings_negs = reps[2:]

        scores_pos = self.similarity_fct(embeddings_query, embeddings_pos)
        scores_negs = [self.similarity_fct(embeddings_query, embeddings_neg) for embeddings_neg in embeddings_negs]

        ### flops
        flops_doc = self.lambda_d * self.FLOPS(torch.cat(reps[1:], 0))
        #flops_query = self.lambda_q*(self.FLOPS(embeddings_query))
        flops_query = self.lambda_q * self.L1(embeddings_query)

        ### hard negative kldiv
        scores = torch.stack([torch.diagonal(scores_pos, 0)] + [torch.diagonal(scores_neg, 0) for scores_neg in scores_negs], dim=1)
        p_scores = torch.nn.functional.softmax(scores, dim=-1)
        log_scores = torch.log(p_scores)

        loss_vector = self.loss_fct(log_scores, labels)
        
        # inbatch loss
        pred_inbatch = torch.cat([scores_pos] + [scores_neg for scores_neg in scores_negs],1)
        labels_inbatch = torch.tensor(range(len(pred_inbatch)), dtype=torch.long, device=pred_inbatch.device)  
       
        loss_inbatch = self.loss_inbatch(pred_inbatch, labels_inbatch)
        
        return torch.mean(loss_vector) + loss_inbatch * self.inbatch_p + flops_doc + flops_query

class CRLossSplade(nn.Module):
    def __init__(self, model,similarity_fct = pairwise_dot_score, lambda_d=8e-2, lambda_q=1e-1):
        super(CRLossSplade, self).__init__()
        self.model = model
        self.similarity_fct = similarity_fct
        self.loss_fct = nn.MSELoss()
        self.lambda_d = lambda_d
        self.lambda_q = lambda_q
        self.FLOPS = FLOPS()
        self.L1=L1()

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        
        scores  = [self.similarity_fct(reps[0], doc_emb) for doc_emb in reps[1:]]
        scores = torch.stack(scores, dim=1)

        nway = int(labels.shape[1]/2)
        ys = labels[:,:nway]
        idx = labels[:,nway:]

        scores_diffs = torch.triu(scores[:, :, None] - scores[:, None, :])
        weights = torch.abs(idx[:, :, None] - idx[:, None, :])
        ys_pairs = (ys[:, :, None] - ys[:, None, :]) > 0
        losses = torch.log(1. + torch.exp(-scores_diffs)) * weights#[bz, topk, topk]
        flops_doc = self.lambda_d*self.FLOPS(torch.cat(reps, 0))
        flops_query = self.lambda_q * self.L1(reps[0])

        return torch.sum(losses[ys_pairs]) + flops_doc + flops_query


class MarginMSELossSpladeInBatch(nn.Module):
    """
    Compute the MSE loss between the |sim(Query, Pos) - sim(Query, Neg)| and |gold_sim(Q, Pos) - gold_sim(Query, Neg)|
    By default, sim() is the dot-product
    For more details, please refer to https://arxiv.org/abs/2010.02666
    """
    def __init__(self, model, similarity_fct = dot_score, lambda_d=8e-2, lambda_q=1e-1, de_teacher = None):
        """
        :param model: SentenceTransformerModel
        :param similarity_fct:  Which similarity function to use
        """
        super(MarginMSELossSpladeInBatch, self).__init__()
        self.model = model
        self.similarity_fct = similarity_fct
        self.loss_fct_pair = nn.MSELoss()
        if de_teacher is not None:
            self.loss_inbatch = nn.MSELoss()
        else:
            self.loss_inbatch = nn.CrossEntropyLoss()
        self.lambda_d = lambda_d
        self.lambda_q = lambda_q
        self.FLOPS = FLOPS()
        if de_teacher is not None:
            self.de_teacher = de_teacher
        else:
            self.de_teacher = None

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor, inbatch_p = 0):
        # sentence_features: query, positive passage, negative passage
        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        embeddings_query = reps[0]
        embeddings_pos = reps[1]
        embeddings_neg = reps[2]
        #print(labels.shape)
        scores_pos = self.similarity_fct(embeddings_query, embeddings_pos)
        scores_neg = self.similarity_fct(embeddings_query, embeddings_neg)


        ### if there is a dual encoder teacher
        if self.de_teacher is not None:
            embeddings_query_teacher = self.de_teacher.inference.query(sentence_features[0]['input_ids'], sentence_features[0]['attention_mask'])
            embeddings_pos_teacher = self.de_teacher.inference.doc(sentence_features[1]['input_ids'], sentence_features[1]['attention_mask'])
            embeddings_neg_teacher = self.de_teacher.inference.doc(sentence_features[2]['input_ids'], sentence_features[2]['attention_mask'])
            
            iblabels = []
            for emb_query_teacher in embeddings_query_teacher:
                emb_q = torch.transpose(emb_query_teacher.unsqueeze(0), 1,2)
                pos_scores = self.de_teacher.inference.score(emb_q, embeddings_pos_teacher)
                neg_scores = self.de_teacher.inference.score(emb_q, embeddings_neg_teacher)
                iblabels.append(torch.cat([pos_scores, neg_scores])) #1x2B
            labels_inbatch = torch.stack(iblabels)
            #print("labels_inbatch", labels_inbatch)
            
        ### flops
        flops_doc = self.lambda_d*self.FLOPS(torch.cat(reps,0))
        flops_query = self.lambda_q*(self.L1(embeddings_query))
        ### hard negative mse
        loss_pair = self.loss_fct_pair(torch.diagonal(scores_pos, 0) - torch.diagonal(scores_neg, 0), labels[:,0] - labels[:,1])

        #print("labels", labels)
        
        # inbatch loss
        pred_inbatch = torch.cat([scores_pos,scores_neg],1)
        #print("pred", pred_inbatch)
        if self.de_teacher is None:
            labels_inbatch = torch.tensor(range(len(pred_inbatch)), dtype=torch.long, device=pred_inbatch.device)  
        else:
            pred_inbatch = torch.diagonal(pred_inbatch, 0).unsqueeze(1) - pred_inbatch
            labels_inbatch = torch.diagonal(labels_inbatch, 0).unsqueeze(1) - labels_inbatch
            labels_inbatch = labels_inbatch.to(pred_inbatch.device)
        
        loss_inbatch = self.loss_inbatch(pred_inbatch, labels_inbatch)
        
        if inbatch_p is None:
            # upweight uncertain results
            pred_probs = F.softmax(pred_inbatch, dim=-1)
            pred_entropy = - torch.sum(pred_probs * torch.log(pred_probs + 1e-6), dim=1)
            instance_weight = pred_entropy / torch.log(torch.ones_like(pred_entropy) * pred_inbatch.size(1))
            instance_weight = instance_weight.detach().mean()
            results =  2 * (1 - instance_weight) * loss_pair + 2 * instance_weight * loss_inbatch + flops_doc + flops_query
            #print("loss pair", loss_pair) #tensor(505525.8438, device='cuda:0', grad_fn=<MseLossBackward>)
            #print("loss inbatch", loss_inbatch) # tensor(614.6311, device='cuda:0', grad_fn=<NllLossBackward>)
            #print(pred_entropy)
            #print(instance_weight)
            return results
        
        return loss_pair + loss_inbatch * inbatch_p + flops_doc + flops_query


############ Colbert Loss##############
class MarginMSELossColBERTWithDense(nn.Module):
    """
    Compute the MSE loss between the |sim(Query, Pos) - sim(Query, Neg)| and |gold_sim(Q, Pos) - gold_sim(Query, Neg)|
    By default, sim() is the dot-product
    For more details, please refer to https://arxiv.org/abs/2010.02666
    """
    def __init__(self, model, similarity_fct = pairwise_dot_score, dense_weight=0.0, scaled = False, alpha = 0.2, weight_option="default"):
        """
        :param model: SentenceTransformerModel
        :param similarity_fct:  Which similarity function to use
        """
        super(MarginMSELossColBERTWithDense, self).__init__()
        self.model = model
        self.similarity_fct = similarity_fct
        self.loss_fct_pair = nn.MSELoss()
        self.dense_weight = dense_weight
        self.scaled = scaled
        self.alpha = alpha
        self.weight_option = weight_option

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        
        # sentence_features: query, positive passage, negative passage
        results = [self.model(sentence_feature) for sentence_feature in sentence_features]
        
        # token
        token_reps = [result['last_layer_embeddings'] for result in results]
        masks = [result['attention_mask'] for result in results]
        token_rep_query = torch.nn.functional.normalize(token_reps[0], p=2, dim=2)
        token_rep_pos =  token_reps[1] * masks[1].unsqueeze(-1)
        token_rep_pos = torch.nn.functional.normalize(token_rep_pos)
        token_rep_neg = token_reps[2] * masks[2].unsqueeze(-1)
        token_rep_neg = torch.nn.functional.normalize(token_rep_neg)

        dense_scores_pos = (token_rep_query @ token_rep_pos.permute(0,2,1)).max(2).values.sum(1)
        dense_scores_neg = (token_rep_query @ token_rep_neg.permute(0,2,1)).max(2).values.sum(1)

        # cls
        clss = [result['cls'] for result in results]
        cls_query = clss[0]
        cls_pos =  clss[1]
        cls_neg = clss[2]
        cls_scores_pos = self.similarity_fct(cls_query, cls_pos)
        cls_scores_neg = self.similarity_fct(cls_query, cls_neg)
        preds = self.dense_weight * cls_scores_pos +  dense_scores_pos - self.dense_weight * cls_scores_neg - dense_scores_neg
        
        if self.scaled:
            if self.weight_option == "default":
                weight = self.alpha/(1+torch.exp(labels[:,2]-labels[:,1])) + 1
            elif self.weight_option == "mrr_diff":
                weight = self.alpha * (1/labels[:,2]-1/labels[:,1]) + 1
            loss_pair = (weight * (preds - labels[:,0]) ** 2).mean()
        else:
            loss_pair = self.loss_fct_pair(preds, labels)
       
        return loss_pair 


class KLDivLossColBERT(nn.Module):
    
    def __init__(self, model, alpha = 0.2, beta = 0.0, focal=False, gamma = 2.0, multipos = False, ib_p = 0.0, sample_upweight = 0, temp = 1):
        super(KLDivLossColBERT, self).__init__()
        self.model = model
        self.focal = focal
        if self.focal:
            self.loss_fct = torch.nn.KLDivLoss(reduction='none', log_target=True)
        else:
            self.loss_fct = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)
        self.alpha = alpha
        self.gamma = gamma
        self.multipos = multipos
        self.beta = beta
        self.inbatch_p = ib_p
        self.sample_upweight = sample_upweight
        self.temp = temp

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        results = [self.model(sentence_feature) for sentence_feature in sentence_features]
        # token
        token_reps = [result['last_layer_embeddings'] for result in results]
        masks = [result['attention_mask'] for result in results]
        #print("qshape", token_reps[0].shape) # B x QLEN x 128
        token_rep_query = torch.nn.functional.normalize(token_reps[0], p=2, dim=2)
        token_scores = []
        for idx in range(1, len(token_reps)):
            token_rep = token_reps[idx] * masks[idx].unsqueeze(-1)
            #token_rep = torch.nn.functional.normalize(token_rep) BIG CHANGE
            token_rep = torch.nn.functional.normalize(token_rep, p=2, dim=2)
            token_level_score = token_rep_query @ token_rep.permute(0,2,1)
            iter_mask = ~masks[idx].unsqueeze(1).repeat(1, token_level_score.shape[1], 1).bool()
            token_level_score[iter_mask] = -9999
            token_scores.append(token_level_score.max(2).values.sum(1))
        
        scores = torch.stack(token_scores, dim=1)
        p_scores = torch.nn.functional.softmax(scores/self.temp, dim=-1)
        log_scores = torch.log(p_scores)

    
        if self.multipos:
            nway = int(labels.shape[1]/2)
            loss_vector = self.loss_fct(log_scores, labels[:, :nway])
        else:
            loss_vector = self.loss_fct(log_scores, labels)

        if self.focal:
            if self.multipos:
                wmasks = labels[:,nway:]
            else:
                wmasks = torch.zeros_like(loss_vector)
                wmasks[:,0] = 1

            loss_vector = loss_vector * wmasks * (1-p_scores) ** self.gamma + loss_vector * (1 - wmasks) * (p_scores) ** (self.gamma - self.beta)
            
        if self.sample_upweight != 0:
            if self.multipos:
                nway = int(labels.shape[1]/2)
                wmasks = labels[:,nway:]
                label = labels[:, :nway]
            else:
                wmasks = torch.zeros_like(loss_vector)
                wmasks[:,0] = 1
                label = labels
            
            loss_vector = self.loss_fct(log_scores, label)
            teacher_pos_score = torch.min(wmasks * label, dim = 1).values
            teacher_neg_score = torch.max((1 - wmasks) * label, dim = 1).values
            teacher_pos_neg_ratio = teacher_pos_score / teacher_neg_score

            sample_weight = self.sample_upweight/(self.sample_upweight+torch.exp(-teacher_pos_neg_ratio.unsqueeze(1))) # (1 + self.sample_upweight) * teacher_pos_neg_ratio.unsqueeze(1)
            return torch.mean(loss_vector * sample_weight)

        if self.inbatch_p > 0:
            if self.multipos:
                wmasks = labels[:,2*nway:]
                #print('wmasks.shape', wmasks.shape) #batchQ * nway + 1
            # inbatch loss
            token_q_scores_ib = []
            token_q_masks_ib = []
            #print("token_rep_query.shape", token_rep_query.shape) batchQ * 32 *768
            #print("wmask.shape", wmasks.shape) batch * (nway + 1)
            # print("len(token_reps)", len(token_reps)) nway + 2
            for qindex in range(token_rep_query.shape[0]): #batchQ
                ib_scores = []
                ib_masks = []
                for idx in range(1, len(token_reps)): #nway + 1
                    token_rep = token_reps[idx] * masks[idx].unsqueeze(-1)
                    #token_rep = torch.nn.functional.normalize(token_rep)
                    token_rep = torch.nn.functional.normalize(token_rep, p=2, dim=2) #BIG CHANGE
                    q_d_onedim = (token_rep_query[qindex:(qindex+1)] @ token_rep.permute(0,2,1)).max(2).values.sum(1)
                    ib_scores.append(q_d_onedim)
                    mask_onedim = torch.zeros(q_d_onedim.shape).to(q_d_onedim.device)
                    if self.multipos:
                        mask_onedim[qindex] = wmasks[qindex, idx-1]
                    else:
                        if idx == 1:
                            mask_onedim[0,qindex] = 1
                    ib_masks.append(mask_onedim)

                token_q_scores_ib.append(torch.cat(ib_scores)) #nway+1 * batchD
                token_q_masks_ib.append(torch.cat(ib_masks))   
            pred_inbatch = torch.log_softmax(torch.stack(token_q_scores_ib,0), 1) 
            
            pred_inbatch = pred_inbatch * torch.stack(token_q_masks_ib, 0) # batchQ * nway,batchD 
            logp = pred_inbatch.sum(1)
            loss_inbatch = self.inbatch_p * torch.mean(logp)
            print(f"loss kl: {torch.mean(loss_vector)}, loss ib: {loss_inbatch}")
            return torch.mean(loss_vector) + loss_inbatch
        else:
            #print(f"loss kl: {torch.mean(loss_vector)}")
            return torch.mean(loss_vector)

class KLFocalLossColBERT(nn.Module):
        
    def __init__(self, model, gamma = 5, similarity_fct = pairwise_dot_score):
        super(KLFocalLossColBERT, self).__init__()
        self.model = model
        self.loss_fct = torch.nn.KLDivLoss(reduction='none', log_target=True)
        self.gamma = gamma
        self.similarity_fct = similarity_fct

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        results = [self.model(sentence_feature) for sentence_feature in sentence_features]
        token_reps = [result['last_layer_embeddings'] for result in results]
        masks = [result['attention_mask'] for result in results]
        token_rep_query = torch.nn.functional.normalize(token_reps[0], p=2, dim=2)

        token_scores = []
        for idx in range(1, len(token_reps)):
            token_rep = token_reps[idx] * masks[idx].unsqueeze(-1)
            token_rep = torch.nn.functional.normalize(token_rep)
            token_scores.append((token_rep_query @ token_rep.permute(0,2,1)).max(2).values.sum(1))
        
        scores = torch.stack(token_scores, dim=1)
        p_scores = torch.nn.functional.softmax(scores, dim=-1)
        log_scores = torch.log(p_scores)

        nway = int(labels.shape[1]/2)
        losses = self.loss_fct(log_scores, labels[:,:nway])
        loss_vector = losses * (p_scores) ** self.gamma
        return torch.mean(loss_vector)


class KLDivLossColBERTPrintLoss(nn.Module):
    
    def __init__(self, model, scaled = False, alpha = 0.2, beta = 0.0, focal=False, gamma = 2.0, multipos = False):
        super(KLDivLossColBERTPrintLoss, self).__init__()
        self.model = model
        self.scaled = scaled
        self.focal = focal
        self.loss_fct = torch.nn.KLDivLoss(reduction='none', log_target=True)
        
        self.alpha = alpha
        self.gamma = gamma
        self.multipos = multipos
        self.beta = beta

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        results = [self.model(sentence_feature) for sentence_feature in sentence_features]
        # token
        token_reps = [result['last_layer_embeddings'] for result in results]
        masks = [result['attention_mask'] for result in results]
        token_rep_query = torch.nn.functional.normalize(token_reps[0], p=2, dim=2)

        token_scores = []
        for idx in range(1, len(token_reps)):
            token_rep = token_reps[idx] * masks[idx].unsqueeze(-1)
            #token_rep = torch.nn.functional.normalize(token_rep)
            token_rep = torch.nn.functional.normalize(token_rep, p=2, dim=2) #BIG CHANGE
            token_scores.append((token_rep_query @ token_rep.permute(0,2,1)).max(2).values.sum(1))
        
        
        scores = torch.stack(token_scores, dim=1)
        p_scores = torch.nn.functional.softmax(scores, dim=-1)
        log_scores = torch.log(p_scores)

        if self.scaled == True:
            if self.multipos:
                nway = int(labels.shape[1]/3)
            else:
                nway = int(labels.shape[1]/2)
            losses = self.loss_fct(log_scores, labels[:,:nway])
            if not self.focal:
                loss_vector = losses
            elif self.multipos:
                wmasks = labels[:,2*nway:]
                weights =  self.gamma - self.alpha * torch.stack([(1/labels[:,i]-1/labels[:,nway]) for i in range(nway, 2*nway)], 1)
                loss_pos = losses * wmasks * (1-p_scores) ** weights
                loss_neg = losses * (1 - wmasks) * (p_scores) ** weights
                loss_vector = loss_pos + loss_neg
            else:
                wmasks = torch.zeros_like(losses)
                wmasks[:,0] = 1
                weights =  self.gamma - self.alpha * torch.stack([(1/labels[:,i]-1/labels[:,nway]) for i in range(nway, labels.shape[1])], 1)
                loss_pos = losses * wmasks * (1-p_scores) ** weights
                loss_neg = losses * (1 - wmasks) * (p_scores) ** weights
                loss_vector = loss_pos + loss_neg
        else:
            if self.multipos:
                nway = int(labels.shape[1]/2)
                wmasks = labels[:,nway:]
                loss_vector = self.loss_fct(log_scores, labels[:, :nway])
                loss_pos = loss_vector * wmasks
                loss_neg = loss_vector * (1-wmasks)
            else:
                loss_vector = self.loss_fct(log_scores, labels)
                wmasks = torch.zeros_like(loss_vector)
                wmasks[:,0] = 1
                loss_pos = loss_vector * wmasks
                loss_neg = loss_vector * (1-wmasks)
            if self.focal:
                loss_pos = loss_vector * wmasks * (1-p_scores) ** self.gamma
                loss_neg = loss_vector * (1 - wmasks) * (p_scores) ** (self.gamma - self.beta)
                loss_vector = loss_pos + loss_neg


        return torch.mean(loss_vector), torch.mean(loss_neg), torch.mean(loss_pos)


class KLAddRegLossColBERT(nn.Module):
    
    def __init__(self, model, beta_p = 0.0,  sample_upweight = 0, temp = 1):
        super(KLAddRegLossColBERT, self).__init__()
        self.model = model
        self.loss_fct = torch.nn.KLDivLoss(reduction='none', log_target=True)
        self.beta_p = beta_p
        self.sample_upweight = sample_upweight
        self.temp = temp

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        
        results = [self.model(sentence_feature) for sentence_feature in sentence_features]
        # token
        token_reps = [result['last_layer_embeddings'] for result in results]
        masks = [result['attention_mask'] for result in results]
        token_rep_query = torch.nn.functional.normalize(token_reps[0], p=2, dim=2)

        token_scores = []
        for idx in range(1, len(token_reps)):
            token_rep = token_reps[idx] * masks[idx].unsqueeze(-1)
            #token_rep = torch.nn.functional.normalize(token_rep)
            token_rep = torch.nn.functional.normalize(token_rep, p=2, dim=2)
            token_level_score = token_rep_query @ token_rep.permute(0,2,1)
            iter_mask = ~masks[idx].unsqueeze(1).repeat(1, token_level_score.shape[1], 1).bool()
            token_level_score[iter_mask] = -9999
            token_scores.append(token_level_score.max(2).values.sum(1))

        scores = torch.stack(token_scores, dim=1)
        p_scores = torch.nn.functional.softmax(scores/self.temp, dim=-1)
        log_scores = torch.log(p_scores)

        
        nway = int(labels.shape[1]/2)
        losses = self.loss_fct(log_scores, labels[:,:nway])
        wmasks = labels[:,nway:]
        if self.sample_upweight > 0:
            label_p = torch.softmax(labels[:,:nway],dim=1)
            
            teacher_pos_score = torch.min(torch.where(wmasks == 1, label_p, torch.tensor(float('inf'), dtype = labels.dtype).to(labels.device)), dim = 1).values
            teacher_neg_score = torch.max(torch.where(wmasks == 0, label_p, torch.tensor(-float('inf'), dtype = labels.dtype).to(labels.device)), dim = 1).values
            
            teacher_pos_neg_ratio = teacher_pos_score / teacher_neg_score
            loss_pos =  torch.sum(p_scores * log_scores * wmasks)
            loss_neg =  torch.sum(p_scores * (1 - wmasks))

            sample_weight = self.sample_upweight/(self.sample_upweight+torch.exp(-teacher_pos_neg_ratio.unsqueeze(1))) # (1 + self.sample_upweight) * teacher_pos_neg_ratio.unsqueeze(1)
            return torch.mean(losses * sample_weight) + self.beta_p * loss_pos + self.beta_p/np.log(2) * loss_neg
        else:
            loss_pos =  torch.sum(p_scores * log_scores * wmasks)
            loss_neg =  torch.sum(p_scores * (1 - wmasks))
            
            return torch.mean(losses) + self.beta_p * loss_pos + self.beta_p/np.log(2)  * loss_neg


class KLLogRegLossColBERT(nn.Module):
    
    def __init__(self, model, beta_p = 0.0, sample_upweight = 0):
        super(KLLogRegLossColBERT, self).__init__()
        self.model = model
        self.loss_fct = torch.nn.KLDivLoss(reduction='none', log_target=True)
        self.beta_p = beta_p
        self.sample_upweight = sample_upweight

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        
        results = [self.model(sentence_feature) for sentence_feature in sentence_features]
        # token
        token_reps = [result['last_layer_embeddings'] for result in results]
        masks = [result['attention_mask'] for result in results]
        token_rep_query = torch.nn.functional.normalize(token_reps[0], p=2, dim=2)

        token_scores = []
        for idx in range(1, len(token_reps)):
            token_rep = token_reps[idx] * masks[idx].unsqueeze(-1)
            #token_rep = torch.nn.functional.normalize(token_rep)
            token_rep = torch.nn.functional.normalize(token_rep, p=2, dim=2)
            token_level_score = token_rep_query @ token_rep.permute(0,2,1)
            iter_mask = ~masks[idx].unsqueeze(1).repeat(1, token_level_score.shape[1], 1).bool()
            token_level_score[iter_mask] = -9999
            token_scores.append(token_level_score.max(2).values.sum(1))
        
        scores = torch.stack(token_scores, dim=1)
        p_scores = torch.nn.functional.softmax(scores, dim=-1)
        log_scores = torch.log(p_scores)

        
        nway = int(labels.shape[1]/2)
        losses = self.loss_fct(log_scores, labels[:,:nway])
        wmasks = labels[:,nway:]
        if self.sample_upweight > 0:
            teacher_pos_score = torch.min(wmasks * labels[:,:nway], dim = 1).values
            teacher_neg_score = torch.max((1 - wmasks) * labels[:,:nway], dim = 1).values
            teacher_pos_neg_ratio = teacher_pos_score / teacher_neg_score

            loss_pos =  -torch.sum(log_scores * wmasks)

            sample_weight = self.sample_upweight/(self.sample_upweight+torch.exp(-teacher_pos_neg_ratio.unsqueeze(1))) # (1 + self.sample_upweight) * teacher_pos_neg_ratio.unsqueeze(1)
            return torch.mean(losses * sample_weight) + self.beta_p * loss_pos
        else:
            loss_pos =  -torch.sum(log_scores * wmasks)
            
            return torch.mean(losses) + self.beta_p * loss_pos



class ConvexSHLossColBERT(nn.Module):
    
    def __init__(self, model, alpha = 0.2, gamma = 2.0):
        super(ConvexSHLossColBERT, self).__init__()
        self.model = model
        self.loss_fct = torch.nn.KLDivLoss(reduction='none', log_target=True)
        self.alpha = alpha
        self.gamma = gamma


    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        results = [self.model(sentence_feature) for sentence_feature in sentence_features]
        # token
        token_reps = [result['last_layer_embeddings'] for result in results]
        masks = [result['attention_mask'] for result in results]
        token_rep_query = torch.nn.functional.normalize(token_reps[0], p=2, dim=2)

        token_scores = []
        for idx in range(1, len(token_reps)):
            token_rep = token_reps[idx] * masks[idx].unsqueeze(-1)
            #token_rep = torch.nn.functional.normalize(token_rep)
            token_rep = torch.nn.functional.normalize(token_rep, p=2, dim=2) #BIG CHANGE
            token_scores.append((token_rep_query @ token_rep.permute(0,2,1)).max(2).values.sum(1))
        
        
        scores = torch.stack(token_scores, dim=1)
        p_scores = torch.nn.functional.softmax(scores, dim=-1)
    
        nway = int(labels.shape[1]/3)
        wmasks = labels[:,2*nway:]
        teacher = labels[:,:nway]

        p_scores = p_scores * wmasks + (1-p_scores) * (1-wmasks) # modify top 1 to non-top one for negatives
        teacher_inverse = teacher * wmasks + (1-teacher) * (1-wmasks) #modify top 1 to non-top one for negatives - teacher

        log_scores = torch.log(p_scores)

        #losses = self.loss_fct(log_scores, teacher)
        losses = teacher * (teacher_inverse.log() - log_scores)
        
        weights =  self.gamma - self.alpha * torch.stack([(1/labels[:,i]-1/labels[:,nway]) for i in range(nway, 2*nway)], 1)
        loss_vector = losses * wmasks * (1-p_scores) ** weights + losses * (1 - wmasks) * (p_scores) ** weights

        return torch.mean(loss_vector)


class MarginKLDivLossColBERT(nn.Module):
    def __init__(self, model, similarity_fct = pairwise_dot_score, scaled = False, prf=False, alpha = 0.2, weight_option="default"):
        super(MarginKLDivLossColBERT, self).__init__()
        self.model = model
        self.similarity_fct = similarity_fct
        self.scaled = scaled
        if self.scaled:
            self.loss_fct = torch.nn.KLDivLoss(reduction='none', log_target=True)
        else:
            self.loss_fct = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)
        self.prf = prf
        self.alpha = alpha
        self.weight_option = weight_option

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        results = [self.model(sentence_feature) for sentence_feature in sentence_features]

        # token
        token_reps = [result['last_layer_embeddings'] for result in results]
        masks = [result['attention_mask'] for result in results]
        token_rep_query = torch.nn.functional.normalize(token_reps[0], p=2, dim=2)
        
        if self.prf:
            token_rep_query_prf = torch.nn.functional.normalize(token_reps[1], p=2, dim=2)
            idx_pos = 2
        else:
            idx_pos = 1
            
        token_rep_pos =  token_reps[idx_pos] * masks[idx_pos].unsqueeze(-1)
        token_rep_pos = torch.nn.functional.normalize(token_rep_pos)
        dense_scores_pos = (token_rep_query @ token_rep_pos.permute(0,2,1)).max(2).values.sum(1)
        
        if self.prf:
            dense_scores_pos_prf = (token_rep_query_prf @ token_rep_pos.permute(0,2,1)).max(2).values.sum(1)
            dense_scores_pos = dense_scores_pos + 0.1 * dense_scores_pos_prf
            
        token_scores = []
        
        for idx in range(idx_pos + 1, len(token_reps)):
            token_neg_rep = token_reps[idx] * masks[idx].unsqueeze(-1)
            token_neg_rep = torch.nn.functional.normalize(token_neg_rep)
            
            dense_scores_neg = (token_rep_query @ token_neg_rep.permute(0,2,1)).max(2).values.sum(1)
            if self.prf:
                dense_scores_neg_prf = (token_rep_query_prf @ token_neg_rep.permute(0,2,1)).max(2).values.sum(1)
                dense_scores_neg = dense_scores_neg + 0.1 * dense_scores_neg_prf
                
            token_scores.append(dense_scores_neg - dense_scores_pos)
        
        scores = torch.stack(token_scores, dim=1)
        log_scores = torch.nn.functional.log_softmax(scores, dim=-1)
        
       
        if self.scaled == True:
            nway = int((labels.shape[1]-1)/2)
            losses = self.loss_fct(log_scores, labels[:,:nway])
            if self.weight_option == "default":
                weights = torch.stack([self.alpha/(1+torch.exp(labels[:,i]-labels[:,nway])) + 1 for i in range(nway + 1, labels.shape[1])], 1)
            elif self.weight_option == "mrr_diff":
                weights =  torch.stack([self.alpha * (1/labels[:,i]-1/labels[:,nway]) + 1 for i in range(nway + 1, labels.shape[1])], 1)
                
            return torch.mean(losses * weights)
        
        return torch.mean(self.loss_fct(log_scores, labels))

class MultipleNegativesRankingLossColBERT(nn.Module):
    def __init__(self, model, scaled: bool = False, prf = False, alpha = 0.2,weight_option='default'):
        super(MultipleNegativesRankingLossColBERT, self).__init__()
        self.model = model
        self.scaled = scaled
        if self.scaled:
            self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')
        else:
            self.cross_entropy_loss = nn.CrossEntropyLoss()
            
        self.prf = prf
        self.alpha = alpha
        self.weight_option = weight_option


    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], wlabels: Tensor):
        results = [self.model(sentence_feature) for sentence_feature in sentence_features]
        
        # token
        token_reps = [result['last_layer_embeddings'] for result in results]
        masks = [result['attention_mask'] for result in results]
        token_rep_query = torch.nn.functional.normalize(token_reps[0], p=2, dim=2)
        
        if self.prf:
            token_rep_query_prf = torch.nn.functional.normalize(token_reps[1], p=2, dim=2)
            idx_pos = 2
        else:
            idx_pos = 1
            
        token_scores = []
        for idx in range(idx_pos, len(token_reps)):
            token_neg_rep = token_reps[idx] * masks[idx].unsqueeze(-1)
            token_neg_rep = torch.nn.functional.normalize(token_neg_rep)
            
            dense_scores_neg = (token_rep_query @ token_neg_rep.permute(0,2,1)).max(2).values.sum(1)
            if self.prf:
                dense_scores_neg_prf = (token_rep_query_prf @ token_neg_rep.permute(0,2,1)).max(2).values.sum(1)
                dense_scores_neg = dense_scores_neg + 0.1 * dense_scores_neg_prf
              
            token_scores.append(dense_scores_neg)
        
        token_scores = torch.stack(token_scores, dim=1)    
        labels = torch.zeros(token_scores.shape[0], device=token_scores.device, dtype=torch.long)  # Example a[i] should match with b[i]
        
       
        if self.scaled:
            if self.weight_option == "default":
                weight = torch.cat([self.alpha/(1+torch.exp(wlabels[:,i]-wlabels[:,0])) + 1 for i in range(1, wlabels.shape[1])])
            elif self.weight_option == "mrr_diff":
                weight =  torch.cat([self.alpha * (1/labels[:,i]-1/labels[:,1]) + 1 for i in range(1, wlabels.shape[1])])
            
            losses = self.cross_entropy_loss(token_scores, labels)
            loss_pair = (weight * losses).mean()
        else:
            loss_pair = self.cross_entropy_loss(token_scores, labels)
       
        return loss_pair

    def get_config_dict(self):
        return {'scale': self.scaled}

class KLDivLossColBERTInBatch(nn.Module):
    def __init__(self, model, inbatch_p = 0.0):
        """
        :param model: SentenceTransformerModel
        :param similarity_fct:  Which similarity function to use
        """
        super(KLDivLossColBERTInBatch, self).__init__()
        self.model = model
        self.loss_fct = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)
        self.loss_inbatch = nn.CrossEntropyLoss()
        self.inbatch_p = inbatch_p

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        # sentence_features: query, positive passage, negative passage
        results = [self.model(sentence_feature) for sentence_feature in sentence_features]

        # token kl
        token_reps = [result['last_layer_embeddings'] for result in results]
        masks = [result['attention_mask'] for result in results]
        token_rep_query = torch.nn.functional.normalize(token_reps[0], p=2, dim=2)

        token_scores = []
        for idx in range(1, len(token_reps)):
            token_reps[idx] = token_reps[idx] * masks[idx].unsqueeze(-1)
            token_reps[idx] = torch.nn.functional.normalize(token_reps[idx])
            token_scores.append((token_rep_query @ token_reps[idx].permute(0,2,1)).max(2).values.sum(1))
        
        scores = torch.stack(token_scores, dim=1)
        p_scores = torch.nn.functional.softmax(scores, dim=-1)
        log_scores = torch.log(p_scores)
        loss_vector = self.loss_fct(log_scores, labels)

        # inbatch loss
        token_q_scores_ib = []
        for qindex in range(token_rep_query.shape[0]): #batchQ
            ib_scores = []
            for idx in range(1, len(token_reps)): #nway + 1
                ib_scores.append((token_rep_query[qindex:(qindex+1)] @ token_reps[idx].permute(0,2,1)).max(2).values.sum(1))
            token_q_scores_ib.append(torch.cat(ib_scores)) #nway+1 * batchD

        pred_inbatch = torch.stack(token_q_scores_ib,0) # batchQ * nway,batchD 
        labels_inbatch = torch.tensor(list(range(len(pred_inbatch))), dtype=torch.long, device=pred_inbatch.device)  
        loss_inbatch = self.inbatch_p * self.loss_inbatch(pred_inbatch, labels_inbatch)

        return torch.mean(loss_vector) + loss_inbatch


class CRLossColBERT(nn.Module):
    def __init__(self, model):
        super(CRLossColBERT, self).__init__()
        self.model = model

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        results = [self.model(sentence_feature) for sentence_feature in sentence_features]
        # token
        token_reps = [result['last_layer_embeddings'] for result in results]
        masks = [result['attention_mask'] for result in results]
        token_rep_query = torch.nn.functional.normalize(token_reps[0], p=2, dim=2)

        token_scores = []
        for idx in range(1, len(token_reps)):
            token_rep = token_reps[idx] * masks[idx].unsqueeze(-1)
            #token_rep = torch.nn.functional.normalize(token_rep)
            token_rep = torch.nn.functional.normalize(token_rep, p=2, dim=2) #BIG CHANGE
            token_scores.append((token_rep_query @ token_rep.permute(0,2,1)).max(2).values.sum(1))
        
        
        scores = torch.stack(token_scores, dim=1)

        nway = int(labels.shape[1]/2)
        ys = labels[:,:nway]
        idx = labels[:,nway:]

        scores_diffs = torch.triu(scores[:, :, None] - scores[:, None, :])
        weights = torch.abs(idx[:, :, None] - idx[:, None, :])
        ys_pairs = (ys[:, :, None] - ys[:, None, :]) > 0
        losses = torch.log(1. + torch.exp(-scores_diffs)) * weights#[bz, topk, topk]

        return torch.sum(losses[ys_pairs])


############### CE loss ######################
class MarginKLDivLossCE(nn.Module):
    def __init__(self, model, scaled = False, alpha = 0.2):
        super(MarginKLDivLossCE, self).__init__()
        self.model = model
        self.scaled = scaled
        if self.scaled:
            self.loss_fct = torch.nn.KLDivLoss(reduction='none', log_target=True)
        else:
            self.loss_fct = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)
        self.alpha = alpha
        
    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor, scaled = False):
        reps = [self.model(sentence_feature)['cls'] for sentence_feature in sentence_features]

        scores  = torch.cat([reps[0] - x for x in reps[1:]], dim=1)        
        log_scores = torch.nn.functional.log_softmax(scores, dim=-1)
        
        if self.scaled == True:
            nway = int((labels.shape[1]+1)/2)
            losses = self.loss_fct(log_scores, labels[:,:nway])
            weights = torch.stack([self.alpha/(1+torch.exp(labels[:,i]-labels[:,nway])) + 1 for i in range(nway + 1, labels.shape[1])], 1)

            return torch.mean(losses * weights)
     
        return torch.mean(self.loss_fct(log_scores, labels[:,:nway]))


class MultipleNegativesRankingLossCE(nn.Module):
    def __init__(self, model):
        super(MultipleNegativesRankingLossCE, self).__init__()
        self.model = model
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        reps = [self.model(sentence_feature)['cls'] for sentence_feature in sentence_features]
        scores  = torch.cat(reps, dim=1)
        #scores = torch.cat([reps[0] - reps[i] for i in range(1, len(reps))], dim=1)
        
        class_labels = torch.tensor([0] * scores.shape[0], dtype=torch.long, device=scores.device)  # Example a[i] should match with b[i]

        return self.cross_entropy_loss(scores, class_labels)

    def get_config_dict(self):
        return 


class MarginMSELossSpladeAdapt(nn.Module):
    """
    Compute the MSE loss between the |sim(Query, Pos) - sim(Query, Neg)| and |gold_sim(Q, Pos) - gold_sim(Query, Neg)|
    By default, sim() is the dot-product
    For more details, please refer to https://arxiv.org/abs/2010.02666
    """
    def __init__(self, model, similarity_fct = pairwise_dot_score, lambda_uni = 1e-2):
        """
        :param model: SentenceTransformerModel
        :param similarity_fct:  Which similarity function to use
        """
        super(MarginMSELossSpladeAdapt, self).__init__()
        self.model = model
        self.similarity_fct = similarity_fct
        self.loss_fct = nn.MSELoss()
        self.uni = UNIFORM()
        self.lambda_uni = lambda_uni

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        # sentence_features: query, positive passage, negative passage
        embeddings_query = self.model(sentence_features[0])['sentence_embedding'] 
        embeddings_pos = self.model[0](sentence_features[1])['sentence_embedding'] 
        embeddings_neg = self.model[0](sentence_features[2])['sentence_embedding'] 

        scores_pos = self.similarity_fct(embeddings_query, embeddings_pos)
        scores_neg = self.similarity_fct(embeddings_query, embeddings_neg)
        margin_pred = scores_pos - scores_neg

        overall_loss = self.loss_fct(margin_pred, labels)
        uni_d = self.uni(torch.nn.functional.normalize(embeddings_pos,dim=1))
        uni_q = self.uni(torch.nn.functional.normalize(embeddings_query,dim=1))

        uniform_dist = self.lambda_uni * (uni_q - uni_d) ** 2
        #print(f"marginmse: {overall_loss}, unimse: {uniform_dist}")
        overall_loss +=  uniform_dist
 
        return overall_loss


class DistillLossCQColBERT(nn.Module):
    
    def __init__(self, model, teacher = "ce", loss_type = 'kldiv'):
        super(DistillLossCQColBERT, self).__init__()
        self.model = model
        if loss_type == "kldiv":
            self.loss_fct = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)
        else:
            self.loss_fct = nn.MSELoss()
        self.loss_type = loss_type
        self.teacher = teacher
    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        q_result = self.model[0](sentence_features[0])
        q_token_reps = q_result['last_layer_embeddings']
        token_rep_query = torch.nn.functional.normalize(q_token_reps, p=2, dim=2)
        
        d_results_orig = [self.model[0](sentence_feature) for sentence_feature in sentence_features[1:]]
        d_results_cq = [self.model[1](sentence_feature, training = True) for sentence_feature in d_results_orig]
        
        d_token_reps_cq = [d_result['cq_embeddings'] for d_result in d_results_cq]
        d_masks = [d_result['attention_mask'] for d_result in d_results_cq]

        d_token_reps_orig = [d_result['last_layer_embeddings'] for d_result in d_results_orig]

        token_scores = []
        token_scores_label = []
        for idx in range(len(d_results_cq)):
            token_rep_cq = d_token_reps_cq[idx] * d_masks[idx].unsqueeze(-1)
            token_rep_cq = torch.nn.functional.normalize(token_rep_cq, p=2, dim=2)
            token_level_score_cq = token_rep_query @ token_rep_cq.permute(0,2,1)

            token_rep_label = d_token_reps_orig[idx] * d_masks[idx].unsqueeze(-1)
            token_rep_label = torch.nn.functional.normalize(token_rep_label, p=2, dim=2)
            token_level_score_label = token_rep_query @ token_rep_label.permute(0,2,1)

            iter_mask = ~d_masks[idx].unsqueeze(1).repeat(1, token_level_score_cq.shape[1], 1).bool()
            token_level_score_cq[iter_mask] = -9999
            token_level_score_label[iter_mask] = -9999
            token_scores.append(token_level_score_cq.max(2).values.sum(1))
            token_scores_label.append(token_level_score_label.max(2).values.sum(1))
        
        scores = torch.stack(token_scores, dim=1)
        p_scores = torch.nn.functional.softmax(scores, dim=-1)
        log_scores = torch.log(p_scores)
        if self.loss_type == 'kldiv':
            if self.teacher != "default": #cross encocer
                loss_vector = self.loss_fct(log_scores, labels)
            else:
                scores_label = torch.stack(token_scores_label, dim=1)
                p_scores_label = torch.nn.functional.softmax(scores_label, dim=-1)
                log_scores_label = torch.log(p_scores_label)
                loss_vector = self.loss_fct(log_scores, log_scores_label)
            return torch.mean(loss_vector)
        else:
            
            mmse_cp  = torch.cat([token_scores[0] - token_scores[i] for i in range(1, len(token_scores))], dim = 0) 
            mmse_label  = torch.cat([labels[:,0] - labels[:,i] for i in range(1, labels.shape[1])], dim = 0) 
            mmse_orig  = torch.cat([token_scores_label[0] - token_scores_label[i] for i in range(1, len(token_scores))], dim = 0) 
            if self.teacher != "default": #cross encocer
                return self.loss_fct(mmse_cp, mmse_label)
            else: # uncompressed
                return self.loss_fct(mmse_cp, mmse_orig)