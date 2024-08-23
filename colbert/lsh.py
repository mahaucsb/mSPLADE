import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

class LshModule(torch.nn.Module):
    def __init__(self, num_bits = 1024, mu = 0, sigma = 0.1,requires_grad=False):
        super().__init__()
        self.r = nn.Parameter(torch.randn(num_bits, 128) * sigma + mu, requires_grad=requires_grad)  #num_bits, 768
        self.d = num_bits
        self.pi = nn.Parameter(torch.Tensor([np.pi]),requires_grad=requires_grad)
        
    def LSH(self, emb):
        lsh = torch.matmul(emb, self.r.T)
        return (lsh>0).float() #batch, qlen, num_bits
    
    def LSH_cos(self,u, v):
        #return torch.cos(self.pi / self.d * torch.count_no nzero((u != v)))
        print((u!=v).sum())
        return torch.count_nonzero((u != v))
    
    def forward(self, emb1, emb2):
        #emb1: BATCH*QLEN*768, emb2: BATCH*DLEN*768
        emb1_en = self.LSH(emb1) 
        emb2_en = self.LSH(emb2)
        
        result = torch.cos(self.pi / self.d * torch.cdist(emb1_en, emb2_en, p=0))
        return result#torch.cat(result,-1)#.permute(2,0,1)    

    
class LshSimmatModule(torch.nn.Module):

    def __init__(self, padding=-1,num_bits=1024):
        super().__init__()
        self.padding = padding
        self._hamming_index_loaded = None
        self._hamming_index = None
        self.LSH = LshModule(num_bits=num_bits)
        
    def forward(self, query_embed, doc_embed, query_tok, doc_tok):
        simmat = []

        for a_emb, b_emb in zip(query_embed, doc_embed):
            BAT, A, B = a_emb.shape[0], a_emb.shape[1], b_emb.shape[1]
            # embeddings -- cosine similarity matrix
            # a_emb: BAT * 20 * 768

            sim = self.LSH.forward(a_emb,b_emb)
            
            # nullify padding (indicated by -1 by default)
            nul = torch.zeros_like(sim)
            sim = torch.where(query_tok.reshape(BAT, A, 1).expand(BAT, A, B) == self.padding, nul, sim)
            sim = torch.where(doc_tok.reshape(BAT, 1, B).expand(BAT, A, B) == self.padding, nul, sim)

            simmat.append(sim)

        return torch.stack(simmat, dim=1)
        

