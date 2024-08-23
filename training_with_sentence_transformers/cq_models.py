
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from typing import Union, Tuple, List, Iterable, Dict
import os
import json
import math
from torch.nn.parameter import Parameter


# Quick utility function to sample from the Gumbel-distribution: -Log(-Log(Uniform)), eps to avoid numerical errors
def sample_gumbel(input_size, eps=1e-20):
    unif = torch.rand(input_size)

    return -torch.log(-torch.log(unif + eps) + eps)

class Encoder(nn.Module):
    '''
    The encoder takes embedding dimension, M, and K (from MxK coding scheme introduced in paper)
    as parameters. From a word's baseline embedding,
    it outputs Gumbel-softmax or one-hot encoding vectors d_w, reshaped for the decoder.
    In the original paper, the hidden_layer was fixed at M * K/2.

    Input shape: BATCH_SIZE X EMBEDDING_DIM
    Output shape: BATCH_SIZE X M X K X 1, Code (K X 1)

    '''
    def __init__(self, emb_size,M, K, hidden_size= None):
        super(Encoder, self).__init__()
        # input is cat[non-static, static]
        self.emb_size = emb_size
        self.K = K
        self.M = M
        # If not otherwise specified, use hidden_size indicated by paper
        if not hidden_size:
            #hidden_size = int(M * K / 2)
            hidden_size = 128
        
        # This layer maps from the hidden layer to BATCH_SIZE X M K
        self.get_residual = nn.Linear(self.emb_size * 2, hidden_size)
        
        self.alpha_w = nn.Linear(int(M * K // 2),int(M * K))
        self.h_w = nn.Linear(hidden_size, int(M * K // 2))
            
    def forward(self, x, tau=1, eps=1e-20, training=True):
        # We apply hidden layer projection from original embedding
        #print("last layer emb shape", x.shape) ([16, 175, 128])
        hw = F.tanh(self.get_residual(x))
        hw = F.tanh(self.h_w(hw))
        
        # We apply second projection and softplus activation
        alpha = F.softplus(self.alpha_w(hw))
        # This rearranges alpha to be more intuitively BATCH_SIZE X M X K
        alpha = alpha.view(-1, self.M, self.K)
        # Take the log of all elements
        log_alpha = torch.log(alpha)
        if training:
            # We apply Gumbel-softmax trick to get code vectors d_w
            d_w = F.softmax((log_alpha + sample_gumbel(log_alpha.size()).to(log_alpha.device)) / tau, dim=-1)
            # Find argmax of all d_w vectors
            _, ind = d_w.max(dim=-1)
        if not training:
            _, ind = alpha.max(dim=-1)
            # Allows us when not training to convert soft vector to a hard, binarized one-hot encoding vector
            d_w = torch.zeros_like(alpha).scatter_(-1, ind.unsqueeze(2), 1.0)
        # d_w is now BATCH x M x K x 1
        d_w = d_w.unsqueeze(-1)
        return d_w, ind

class Decoder(nn.Module):
    '''
    The decoder receives d_w as input from the encoder, and outputs the embedding generated by this code.
    It stores a set of source dictionaries, represented by A, and computes the proper embedding from a summation
    of M matrix-vector products.

    INPUT SHAPE: BATCH_SIZE X M X K X 1
    OUTPUT SHAPE: BATCH_SIZE X EMBEDDING_DIM
    '''
    def __init__(self, M, K, output_size, emb_size, position=False):
        super(Decoder, self).__init__()
        self.output_size = output_size
        self.K = K
        self.M = M
        self.emb_size = emb_size

        # Contains source dictionaries for computing embedding given codes
        self.A = Source_Dictionary(M, output_size // M, K)
        self.final = nn.Linear(self.emb_size, self.emb_size)
        self.position = position
        if position:
            # This linear layer maps to latent hidden representation
            self.c_e = nn.Linear(self.emb_size  + output_size + 768, self.emb_size)
        else:
            self.c_e = nn.Linear(self.emb_size + output_size, self.emb_size)

    # Following the formula in the paper, performs multiplication and summation over the M matrix-vector products
    def forward(self, d_w, static):
        output = self.A(d_w) 
        #print("decoder output shape", output.shape) # 128 x M * output_size/ M 
        #print("self.outputsize", self.output_size)
        codeapprox = output.reshape(output.shape[0], self.output_size)
        #add layer 2#
        output = F.tanh(self.c_e(torch.cat([codeapprox, static], dim = 1)))
        output = self.final(output)
        ######
        #output = self.c_e(torch.cat([codeapprox, static], dim = 1))
        return output

class Source_Dictionary(nn.Module):
    r"""I basically modified the source code for the nn.Linear() class
        Removed bias, and the weights are of dimension M X EMBEDDING_SIZE X K
        INPUT: BATCH_SIZE X M X K X 1

        OUTPUT:BATCH_SIZE X K X 1
    """

    def __init__(self, M, emb_size, K):
        super(Source_Dictionary, self).__init__()
        # The weight of the dictionary is the set of M dictionaries of size EMB X K
        self.weight = Parameter(torch.Tensor(M, emb_size, K))
        self.reset_parameters()

    # Initialize parameters of Source_Dictionary
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    # Operation necessary for proper batch matrix multiplication
    def forward(self, input):
        result = torch.matmul(self.weight, input)
        return result.squeeze(-1)



class CodeLearner(nn.Module):
    """Feed-forward function with  activiation function.
    """
    def __init__(self, emb_size, M, K, hidden_size = None):
        super(CodeLearner, self).__init__()
        '''
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.activation_function = activation_function
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        if init_weight is not None:
            self.linear.weight = nn.Parameter(init_weight)

        if init_bias is not None:
            self.linear.bias = nn.Parameter(init_bias)
        '''
        if not hidden_size:
            #hidden_size = int(M * K / 2)
            hidden_size = 128
        # Initialize encoder and decoder components
        self.encoder = Encoder(emb_size, M, K, hidden_size)
        self.decoder = Decoder(M = M, K = K, output_size = hidden_size, emb_size = emb_size)
        self.emb_size = emb_size
        self.M = M
        self.K = K 
        self.hidden_size = hidden_size


    def forward(self, features: Dict[str, Tensor], tau=1, eps=1e-20, training=True):
        #print("features['cq_embedding']", features['cq_embedding'].shape 16，dlen, 256
        cq_reshape = features['cq_embedding'].reshape(-1, features['cq_embedding'].shape[-1])
        d_w, _ = self.encoder(cq_reshape, tau, eps, training)
        comp_emb = self.decoder(d_w, cq_reshape[:,self.emb_size:])
        features.update({'cq_embeddings': comp_emb.reshape(features['cq_embedding'].shape[0], features['cq_embedding'].shape[1], -1)})
        return features

    def get_sentence_embedding_dimension(self) -> int:
        return self.hidden_size

    def get_config_dict(self):
        return {'M': self.M, 'K': self.K, 'hidden_size': self.hidden_size, 'emb_size': self.emb_size}

    def save(self, output_path):
        with open(os.path.join(output_path, 'config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut)

        torch.save(self.state_dict(), os.path.join(output_path, 'pytorch_model.bin'))
        
    def __repr__(self):
        return "CQ({})".format(self.get_config_dict())

    @staticmethod
    def load(input_path):
        with open(os.path.join(input_path, 'config.json')) as fIn:
            config = json.load(fIn)

        
        model = CodeLearner(**config)
        model.load_state_dict(torch.load(os.path.join(input_path, 'pytorch_model.bin'), map_location=torch.device('cpu')))
        return model


