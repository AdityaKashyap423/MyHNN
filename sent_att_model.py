#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 15:51:47 2020

@author: aditya
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import matrix_mul, element_wise_mul


class SentAttModel(nn.Module):
    def __init__(self,hidden_size = 100):
        super(SentAttModel,self).__init__()
        # self.attention_model = nn.Transformer(d_model=768, nhead=8, num_encoder_layers=2, num_decoder_layers=2,dim_feedforward=256)


        self.sent_weight = nn.Parameter(torch.Tensor(2*hidden_size,2*hidden_size))
        self.sent_bias = nn.Parameter(torch.Tensor(1, 2 * hidden_size))
        self.context_weight = nn.Parameter(torch.Tensor(2 * hidden_size, 1))
        self.embed_size = 768
        
        self.gru = nn.GRU(self.embed_size, hidden_size, bidirectional=True)
        self._create_weights(mean=0.0, std=0.05)
        
    def _create_weights(self, mean=0.0, std=0.05):

        self.sent_weight.data.normal_(mean, std)
        self.context_weight.data.normal_(mean, std)
        self.sent_bias.data.normal_(mean, std)
        
    def forward(self, x, hidden_state):
        # output = self.attention_model(x,hidden_state)
        # return output 

        f_output, h_output = self.gru(x.float(),hidden_state)
        output = matrix_mul(f_output, self.sent_weight, self.sent_bias)        
        output = matrix_mul(output, self.context_weight).permute(1,0)
        output = F.softmax(output)
        output = element_wise_mul(f_output,output.permute(1,0))
        return output,h_output




    
