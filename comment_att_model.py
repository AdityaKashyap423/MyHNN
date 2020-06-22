#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 16:28:34 2020

@author: aditya
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import matrix_mul, element_wise_mul

class CommentAttModel(nn.Module):
    
    def __init__(self,sent_hidden_size = 100,comment_hidden_size = 100, num_classes = 2):
        super(CommentAttModel,self).__init__()
        self.comment_weight = nn.Parameter(torch.Tensor(2 * comment_hidden_size, 2 * comment_hidden_size))
        self.comment_bias = nn.Parameter(torch.Tensor(1, 2 * comment_hidden_size))
        self.context_weight = nn.Parameter(torch.Tensor(2 * comment_hidden_size, 1))
        
        self.gru = nn.GRU(2 * sent_hidden_size, comment_hidden_size, bidirectional=True)
        self.fc = nn.Linear(2 * comment_hidden_size, num_classes)
        self._create_weights(mean=0.0, std=0.05)

        # self.attention_model = nn.Transformer(d_model=768, nhead=8, num_encoder_layers=2, num_decoder_layers=2,dim_feedforward=256)
        # self.fc = nn.Linear(768, num_classes)

    def _create_weights(self, mean=0.0, std=0.05):
        self.comment_weight.data.normal_(mean, std)
        self.context_weight.data.normal_(mean, std)
        self.comment_bias.data.normal_(mean, std)
    
    def forward(self,x,hidden_state):
        
        f_output, h_output = self.gru(x, hidden_state)
        output = matrix_mul(f_output, self.comment_weight, self.comment_bias)
        output = matrix_mul(output, self.context_weight).permute(1, 0)
        output = F.softmax(output)
        output = element_wise_mul(f_output, output.permute(1, 0)).squeeze(0)
        output = self.fc(output)

        return output, h_output

        # output = self.attention_model(x,hidden_state)
        # output = self.fc(output).squeeze(0)
        # return output 
    

        
        
        
        
        
