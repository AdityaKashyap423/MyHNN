#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 15:15:47 2020

@author: aditya
"""

import torch
import torch.nn as nn
from dataset import MyDataSet
from sent_att_model import SentAttModel
from comment_att_model import CommentAttModel

class HierarchicalAttentionNetwork(nn.Module):
    
    def __init__(self,sentence_hidden_size,comment_hidden_size,batch_size,num_classes,max_sentences,max_comments):
        super(HierarchicalAttentionNetwork,self).__init__()
        self.sentence_hidden_size = sentence_hidden_size
        self.comment_hidden_size = comment_hidden_size
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.max_sentences = max_sentences
        self.max_comments = max_comments
        self.sent_att_net = SentAttModel(self.sentence_hidden_size)
        self.comment_att_net = CommentAttModel(self.sentence_hidden_size,self.comment_hidden_size,num_classes)
        self._init_hidden_state()
        
        
    def _init_hidden_state(self,last_batch_size = None):
        
        if last_batch_size:
            batch_size = last_batch_size
        else:
            batch_size = self.batch_size
        
        self.sentence_hidden_state = torch.zeros(2,batch_size,self.sentence_hidden_size)
        self.comment_hidden_state = torch.zeros(2,batch_size,self.comment_hidden_size)

        # self.sentence_hidden_state = torch.zeros(self.max_sentences,batch_size,768)
        # self.comment_hidden_state = torch.zeros(1,batch_size,768)
        
        if torch.cuda.is_available():
            self.sentence_hidden_state = self.sentence_hidden_state.cuda()
            self.comment_hidden_state = self.comment_hidden_state.cuda()
            
    
    def forward(self,x): 
        
        output_list = []
        
        x = x.permute(1,0,2,3)
        
        for instance in x:
            output, self.sentence_hidden_state = self.sent_att_net(instance.permute(1,0,2),self.sentence_hidden_state)
            # output = self.sent_att_net(instance.permute(1,0,2),self.sentence_hidden_state)
            output_list.append(output)
    
        output = torch.cat(output_list, 0)

        output, self.comment_hidden_state = self.comment_att_net(output, self.comment_hidden_state)
        # output = self.comment_att_net(output, self.comment_hidden_state)

        return output 
            
            
if __name__ == "__main__":
    
    from torch.utils.data import DataLoader
    
    max_sentences = 10 # max number of sentences per comment
    max_comments = 10 # max number of comments per author
    dataset_path = "/nlp/data/kashyap/Masters_Thesis/Data/DeepLearning_TrainingData/Self_Identification/MultiClassData/Vectors/dev/" 

    
    # dataset = MyDataSet(dataset_path,max_comments,max_sentences)
    
    training_params = {"batch_size": 32,
                   "shuffle": True,
                   "drop_last": False}
    
    # training_generator = DataLoader(dataset, **training_params)
    model = HierarchicalAttentionNetwork(50,50,32,22,10,50)
    
    src = torch.rand((32,50,10,768))
    if torch.cuda.is_available():
        model = model.cuda()
        src = src.cuda()


    output = model(src)

    print(output.shape)
    # for (feature,label) in training_generator:
    #     model._init_hidden_state()
    #     predictions = model(feature)
        
        

        
        
        
        
        
        
        
    

