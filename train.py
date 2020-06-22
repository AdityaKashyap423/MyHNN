#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 16:43:47 2020

@author: aditya
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import get_max_lengths, get_evaluation,get_evaluation_identity
from dataset import MyDataSet
from HAN import HierarchicalAttentionNetwork
from tensorboardX import SummaryWriter
import argparse
import shutil
import numpy as np

# Import wandb libraries
import wandb




parser = argparse.ArgumentParser(
    """Implementation of the model described in the paper: Hierarchical Attention Networks for Document Classification""")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--num_epoches", type=int, default=10)
parser.add_argument("--dropout", type=float, default = 0.2)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--sent_hidden_size", type=int, default=100) #768
parser.add_argument("--comment_hidden_size", type=int, default=100)
parser.add_argument("--es_min_delta", type=float, default=0.0,
                    help="Early stopping's parameter: minimum change loss to qualify as an improvement")
parser.add_argument("--es_patience", type=int, default=5,
                    help="Early stopping's parameter: number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique.")
parser.add_argument("--test_interval", type=int, default=1, help="Number of epochs between testing phases")
parser.add_argument("--log_path", type=str, default="tensorboard/han_voc")
parser.add_argument("--saved_path", type=str, default="trained_models")
opt = parser.parse_args()

max_sentences = 10 # max number of sentences per comment
max_comments = 50 # max number of comments per author
num_classes = 22



# Set hyperparameters, which can be overwritten with a W&B Sweep
hyperparameter_defaults = dict(
  dropout = opt.dropout,
  sent_hidden_layer_size = opt.sent_hidden_size,
  comment_hiddent_layer_size = opt.comment_hidden_size,
  learn_rate = opt.lr,
  momentum = opt.momentum,
  epochs = opt.num_epoches
)

# Initialize wandb
wandb.init(config=hyperparameter_defaults)
config = wandb.config

# the_identity = "socialist"   


# train_dataset_path = "/nlp/data/kashyap/Masters_Thesis/Data/DeepLearning_TrainingData/Self_Identification/MultiClassData/Vectors/gender/train/"
# test_dataset_path = "/nlp/data/kashyap/Masters_Thesis/Data/DeepLearning_TrainingData/Self_Identification/MultiClassData/Vectors/gender/dev/"

train_dataset_path = "/nlp/data/kashyap/Masters_Thesis/Data/DeepLearning_TrainingData/Self_Identification/MultiClassData/Vectors/train/"
test_dataset_path = "/nlp/data/kashyap/Masters_Thesis/Data/DeepLearning_TrainingData/Self_Identification/MultiClassData/Vectors/dev/"


# train_dataset_path = "/nlp/data/kashyap/Masters_Thesis/Data/DeepLearning_TrainingData/Self_Identification/MultiClassData/Vectors/individual_training_data/" + the_identity + "/train/"
# test_dataset_path = "/nlp/data/kashyap/Masters_Thesis/Data/DeepLearning_TrainingData/Self_Identification/MultiClassData/Vectors/individual_training_data/" + the_identity + "/dev/"

if torch.cuda.is_available():
    torch.cuda.manual_seed(123)
else:
    torch.manual_seed(123)
output_file = open(opt.saved_path + os.sep + "logs.txt", "w")
output_file.write("Model's parameters: {}".format(vars(opt)))
training_params = {"batch_size": opt.batch_size,
                   "shuffle": False,
                   "drop_last": True}
test_params = {"batch_size": opt.batch_size,
               "shuffle": False,
               "drop_last": False}





training_set = MyDataSet(train_dataset_path,max_comments,max_sentences)
training_generator = DataLoader(training_set, **training_params)
test_set = MyDataSet(test_dataset_path,max_comments,max_sentences)
test_generator = DataLoader(test_set, **test_params)

model = HierarchicalAttentionNetwork(opt.sent_hidden_size,opt.comment_hidden_size,opt.batch_size,num_classes,max_sentences,max_comments)
wandb.watch(model)



if torch.cuda.is_available():
    model = model.cuda()
    # model = nn.DataParallel(model)

# criterion = nn.CrossEntropyLoss()
criterion = nn.BCEWithLogitsLoss()
# criterion = nn.MultiLabelSoftMarginLoss()
optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr, momentum=opt.momentum)
# optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr)
best_loss = 1e5
best_epoch = 0
model.train()
num_iter_per_epoch = len(training_generator)
for epoch in range(opt.num_epoches):
    for iter, (feature, label) in enumerate(training_generator):
        if torch.cuda.is_available():
            feature = feature.type('torch.FloatTensor').cuda()
            label = label.cuda() # batch size * num of labels
        available_label_ind = label!=-1


        optimizer.zero_grad()
        model._init_hidden_state()
        predictions = model(feature)
        loss = criterion(predictions[available_label_ind], label[available_label_ind]) 
        # loss = criterion(predictions, label)
        loss.backward()
        optimizer.step()
        with torch.no_grad(): #(Nest the remaining lines inside)
        # Use tensor operations in get_evaluation to stay in the CUDA space
            training_metrics = get_evaluation(label.cpu().numpy(), predictions.cpu().numpy(), list_metrics=["accuracy"])
            print("Epoch: {}/{}, Iteration: {}/{}, Lr: {}, Loss: {}, Accuracy: {}".format(
                epoch + 1,
                opt.num_epoches,
                iter + 1,
                num_iter_per_epoch,
                optimizer.param_groups[0]['lr'],
                loss, training_metrics["accuracy"]))




        if iter % 20 == 0:
            model.eval()
            loss_ls = []
            te_label_ls = []
            te_pred_ls = []
            for te_feature, te_label in test_generator:
                num_sample = len(te_label)
                if torch.cuda.is_available():
                    te_feature = te_feature.type('torch.FloatTensor').cuda()
                    te_label = te_label.cuda()
                with torch.no_grad():
                    model._init_hidden_state(num_sample)
                    te_predictions = model(te_feature)
                available_test_label_ind = te_label != -1
                te_loss = criterion(te_predictions[available_test_label_ind], te_label[available_test_label_ind])
                # te_loss = criterion(te_predictions, te_label)
                loss_ls.append(te_loss * num_sample)
                te_label_ls.extend(te_label.clone().cpu()) # don't need to clone 
                te_pred_ls.append(te_predictions.clone().cpu()) # don't need to clone 
            te_loss = sum(loss_ls) / test_set.__len__()
            te_pred = torch.cat(te_pred_ls, 0)
            # te_label = np.array(te_label_ls)
            te_label = np.array(torch.cat([a.reshape((1,num_classes)) for a in te_label_ls],axis=0))
            test_metrics = get_evaluation(te_label, te_pred.numpy(), list_metrics=["accuracy", "confusion_matrix"])

            wandb.log({"loss": te_loss,"accuracy":test_metrics["accuracy"]})
            log_dict = {}
            for key in test_metrics["identity_accuracy"].keys():
                log_dict[key] = test_metrics["identity_accuracy"][key]
            wandb.log(log_dict)
            model.train()

 

 
    if te_loss + opt.es_min_delta < best_loss:
        best_loss = te_loss
        best_epoch = epoch
        torch.save(model, opt.saved_path + os.sep + "whole_model_han_2") 

    # Early stopping
    if epoch - best_epoch > opt.es_patience > 0:
        print("Stop training at epoch {}. The lowest loss achieved is {}".format(epoch, te_loss))
        break 


