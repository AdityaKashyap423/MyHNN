#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 18:06:51 2020

@author: aditya
"""

import pandas as pd
from tqdm import tqdm
from datetime import datetime
import nltk
import numpy as np
from torch.utils.data.dataset import Dataset
from sentence_transformers import SentenceTransformer
import pickle
from collections import Counter

np.random.seed(42)


start = datetime.now()




class MyDataSet(Dataset):

    def __init__(self,dataset_path,max_comments,max_sentences):
        super(MyDataSet, self).__init__()
        self.dataset_path = dataset_path # Directory in which the data is stored
        self.max_comments = max_comments # Maximum number of comments per author
        self.max_sentences = max_sentences # Maximum number of sentences per comment
        self.data_distribution = self.get_data_distribution()
        self.instance_file_mapping = self.build_mapping()
        self.file_instance_mapping = self.build_reverse_mapping()
        self.current_file = ""
        self.current_file_data = ""


    def get_data_distribution(self):
        with open(self.dataset_path + "info.txt","r") as f:
            data = f.read().split("\n")[:-1]
        processed_data = []
        for line in data:
            processed_data.append((line.split("\t")[0],int(line.split("\t")[1])))
        return processed_data

    def build_mapping(self):
        total_instance_number = 0
        mapping = {}
        for element in self.data_distribution:
            filename = element[0]
            num_train_samples = element[1]
            for i in range(num_train_samples):
                mapping[total_instance_number] = filename
                total_instance_number += 1
        return mapping

    def build_reverse_mapping(self):
        total_instance_number = 0
        mapping = {}
        for element in self.data_distribution:
            filename = element[0]
            num_train_samples = element[1]
            mapping[filename] = [total_instance_number + a for a in range(num_train_samples)]
            total_instance_number += num_train_samples
        return mapping



    def __len__(self):
        # return min(sum([a[1] for a in self.data_distribution]),600)
        return sum([a[1] for a in self.data_distribution])


    def __getitem__(self,index):
        required_file = self.instance_file_mapping[index]
        
        if self.current_file != required_file:
            with open(required_file,"rb") as f:
                self.current_file_data = pickle.load(f)
                self.current_file = required_file


        file_index = index - min(self.file_instance_mapping[self.current_file])

        current_instance,current_label = self.current_file_data[file_index]
        current_instance = current_instance[:self.max_comments,:self.max_sentences,:]

        # current_label[current_label==-1] = 0 # Added



        # return (current_instance,current_label)
        return (current_instance,get_identity_label(current_label,"man"))
 
    


def get_identity_label(vector,identity):
    all_identities = ["writer","woman","socialist","republican","photographer","parent","musician","mom","man","liberal","leftist","graphic designer","gamer","dog person","democrat","dad","conservative","cat person","athlete","atheist","artist","agnostic"]
    index_ident = {b:a for a,b in enumerate(all_identities)}

    # return vector[index_ident[identity]]

    for i in range(len(all_identities)):
        if i != index_ident[identity]:
            vector[i] = -1

    return vector

def get_text_labels(vector):
    all_identities = ["writer","woman","socialist","republican","photographer","parent","musician","mom","man","liberal","leftist","graphic designer","gamer","dog person","democrat","dad","conservative","cat person","athlete","atheist","artist","agnostic"]
    index_ident = {a:b for a,b in enumerate(all_identities)}

    ident_list = []

    for i in range(len(vector)):
        if vector[i] == 1:
            ident_list.append(index_ident[i])
        elif vector[i] == 0:
            ident_list.append("not " + index_ident[i])

    return ident_list



def print_ident_freq(ident_list):
    all_identities = ["writer","woman","socialist","republican","photographer","parent","musician","mom","man","liberal","leftist","graphic designer","gamer","dog person","democrat","dad","conservative","cat person","athlete","atheist","artist","agnostic"]
    ident_freq = Counter(ident_list)

    for identity in all_identities:
        print(identity + " = " + str(ident_freq[identity]) +  "/" + str(ident_freq["not " + identity]))


if __name__ == "__main__":
    
    from torch.utils.data import DataLoader
    max_sentences = 10 # max number of sentences per comment 
    max_comments = 50 # max number of comments per author
    dataset_path = "/nlp/data/kashyap/Masters_Thesis/Data/DeepLearning_TrainingData/Self_Identification/MultiClassData/Vectors/dev/"
    # dataset_path = "/nlp/data/kashyap/Masters_Thesis/Data/DeepLearning_TrainingData/Self_Identification/MultiClassData/Vectors/gender/train/"
    # dataset_path = "/nlp/data/kashyap/Masters_Thesis/Data/DeepLearning_TrainingData/Self_Identification/MultiClassData/Vectors/individual_training_data/" + the_identity + "/train/"


    
    dataset = MyDataSet(dataset_path,max_comments,max_sentences)
    
    training_params = {"batch_size": 1,
                   "shuffle": False,
                   "drop_last": True}
    
    training_generator = DataLoader(dataset, **training_params)



    ident_list = []    
    for (feature,label) in tqdm(training_generator):
        # ident_list += get_text_labels(label[0])
        print(feature.shape,label.shape)
    #     ident_list += [label] 

    # print(Counter(ident_list))

    # print_ident_freq(ident_list)


        
        
        
        

