import torch 
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader
from torchtext.data import Field

import numpy as np 
from itertools import chain
import json 
import os 


class TrainDataSet(Dataset):
    def __init__(self, json_file_path="./dataset/exercise_data.json", *args, **kwargs):
        super(Dataset, self).__init__()
        self.__data_raw = []
        with open(json_file_path) as f:
            self.__data_raw.extend(json.load(f))
        
        sorted_labels = self.build_labels()
        vocab = self.build_vocab()
        self.__labels_dict = {key: value for key, value in zip(sorted_labels, range(len(sorted_labels)))}
        
    def __getitem__(self, index):
        data_pair_dict = self.__data_raw[index]
        return data_pair_dict["raw_text"], data_pair_dict["brand"]
        
    def __len__(self):
        return len(self.__data_raw)
    
    def __one_hot(self, label):
        one_hot_label = np.zeros(len(self.__labels_dict))
        one_hot_label[self.__labels_dict[label]] = 1
        return one_hot_label
    
    def build_vocab(self):
        sorted_words = sorted(set(chain(*map(lambda x: x["raw_text"].strip().split(), self.__data_raw))))
        if not os.path.exists("./dataset"):
            os.mkdir("./dataset")
        with open("./dataset/vocab.txt", "w+") as f:
            f.write("\n".join(sorted_words))
        return sorted_words
    
    def build_labels(self):
        sorted_labels = sorted(set(map(lambda x: x["brand"], self.__data_raw)))
        if not os.path.exists("./dataset"):
            os.mkdir("./dataset")
        with open("./dataset/labels.txt", "w+") as f:
            f.write("\n".join(sorted_labels))
        return sorted_labels
        
         
    def get_labels_dict(self):
        return self.__labels_dict
        

def TrainDataLoader(json_file_path="./dataset/exercise_data.json", batch_size=1, shuffle=True, *args, **kwargs):
    train_dataset = TrainDataSet(json_file_path, *args, **kwargs)
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    

if __name__ == "__main__":
    train_dataset = TrainDataSet()