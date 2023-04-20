import torch 
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader

import json 


class TrainTestDataSet(Dataset):
    def __init__(self, json_file_path="./dataset/train.json", *args, **kwargs):
        super(Dataset, self).__init__()
        self.__data_raw = []
        with open(json_file_path) as f:
            self.__data_raw.extend(json.load(f))
        
    def __getitem__(self, index):
        data_pair_dict = self.__data_raw[index]
        return data_pair_dict["raw_text"], data_pair_dict["brand"]
        
    def __len__(self):
        return len(self.__data_raw)
    

def TrainTestDataLoader(json_file_path="./dataset/train.json", batch_size=1, shuffle=True, *args, **kwargs):
    dataset = TrainTestDataSet(json_file_path, *args, **kwargs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)



if __name__ == "__main__":
    pass 