import torch
import torch.nn as nn 
import torch.nn.functional as F

import random 
import numpy as np 

try:
    from app.model.utils import load_vocab, load_labels
except:
    from utils import load_vocab, load_labels


class FastText(nn.Module):
    def __init__(self, vocab_size=1000, embed_size=1000, hidden_size=100, num_classes=10, *args, **kwargs) -> None:
        super(FastText, self).__init__(*args, **kwargs)

        self.vocab_size = vocab_size
        self.embed_size = embed_size
        
        self.embed_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        self.feedforward = nn.Sequential(
            nn.Linear(embed_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, num_classes),
        )
        self.activation = nn.Softmax(dim=1)
    
    def forward(self, tokens):
        x = self.embed_layer(tokens)
        x = torch.mean(x, dim=1)
        x = self.feedforward(x)
        x = self.activation(x)
        return x
    
    
class Preprocessor:
    def __init__(self, vocab_path="./dataset/vocab.txt", labels_path="./dataset/labels.txt") -> None:
        self.vocab = load_vocab(vocab_path)
        self.vocab_dict = {key: value for key, value in zip(self.vocab, range(1, len(self.vocab) + 1))}
            
        self.labels = load_labels(labels_path)
        self.labels_dict = {key: value for key, value in zip(self.labels, range(len(self.labels)))}
        
    def get_vocab_indices(self, texts, length=2000):
        tensors = []
        for text in texts:
            words = text.split()
            while len(words) < length:
                words.append("")
            words = random.sample(words, k=length)
            tensors.append(torch.tensor([self.vocab_dict.get(word, 0) for word in words], dtype=torch.long))
        return torch.stack(tensors)
        
    def get_labels_one_hot(self, labels):
        tensors = []
        for label in labels:
            tensors.append(self.__one_hot(self.labels_dict[label]))
        return torch.stack(tensors)
    
    def __one_hot(self, label):
        one_hot_label = torch.zeros(len(self.labels_dict))
        one_hot_label[label] = 1
        return one_hot_label
        

if __name__ == "__main__":
    pass 