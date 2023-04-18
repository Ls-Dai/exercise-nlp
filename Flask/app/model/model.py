import torch
import torch.nn as nn 
import torch.nn.functional as F

import random 
import numpy as np 

class FastText(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_classes, *args, **kwargs) -> None:
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