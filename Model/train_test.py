import torch 
import torch.nn as nn 

from data import TrainDataLoader
from model import FastText, Preprocessor

def train():
    dataloader = TrainDataLoader(json_file_path="./dataset/exercise_data.json", batch_size=10)
    preprocessor = Preprocessor()
    model = FastText(700000, 1000, 100, 10)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    for texts, brands in dataloader:
        optimizer.zero_grad()
        features = preprocessor.get_vocab_indices(texts)
        y_true = preprocessor.get_labels_one_hot(brands)
        y_pred = model(features)
        loss = criterion(y_pred, y_true)
        print(loss)
        loss.backward()
        optimizer.step()
        
if __name__ == "__main__":
    train()