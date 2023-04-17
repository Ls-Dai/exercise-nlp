import torch 
import torch.nn as nn 
import os 

from data import TrainTestDataLoader
from model import FastText, Preprocessor

def train():
    dataloader = TrainTestDataLoader(json_file_path="./dataset/train.json", batch_size=1024)
    preprocessor = Preprocessor()
    model = FastText(3000, 1000, 100, 10)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for _ in range(10):
        total_loss = 0
        for texts, brands in dataloader:
            optimizer.zero_grad()
            features = preprocessor.get_vocab_indices(texts)
            y_true = preprocessor.get_labels_one_hot(brands)
            y_pred = model(features)
            loss = criterion(y_pred, y_true)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        print(total_loss)
    
    if not os.path.exists("./models"):
        os.mkdir("./models")
    torch.save(model.state_dict(), "./models/fasttext.pt")
    
def test():
    dataloader = TrainTestDataLoader(json_file_path="./dataset/test.json", batch_size=1024)
    preprocessor = Preprocessor()
    model = FastText(3000, 1000, 100, 10)
    state_dict = torch.load("./models/fasttext.pt")
    model.load_state_dict(state_dict)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        total_loss = 0
        for texts, brands in dataloader:
            features = preprocessor.get_vocab_indices(texts)
            y_true = preprocessor.get_labels_one_hot(brands)
            y_pred = model(features)
            loss = criterion(y_pred, y_true)
            total_loss += loss.item()
        print(total_loss)
            
def train_and_eval():
    TrainTestDataLoader(json_file_path="./dataset/exercise_data.json", dump=True)
    train()
    test()
    
if __name__ == "__main__":
    test()