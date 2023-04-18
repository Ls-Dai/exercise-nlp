import torch 
import torch.nn as nn 
import os 

from data import TrainTestDataLoader
from model import FastText, Preprocessor

def train():
    dataloader = TrainTestDataLoader(json_file_path="./dataset/train.json", batch_size=1024)
    preprocessor = Preprocessor()
    model = FastText(3000, 1500, 500, 10)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.00001)
    
    for i in range(100):
        train_epoch(i + 1, dataloader, preprocessor, model, criterion, optimizer)
    
    return 
    
def train_epoch(epoch, dataloader, preprocessor, model, criterion, optimizer):
    model.train()
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
    print("epoch: {} \t loss: {}".format(epoch, total_loss))
    if not os.path.exists("./models"):
        os.mkdir("./models")
    torch.save(model.state_dict(), "./models/fasttext.pt")
    
    
def test():
    dataloader = TrainTestDataLoader(json_file_path="./dataset/test.json", batch_size=1024)
    preprocessor = Preprocessor()
    model = FastText(3000, 1500, 500, 10)
    state_dict = torch.load("./models/fasttext.pt")
    model.load_state_dict(state_dict)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        total_loss = 0
        correct = 0
        total = 0
        for texts, brands in dataloader:
            features = preprocessor.get_vocab_indices(texts)
            y_true = preprocessor.get_labels_one_hot(brands)
            y_pred = model(features)
            loss = criterion(y_pred, y_true)
            total_loss += loss.item()
            correct += torch.sum(torch.argmax(y_pred, dim=1).eq(torch.argmax(y_true, dim=1))).item()
            total += y_true.shape[0]
        print("test loss:", total_loss)
        print("test acc:", correct / total)
            
def train_and_eval():
    TrainTestDataLoader(json_file_path="./dataset/exercise_data.json", dump=True)
    dataloader = TrainTestDataLoader(json_file_path="./dataset/train.json", batch_size=1024)
    preprocessor = Preprocessor()
    model = FastText(3000, 1500, 500, 10)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for i in range(100):
        train_epoch(i + 1, dataloader, preprocessor, model, criterion, optimizer)
        test()
    
    if not os.path.exists("./models"):
        os.mkdir("./models")
    torch.save(model.state_dict(), "./models/fasttext.pt")
    
if __name__ == "__main__":
    train_and_eval()