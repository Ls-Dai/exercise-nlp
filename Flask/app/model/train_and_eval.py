import torch 
import torch.nn as nn 
import os 

from dataloader import TrainTestDataLoader
from model import FastText, Preprocessor


def train(data_source="./dataset/train.json", batch_size=1024, model=FastText(3000, 1500, 500, 10), preprocessor=Preprocessor(), criterion=nn.CrossEntropyLoss(), learning_rate=0.0001):
    dataloader = TrainTestDataLoader(json_file_path=data_source, batch_size=batch_size)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.00001)
    
    for i in range(100):
        train_epoch(i + 1, dataloader, preprocessor, model, criterion, optimizer)
        
    
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
    if not os.path.exists("./saved_models"):
        os.mkdir("./saved_models")
    torch.save(model.state_dict(), "./saved_models/fasttext.pt")
    
    
def test(data_source="./dataset/test.json", model=FastText(3000, 1500, 500, 10), saved_model_path="./saved_models/fasttext.pt", preprocessor=Preprocessor(), criterion=nn.CrossEntropyLoss()):
    dataloader = TrainTestDataLoader(json_file_path=data_source, batch_size=1)
    state_dict = torch.load(saved_model_path)
    model.load_state_dict(state_dict)
    model.eval()
    
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
     
            
def train_and_eval(data_source="./dataset/train.json", batch_size=1024, model=FastText(3000, 1500, 500, 10), preprocessor=Preprocessor(), criterion=nn.CrossEntropyLoss(), learning_rate=0.0001):
    dataloader = TrainTestDataLoader(json_file_path=data_source, batch_size=batch_size)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.00001)
    
    for i in range(20):
        train_epoch(i + 1, dataloader, preprocessor, model, criterion, optimizer)
        test()
    
    if not os.path.exists("./saved_models"):
        os.mkdir("./saved_models")
    torch.save(model.state_dict(), "./saved_models/fasttext.pt")
    
    
if __name__ == "__main__":
    train_and_eval()