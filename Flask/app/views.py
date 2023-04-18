from app import app 
from flask import render_template
from flask import request, url_for, redirect

from app.model import FastText, Preprocessor
import torch 

model = FastText(3000, 1500, 500, 10)
preprocessor = Preprocessor(vocab_path="./app/model/dataset/vocab.txt", labels_path="./app/model/dataset/labels.txt")
state_dict = torch.load("./app/model/saved_models/fasttext.pt")
model.load_state_dict(state_dict)
model.eval()
with open("./app/model/dataset/labels.txt") as f:
    labels = f.read().split("\n")

@app.route('/brand', methods=["POST"])
def process():
    payload = request.get_json()
    text = payload['text']
    features = preprocessor.get_vocab_indices([text])
    y_pred = model(features)
    index = torch.argmax(y_pred, dim=1).item()
    return [{"brand": labels[index], "probablity": y_pred[0][index].item()}]