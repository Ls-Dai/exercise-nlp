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

@app.route('/brand', methods=["POST"])
def process():
    payload = request.get_json()
    text = payload['text']
    # features = preprocessor.get_vocab_indices(text)
    # print(features)
    # y_pred = model(features)
    # print(y_pred)
    return [{"brand": "netflix", "probablity": 0.82}]