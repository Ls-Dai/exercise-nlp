from app import app 
from flask import render_template
from flask import request, url_for, redirect

from model import FastText


@app.route('/brand', methods=["POST"])
def process():
    return [{"brand": "netflix", "probablity": 0.82}]