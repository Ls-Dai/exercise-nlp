# exercise-nlp

## Frameworks && Models

### Backend Service

**Flask**: To build a service which could process POST request, I used **Flask** as my backend service framework. I did not implement any templates inside. Just have a function which would process the incoming POST request.

### NLP Model

For classification tasks, considering its latency in production, we should firstly consider simpler models rather than directly using BERT models (which are best but too complex). Normally CNN and FastText are common to use. I chose **FastText** as my NLP model.

**PyTorch** is a good Deep Learning framework to use.

## Files Structures

```
.
├── Flask
│   ├── app
│   │   ├── __init__.py
│   │   ├── model
│   │   │   ├── data.py
│   │   │   ├── dataset
│   │   │   │   ├── exercise_data.json
│   │   │   │   ├── labels.txt
│   │   │   │   ├── test.json
│   │   │   │   ├── train.json
│   │   │   │   ├── train_test_split.py
│   │   │   │   └── vocab.txt
│   │   │   ├── __init__.py
│   │   │   ├── main.py
│   │   │   ├── model.py
│   │   │   ├── models
│   │   │   │   └── fasttext.pt
│   │   │   ├── __pycache__
│   │   │   │   ├── data.cpython-38.pyc
│   │   │   │   └── model.cpython-38.pyc
│   │   │   ├── saved_models
│   │   │   │   └── fasttext.pt
│   │   │   └── train_and_eval.py
│   │   ├── templates
│   │   │   └── index.html
│   │   └── views.py
│   └── run.py
├── LICENSE
├── README.md
└── test.sh
```

`Flask/app/` is the app directory.

`Flask/run.py` is the start script for this app.

`Flask/app/model/` is the model directory, it is about model training, evaluation. 

`Flask/app/dataset/` is the dataset directory for model training and evaluation. 

`Flask/app/dataset/saved_models` contains the saved models.

`Flask/app/views.py` contains the server processing code, which would read the saved models from `Flask/app/dataset/saved_models` and do inference. 

`Flask/app/model/utils.py` provides lots of functions handling vocab build, labels build and train-test split.

`Flask/app/model/main.py` provides entrance to run scripts.

## Model Performance 

Currently I split `train: test (val) = 7: 3` randomly. The accuracy over test dataset is `70.9%`

The dataset is dirty. I took use of those words whose length is between 2 to 29 to build my model.

## Model Performance 

Currently I split `train: test (val) = 7: 3` randomly. The accuracy over test dataset is `70.9%`

The dataset is dirty. I took use of those words whose length is between 2 to 29 to build my model.

## How to Use This Codebase 

Install the dependency 

```
pip install -r requirements.txt
```

Then notice that we should at least have `./Flask/app/model/dataset/exercise_data.json`

Inside `./Flask/app/model/`, run 

```
python main.py -m build
```

This will generate training and testing dataset. 

Afterwards, run

```
python main.py -m train_and_eval
```

It will train and eval the model. The model file would be saved to `./Flask/app/model/saved_models`.

Instead of `train_and_eval`, alternatively, you can run,

```
python main.py -m train
```

```
python main.py -m test
```

Those would provide seperate training and testing processes. For example, after training, you can see the model performace (metrics) by,

```
python main.py -m test
```

It shows,

```
(base) lisen@pineapple:/home/lisen/exercise-nlp/Flask/app/model$ python main.py -m test
test loss: 2592.2102720737457
test acc: 0.7037542662116041
```

There are more arg options in the future to manipulate the training. 

Then we are trying to start the service in `./Flask/` by running,

```
python run.py
```
```
(base) lisen@pineapple:/home/lisen/exercise-nlp/Flask$ python run.py 
 * Serving Flask app 'app'
 * Debug mode: on
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:55555
 * Running on http://128.59.9.215:55555
Press CTRL+C to quit
 * Restarting with watchdog (inotify)
 * Debugger is active!
 * Debugger PIN: 545-816-954
```

Now the service is started. Keep it on. Start **another terminal**, in `./`, run

```
bash test.sh
```

You should see the result from the service right now.

```
(base) lisen@pineapple:/home/lisen/exercise-nlp$ bash test.sh 
[
  {
    "brand": "netflix",
    "probablity": 0.825648307800293
  }
]
```



## TODO 

Current codebase still has space to get better. There are two important things I need to do. 

1. Try a way to clean the database. For example, the non-sense words, the adjoined words. 
2. Robust function in my service. For example, input check. 
3. Make a more robust and readable data pipeline for model training and testing.



## Present my plan to

a. Deploy model to production

I'll deploy model as a micro-service on AWS or Azure, creating an REST APIs that allows other software systems to interact with it. Also, once your model is deployed, it is important to monitor it to ensure that it is performing as expected, for example, performance metrics, logging, and alerting systems.

b. Maintain

I should have a monitoring system which monitor the performance of the model and addressing any issues that arise. Also, I need to have some scripts that update the model as new data becomes available or as the business requirements change.

c. Continuously train

Setting up a data pipeline that feeds new data into the model on a regular basis. Therefore we can update the model architecture or hyperparameters as needed. Retraining the model on a regular basis, such as daily or weekly, which takes new data to train for fewer epochs to finetune the current model. Then push this model to production. This is Online Learning. We need to develop a particular CI / CD for it. 


