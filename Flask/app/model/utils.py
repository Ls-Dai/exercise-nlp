import random
import json
from itertools import chain


def train_and_test_split(split_rate=0.6,
                         raw_path="./dataset/exercise_data.json",
                         train_target="./dataset/train.json",
                         test_target="./dataset/test.json",
                         ):
    data_raw = []
    with open(raw_path) as f:
        data_raw.extend(json.load(f))

    random.shuffle(data_raw)
    length = len(data_raw)

    with open(train_target, "w") as f:
        json.dump(data_raw[0: int(split_rate * length)], f)

    with open(test_target, "w") as f:
        json.dump(data_raw[int(split_rate * length):], f)
    return data_raw[0: int(split_rate * length)], data_raw[int(split_rate * length):]


def build_vocab(source_path: str="./dataset/exercise_data.json", target_path:str="./dataset/vocab.txt", min_word_len: int=0, max_word_len: int=30):
    data_raw = []
    with open(source_path) as f:
        data_raw.extend(json.load(f))
    sorted_words = sorted(set(chain(*filter(lambda x: min_word_len < len(x) < max_word_len, map(lambda x: x["raw_text"].strip().split(), data_raw)))))
    with open(target_path, "w+") as f:
        f.write("\n".join(sorted_words))
    return sorted_words

        
def build_labels(source_path: str="./dataset/exercise_data.json", target_path:str="./dataset/labels.txt"):
    data_raw = []
    with open(source_path) as f:
        data_raw.extend(json.load(f))
    sorted_labels = sorted(set(map(lambda x: x["brand"], data_raw)))
    with open(target_path, "w+") as f:
        f.write("\n".join(sorted_labels))
    return sorted_labels


def load_vocab(path: str="./dataset/exercise_data.json"):
    with open(path, "r") as f:
        vocab = f.read().split("\n")
    return vocab


def load_labels(path: str="./app/model/dataset/labels.txt"):
    with open(path, "r") as f:
        labels = f.read().split("\n")
    return labels