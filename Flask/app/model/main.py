import argparse 
from typing import * 

from utils import train_and_test_split, build_vocab, build_labels
from train_and_eval import train, test, train_and_eval


def main(args):
    if args.mode == "build":
        train_and_test_split()
        build_vocab()
        build_labels()
    if args.mode == "train":
        train()
    if args.mode == "test":
        test()
    if args.mode == "train_and_eval":
        train_and_eval()
    return
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", type=str, default="build", help="Task options")
    parser.add_argument("-e", "--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("-b", "--batch_size", type=int, default=64, help="Batch Size")
    parser.add_argument("-p", "--dataset_path", type=str, default="./dataset/train.json", help="Path to dataset") 
        
    args = parser.parse_args()
    main(args) 