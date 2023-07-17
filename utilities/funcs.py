import os
from argparse import ArgumentParser
import argparse


def get_arguments():
    parser = ArgumentParser()
    parser.add_argument("-d", "--dataset",
                        help="What dataset to train on", 
                        required=True)
    parser.add_argument("-a", "--algorithm",
                        help="What active learning algorithm to evaluate",  
                        choices=["random",
                                 "cheating",
                                 "keychain",
                                 "bald", 
                                 "coreset",
                                 "entropy",
                                 "margin"])
    parser.add_argument("-r", "--random_seed",
                        help="What random seed to use",  
                        type=int,
                        default=42)
    
    parser.add_argument("-vs", "--val_share",
                    help="What share of labeled instances to use for validation",
                    type=float,
                    default=0.5)
    return parser.parse_args()

def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)