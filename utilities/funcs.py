import os
from argparse import ArgumentParser
import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_arguments():
    parser = ArgumentParser()
    parser.add_argument("-d", "--dataset",
                        help="What dataset to train on", 
                        required=True, 
                        choices=["dna", "splice", "toy"])
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
    
    parser.add_argument("-hi", "--hindered_iters",
                        help="How many hindered iterations should happen to retune the classifier",
                        type=int,
                        default=10)
    parser.add_argument("-v", "--visualizer",
                    help="Whether to visualize the learning process",
                    type=str2bool,
                    default=False)
    return parser.parse_args()

def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)