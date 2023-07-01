from argparse import ArgumentParser


def get_arguments():
    parser = ArgumentParser()
    parser.add_argument("-d", "--dataset",
                        help="What dataset to train on", 
                        required=True, 
                        choices=["dna", "splice", "toy"])
    parser.add_argument("-a", "--algorithm",
                        help="What active learning algorithm to evaluate",  
                        required=True,
                        choices=["random",
                                 "cheating",
                                 "keychain",
                                 "bald", 
                                 "coreset",
                                 "entropy"])
    parser.add_argument("-r", "--random_seed",
                        help="What random seed to use",  
                        type=int,
                        default=42)
    return parser.parse_args()