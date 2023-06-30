from argparse import ArgumentParser


def get_arguments():
    parser = ArgumentParser()
    parser.add_argument("-d", "--dataset", help="What dataset to train on", required=True, choices=["dna", "splice", "toy"])
    parser.add_argument("-a", "--algorithm", help="What active learning algorithm to evaluate", choices=["random", 
                                                                                                         "cheating",
                                                                                                         "keychain",
                                                                                                         "bald", 
                                                                                                         "coreset",
                                                                                                         "entropy"])
    return parser.parse_args()