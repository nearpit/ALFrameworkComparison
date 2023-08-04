from argparse import ArgumentParser, BooleanOptionalAction
    
def get_arguments():
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(help="sub-command help")

    parser.add_argument("-d", "--dataset",
                        help="What dataset to train on", 
                        required=True)
    parser.add_argument("-a", "--algorithm",
                        help="What active learning algorithm to evaluate",  
                        choices=["random",
                                 "bald", 
                                 "coreset",
                                 "entropy"])
    parser.add_argument("-r", "--random_seed",
                        help="What random seed to use",  
                        type=int,
                        default=42)
    
    parser.add_argument("-vs", "--val_share",
                    help="What share of unviolated labeled instances to use for validation",
                    type=float,
                    default=0.25)
    
    parser.add_argument("-n_il", "--n_initially_labeled",
                    help="What number of labeled instances to start with",
                    type=int,
                    default=20)
    
    parser.add_argument("-hpo_mode",
                        choices=["constant", "online"])
    parser.add_argument("-s", "--split",
                        choices=["whole", "initial", "static", "dynamic"])
    
    args = parser.parse_args()
   
    if args.hpo_mode == 'constant' and args.split not in ['whole', 'initial']:
        parser.error("When -hpo_mode is 'constant', --split must be 'whole' or 'initial'")
    elif args.hpo_mode == 'online' and args.split not in ['static', 'dynamic']:
        parser.error("When -hpo_mode is 'online', --split must be 'static' or 'dynamic'")

    return args