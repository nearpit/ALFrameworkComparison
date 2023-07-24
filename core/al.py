import logging
import acquisitions, datasets, core, utilities

class ActiveLearning:
    n_trials = 2 # DEBUG
    
    def __init__(self, args):
        Dataclass = getattr(datasets, args.dataset.capitalize())
        self.budget = Dataclass.configs["budget"]
        self.random_seed = args.random_seed
        self.args = args
        Acqclass = getattr(acquisitions, args.algorithm.capitalize())
        self.pool = core.Pool(data=Dataclass.get_data_dict(), args=args)

        self.hyper_path = f"temp/hypers/{args.dataset}/"
        self.hyper_filename = f"{str(args.random_seed)}_{args.val_share}_{args.n_initially_labeled}_{args.online}.pkl"
        self.best_hypers = utilities.retrieve_pkl(self.hyper_path + self.hyper_filename)
        self.clf = core.Learnable(pool=self.pool, 
                                  model_configs=self.best_hypers,
                                  random_seed=self.random_seed)
        self.acq = Acqclass(clf=self.clf, pool=self.pool, random_seed=self.random_seed, budget=self.budget)
        if Dataclass.visualize:
            self.visualizer = utilities.Visualize(self.pool, self.clf, self.acq, total_budget=self.budget)

        self.results_path = f"results/{args.dataset}/{args.algorithm}/{args.n_initially_labeled}_il/val_share_{args.val_share}/online_{args.online}/"

        self.results = [["dataset", 
                        "algorithm", 
                        "random_seed",
                        "idx_added", 
                        "features_added",
                        "target_added",
                        "test_loss", 
                        "test_acc", 
                        "val_loss", 
                        "val_acc", 
                        "train_loss",
                        "train_acc",
                        "len_ulb", 
                        "len_lb", 
                        "val_share",
                        "online",
                        "n_initially_labeled",
                        "iteration"]]
       
    def run(self):
        abs_idx = None
        train_perf, val_perf, test_perf = self.train_first_hypers(self.args.online)
        self.show_intermediate_results(abs_idx, train_perf, val_perf, test_perf)
        
        if self.budget <  self.pool.get_len("unlabeled"):
            for iteration in range(0, self.budget):

                abs_idx = self.acq.query()
                self.visualize(iteration, train_perf, val_perf, test_perf, abs_idx)

                self.pool.add_new_inst(abs_idx)
                if self.args.online:
                    train_perf, val_perf, test_perf = self.clf.tune_model(n_trials=self.n_trials, online=self.args.online)
                else:
                    train_perf, val_perf, test_perf = self.clf.train_model()

                # LOGGINGS
                self.show_intermediate_results(abs_idx, train_perf, val_perf, test_perf)
                features_added, target_added = self.pool[abs_idx]
                self.append_store_results(abs_idx, features_added, target_added, train_perf, val_perf, test_perf, iteration)

        # if we train on the whole dataset
        elif self.args.n_initially_labeled == -1:        
            self.append_store_results(abs_idx, None, train_perf, val_perf, test_perf, 0)

        else:
            logging.warning(f'{"?"*70}Check your budget{"?"*70}')
        logging.warning(f'{"!"*70} DONE {"!"*70}')
        self.visualize("final", train_perf, val_perf, test_perf, abs_idx)


    def train_first_hypers(self, online):
        if self.best_hypers: # If the hypers were tuned beforehand by other algorithm workload
            train_perf, val_perf, test_perf = self.clf.train_model()
        else: # if there are no hypers found
            train_perf, val_perf, test_perf = self.clf.tune_model(n_trials=self.n_trials, online=online)
            hypers = self.clf.model_configs.copy()
            utilities.store_pkl(hypers, filename=self.hyper_filename, path=self.hyper_path)

        return train_perf, val_perf, test_perf
    
    def show_intermediate_results(self, abs_idx, train_perf, val_perf, test_perf):
        logging.warning(f'{abs_idx} {self.pool.get_len("all_labeled")} {self.pool.get_len("unlabeled")} {utilities.get_name(args=self.args)}\n{train_perf}\n{val_perf}\n{test_perf}')

    def visualize(self, iteration, train_perf, val_perf, test_perf, chosen_idx):
        if hasattr(self, "visualizer"):
            self.visualizer.make_plots(self.args, iteration, train_perf, val_perf, test_perf, self.results_path + "plots/", chosen_idx)

    def append_store_results(self, abs_idx, features_added, target_added, train_perf, val_perf, test_perf, iteration):
        self.results.append(utilities.gather_results(self.args, abs_idx, features_added, target_added, test_perf, val_perf, train_perf, self.pool, iteration))
        utilities.store_csv(self.results, filename=str(self.args.random_seed).zfill(2), path=self.results_path)