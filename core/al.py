import time #DELETE
import logging
import acquisitions, datasets, core, utilities


class ActiveLearning:
    #DEBUG
    whole_arch_ntrials = 100
    finetune_ntrials = 40
    finetune_params = ['weight_decay', 'lr']
    
    def __init__(self, args):
        Dataclass = getattr(datasets, args.dataset.capitalize())
        self.budget = Dataclass.configs["budget"]
        self.random_seed = args.random_seed
        self.args = args
        Acqclass = getattr(acquisitions, args.algorithm.capitalize())
        self.pool = core.Pool(data=Dataclass.get_data_dict(), random_seed=self.random_seed)

        self.hyper_path = "results/aux/hypers/"
        self.hyper_filename = utilities.get_name(self.args, include_alg=False) + "_best_hypers.pkl"
        self.best_hypers = utilities.retrieve_pkl(self.hyper_path + self.hyper_filename)
        self.clf = core.Learnable(pool=self.pool, 
                                  model_configs=self.best_hypers,
                                  random_seed=self.random_seed)
        self.acq = Acqclass(clf=self.clf, pool=self.pool, random_seed=self.random_seed)
        self.retuner = utilities.EarlyStopper(patience=args.hindered_iters)
        if args.visualizer:
            self.visualizer = utilities.Visualize(self.pool, self.clf, self.acq)
       
    def run(self):
        results = []
        abs_idx = None
        if self.best_hypers: # If the hypers were tuned beforehand
            train_perf, val_perf, test_perf = self.clf.train_model()
        else: # if there are no hypers found
            train_perf, val_perf, test_perf = self.clf.tune_model(n_trials=self.whole_arch_ntrials)
            hypers = self.clf.model_configs.copy()
            utilities.store_file(hypers, filename=self.hyper_filename, path=self.hyper_path)

        logging.warning(f'{train_perf} {val_perf} {test_perf} {abs_idx} {self.pool.get_len("labeled")} {self.pool.get_len("unlabeled")} {utilities.get_name(args=self.args)}')

        for idx in range(self.budget):

            abs_idx, relative_idx = self.acq.query()
            if hasattr(self, "visualizer"):
                self.visualizer.make_plots(relative_idx, self.args, idx, train_perf, val_perf, test_perf)

            self.pool.add_new_inst(abs_idx)

            train_perf, val_perf, test_perf = self.clf.tune_model(tunable_hypers=self.finetune_params,n_trials=self.finetune_ntrials)

            # LOGGINGS
            logging.warning(f'{train_perf} {val_perf} {test_perf} {abs_idx} {self.pool.get_len("labeled")} {self.pool.get_len("unlabeled")} {utilities.get_name(args=self.args)}')
            results.append(utilities.gather_results(self.args, abs_idx, test_perf, val_perf, self.pool, idx))
            utilities.store_file(results, utilities.get_name(self.args))