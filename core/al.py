import logging
import acquisitions, datasets, core, utilities


class ActiveLearning:
    #DEBUG
    whole_arch_ntrials = 50
    finetune_ntrials = 25
    retuner_patience = 50
    finetune_params = {"weight_decay", "lr"}
    
    def __init__(self, args):
        Dataclass = getattr(datasets, args.dataset.capitalize())
        self.budget = Dataclass.configs["budget"]
        self.random_seed = args.random_seed
        self.args = args
        Acqclass = getattr(acquisitions, args.algorithm.capitalize())
        self.pool = core.Pool(data=Dataclass.get_data_dict(), torch_seed=self.random_seed)

        self.hyper_path = f"temp/hypers/{args.dataset}/"
        self.hyper_filename = str(args.random_seed) + ".pkl"
        self.best_hypers = utilities.retrieve_pkl(self.hyper_path + self.hyper_filename)
        self.clf = core.Learnable(pool=self.pool, 
                                  model_configs=self.best_hypers,
                                  random_seed=self.random_seed)
        self.acq = Acqclass(clf=self.clf, pool=self.pool, random_seed=self.random_seed)
        if args.visualizer:
            self.visualizer = utilities.Visualize(self.pool, self.clf, self.acq)

        self.retuner = utilities.EarlyStopper(self.retuner_patience)
       
    def run(self):
        results = []
        abs_idx = None
        if self.best_hypers: # If the hypers were tuned beforehand
            train_perf, val_perf, test_perf = self.clf.train_model()
        else: # if there are no hypers found
            train_perf, val_perf, test_perf = self.clf.tune_model(n_trials=self.whole_arch_ntrials, use_kfold=True)
            hypers = self.clf.model_configs.copy()
            utilities.store_file(hypers, filename=self.hyper_filename, path=self.hyper_path)

        logging.warning(f'{abs_idx} {self.pool.get_len("labeled")} {self.pool.get_len("unlabeled")} {utilities.get_name(args=self.args)}\n{train_perf}\n{val_perf}\n{test_perf}')

        for idx in range(self.budget):

            abs_idx, relative_idx = self.acq.query()
            if hasattr(self, "visualizer"):
                self.visualizer.make_plots(relative_idx, self.args, idx, train_perf, val_perf, test_perf)

            self.pool.add_new_inst(abs_idx)
            if self.retuner.early_stop(val_perf[0]):
                train_perf, val_perf, test_perf = self.clf.tune_model(n_trials=self.whole_arch_ntrials)
                logging.warning(f'{"!"*1000}')

            else:
                train_perf, val_perf, test_perf = self.clf.tune_model(tunable_hypers=self.finetune_params, n_trials=self.finetune_ntrials)

            # LOGGINGS
            logging.warning(f'{abs_idx} {self.pool.get_len("labeled")} {self.pool.get_len("unlabeled")} {utilities.get_name(args=self.args)}\n{train_perf}\n{val_perf}\n{test_perf}')
            results.append(utilities.gather_results(self.args, abs_idx, test_perf, val_perf, self.pool, idx))
            utilities.store_file(results, utilities.get_name(self.args))