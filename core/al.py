import logging
import acquisitions, datasets, core, utilities


class ActiveLearning:
    n_trials = 50
    
    def __init__(self, args):
        Dataclass = getattr(datasets, args.dataset.capitalize())
        self.budget = Dataclass.configs["budget"]
        self.random_seed = args.random_seed
        self.args = args
        Acqclass = getattr(acquisitions, args.algorithm.capitalize())
        self.pool = core.Pool(data=Dataclass.get_data_dict(), torch_seed=self.random_seed, args=args)

        self.hyper_path = f"temp/hypers/{args.dataset}/"
        self.hyper_filename = f"{str(args.random_seed)}_{args.val_share}_{args.n_initially_labeled}.pkl"
        self.best_hypers = utilities.retrieve_pkl(self.hyper_path + self.hyper_filename)
        self.clf = core.Learnable(pool=self.pool, 
                                  model_configs=self.best_hypers,
                                  random_seed=self.random_seed)
        self.acq = Acqclass(clf=self.clf, pool=self.pool, random_seed=self.random_seed)
        if Dataclass.visualize:
            self.visualizer = utilities.Visualize(self.pool, self.clf, self.acq, total_budget=self.budget)

        self.results_path = f"results/{args.dataset}/{args.algorithm}/il_{args.n_initially_labeled}/val_share_{args.val_share}/"
       
    def run(self):
        results = [["dataset", 
                   "algorithm", 
                   "random_seed",
                   "added", 
                   "test_loss", 
                   "test_acc", 
                   "val_loss", 
                   "val_acc", 
                   "train_loss",
                   "train_acc",
                   "len_ulb", 
                   "len_lb", 
                   "val_share",
                   "n_initially_labeled",
                   "iteration"]]
        abs_idx = None
        if self.best_hypers: # If the hypers were tuned beforehand
            train_perf, val_perf, test_perf = self.clf.train_model()
        else: # if there are no hypers found
            train_perf, val_perf, test_perf = self.clf.tune_model(n_trials=self.n_trials)
            hypers = self.clf.model_configs.copy()
            utilities.store_pkl(hypers, filename=self.hyper_filename, path=self.hyper_path)

        logging.warning(f'{abs_idx} {self.pool.get_len("all_labeled")} {self.pool.get_len("unlabeled")} {utilities.get_name(args=self.args)}\n{train_perf}\n{val_perf}\n{test_perf}')

        if hasattr(self, "visualizer"):
            draw_acq = self.acq.__class__.__name__ != "Cheating"
            self.visualizer.make_plots(self.args, 0, train_perf, val_perf, test_perf, self.results_path + "plots/", draw_acq=draw_acq)
        if self.budget <  self.pool.get_len("unlabeled"):
            for iteration in range(1, self.budget + 1):

                abs_idx, relative_idx = self.acq.query()
            
                self.pool.add_new_inst(abs_idx)

                train_perf, val_perf, test_perf = self.clf.tune_model(n_trials=self.n_trials)

                # LOGGINGS
                logging.warning(f'{abs_idx} {self.pool.get_len("all_labeled")} {self.pool.get_len("unlabeled")} {utilities.get_name(args=self.args)}\n{train_perf}\n{val_perf}\n{test_perf}')
                results.append(utilities.gather_results(self.args, abs_idx, test_perf, val_perf, train_perf, self.pool, iteration))
                utilities.store_csv(results, filename=str(self.args.random_seed).zfill(2), path=self.results_path)

                if hasattr(self, "visualizer") and self.acq.__class__.__name__ != "Cheating":
                    self.visualizer.make_plots(self.args, iteration, train_perf, val_perf, test_perf, self.results_path + "plots/", relative_idx)
        # if we train on the whole dataset
        elif self.args.n_initially_labeled == self.pool.get_len("total"):        
            results.append(utilities.gather_results(self.args, abs_idx, test_perf, val_perf, train_perf, self.pool, 0))
            utilities.store_csv(results, filename=str(self.args.random_seed).zfill(2), path=self.results_path)
        else:
            logging.warning(f'{"?"*70}Check your budget{"?"*70}')


        logging.warning(f'{"!"*70} DONE {"!"*70}')


            