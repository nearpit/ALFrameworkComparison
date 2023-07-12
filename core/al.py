import time #DELETE
import logging
import acquisitions, datasets, core, utilities


class ActiveLearning:
    def __init__(self, args):
        Dataclass = getattr(datasets, args.dataset.capitalize())
        self.budget = Dataclass.configs["budget"]
        self.random_seed = args.random_seed
        self.args = args
        Acqclass = getattr(acquisitions, args.algorithm.capitalize())
        hyper_path = "results/aux/" + utilities.get_name(self.args, include_alg=False) + "_best_hypers.pkl"
        best_hypers = utilities.retrieve_pkl(hyper_path)
        self.pool = core.Pool(data=Dataclass.get_data_dict(), random_seed=self.random_seed)
        self.clf = core.Learnable(pool=self.pool, random_seed=self.random_seed, model_configs=best_hypers)
        self.acq = Acqclass(clf=self.clf, pool=self.pool, random_seed=self.random_seed)
        self.retuner = utilities.EarlyStopper(patience=args.hindered_iters)
        if args.visualizer:
            self.visualizer = utilities.Visualize(self.pool, self.clf, self.acq)
       
    def run(self):
        results = []
        abs_idx = None
        val_perf, test_perf = self.clf.eval_model("val"), self.clf.eval_model("test")
        logging.warning(f'{val_perf} {test_perf} {abs_idx} {self.pool.get_len("labeled")} {self.pool.get_len("unlabeled")} {utilities.get_name(args=self.args)}')


        # start_time = time.time() #DELETE
        for idx in range(self.budget):

            abs_idx, relative_idx = self.acq.query()
            if hasattr(self, "visualizer"):
                self.visualizer.make_plots(relative_idx, self.args, idx, val_perf, test_perf)

            self.pool.add_new_inst(abs_idx)

            self.clf.reset_model()
            self.clf.train_model()
            val_perf, test_perf = self.clf.eval_model("val"), self.clf.eval_model("test")

            
            # logging.warning(f'{"!"*60}{time.time() - start_time}{"!"*60}') #DELETE

            # LOGGINGS
            logging.warning(f'{val_perf} {test_perf} {abs_idx} {self.pool.get_len("labeled")} {self.pool.get_len("unlabeled")} {utilities.get_name(args=self.args)}')
            results.append(utilities.gather_results(self.args, abs_idx, test_perf, val_perf, self.pool, idx))
            utilities.store_results(results, utilities.get_name(self.args))