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
        self.visualizer = utilities.Visualize(self.pool, self.clf)
       
    def run(self):
        results = []
        last_cand = None
        val_perf, test_perf = self.clf.eval_model("val"), self.clf.eval_model("test")
        logging.warning(f'{val_perf} {test_perf} {last_cand} {self.pool.get_len("labeled")} {self.pool.get_len("unlabeled")} {utilities.get_name(args=self.args)}')
        self.visualizer.plot_decision_boundary()


        for idx in range(self.budget):

            last_cand, relative_idx, all_scores = self.acq.query()
            self.visualizer.plot_chosen(all_scores, relative_idx)

            self.pool.add_new_inst(last_cand)

            self.clf.reset_model()
            self.clf.train_model()
            val_perf, test_perf = self.clf.eval_model("val"), self.clf.eval_model("test")

            

            # LOGGINGS
            logging.warning(f'{val_perf} {test_perf} {last_cand} {self.pool.get_len("labeled")} {self.pool.get_len("unlabeled")} {utilities.get_name(args=self.args)}')
            results.append(utilities.gather_results(self.args, last_cand, test_perf, val_perf, self.pool, idx))
            utilities.store_results(results, utilities.get_name(self.args))