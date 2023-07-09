import logging
import acquisitions, datasets, core, utilities


class ActiveLearning:
    def __init__(self, args):
        Dataclass = getattr(datasets, args.dataset.capitalize())
        self.budget = Dataclass.configs["budget"]
        self.random_seed = args.random_seed
        self.args = args
        Acqclass = getattr(acquisitions, args.algorithm.capitalize())

        self.pool = core.Pool(data=self.get_data_dict(Dataclass), random_seed=args.random_seed)
        self.clf = core.Learnable(pool=self.pool, random_seed=args.random_seed)
        self.acq = Acqclass(clf=self.clf, pool=self.pool, random_seed=args.random_seed)
        self.retuner = utilities.EarlyStopper(patience=args.hindered_iters)
        self.last_cand = -1
        self.val_perf = None
        self.test_perf = None


    def get_data_dict(self, Dataclass):
        return {"train": Dataclass(split_name="train"), 
                "val": Dataclass(split_name="val"),
                "test": Dataclass(split_name="test")}

    def run(self):
        retuned = []
        results = []
        for idx in range(self.budget + 1):
            self.val_perf, self.test_perf = self.clf.eval_model("val"), self.clf.eval_model("test")
            logging.log(f'{self.val_perf} {self.test_perf} {self.last_cand} {self.pool.get_len("labeled")} {self.pool.get_len("unlabeled")} {self.args.dataset} {self.args.algorithm} {self.args.random_seed}')


            if self.retuner.early_stop(self.val_perf[0]): # if training is hindered
                logging.log("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!NEW UPSTREAM HYPERS WERE REQUESTED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                self.clf.tune_model()
                retuned.append(idx)
                utilities.store_results(retuned, self.get_name(self.args) + "_retuned", path="results/extra")

             # LOGGINGS
            results.append(self.gather_results(idx))
            utilities.store_results(results, self.get_name(self.args))

            self.last_cand = self.acq.query()
            self.pool.add_new_inst(self.last_cand)

            

            self.clf.reset_model()
            self.clf.train_model()

           



    def gather_results(self, iter):
        return {
            "dataset": self.args.dataset,
            "algorithm": self.args.algorithm,
            "random_seed": self.args.random_seed,
            "last_cand": self.last_cand,
            "test_loss": self.test_perf[0],
            "test_acc": self.test_perf[1]["MulticlassAccuracy"],
            "val_loss": self.val_perf[0],
            "val_acc": self.val_perf[1]["MulticlassAccuracy"],
            "len_ulb": self.pool.get_len("unlabeled"),
            "len_lb": self.pool.get_len("labeled"), 
            "iter": iter
        }
    
    def get_name(self, args):
        return f"{args.dataset}_{args.algorithm}_{args.random_seed}"
