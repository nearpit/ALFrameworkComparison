import acquisitions, datasets, core, utilities


class ActiveLearning:
    def __init__(self, args):
        Dataclass = getattr(datasets, args.dataset.capitalize())
        self.budget = Dataclass.configs["budget"]
        Acqclass = getattr(acquisitions, args.algorithm.capitalize())

        self.pool = core.Pool(data=self.get_data_dict(Dataclass), random_seed=args.random_seed)
        self.clf = core.Learnable(pool=self.pool, random_seed=args.random_seed)
        self.acq = Acqclass(clf=self.clf, pool=self.pool, random_seed=args.random_seed)
        self.retuner = utilities.EarlyStopper(patience=args.hindered_iters)
        self.last_cand = -1
        self.seq_to_tune = [self.clf]


    def get_data_dict(self, Dataclass):
        return {"train": Dataclass(split_name="train"), 
                "val": Dataclass(split_name="val"),
                "test": Dataclass(split_name="test")}

    def run(self):
        last_labeled = -1
        for idx in range(self.budget):
            self.clf.train_model()
            val_perf, test_perf = self.clf.eval_model("val"), self.clf.eval_model("test")
            print(val_perf, test_perf, last_labeled, self.pool.get_len("labeled"), self.pool.get_len("unlabeled"))
            last_labeled = self.acq.query()
            self.pool.add_new_inst(last_labeled)

            if self.retuner.early_stop(val_perf[0]): # if training is hindered
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!NEW UPSTREAM HYPERS WERE REQUESTED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                for model in self.seq_to_tune:
                    model.tune_model()