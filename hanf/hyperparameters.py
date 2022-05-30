import numpy as np
import pandas as pd

class Hyperparameters:

    def __init__(self, nr_configs) -> None:
        self.sample_hyperparams = lambda: {
            'learning_rate': 10 ** (np.random.uniform(-4, -1)),
            'weight_decay': 10.0 ** np.random.uniform(low=-5.0, high=-1.0),
            'momentum': np.random.uniform(low=0.0, high=1.0),
            #'dropout': np.random.uniform(low=0, high=0.2),
            'arch_learning_rate': 10 ** (np.random.uniform(-5, -2)), # for search-phase
            'arch_weight_decay': 10.0 ** np.random.uniform(low=-5.0, high=-1.0), # for search-phase
        }
        #self.sample_hyperparams = lambda: {
        #    'learning_rate': 0.025,
        #    'weight_decay': 3e-4,
        #    'momentum': 0.9,
        #    'dropout': 0.2,
        #    #'arch_learning_rate': 4e-3,
        #    #'arch_weight_decay': 1e-4,
        #}
        self.hyperparams = [self.sample_hyperparams() for _ in range(nr_configs)]

    def read_from_csv(self, file):
        df = pd.read_csv(file, index_col=0)
        arr = []
        for _, row in df.iterrows():
            arr.append(row.to_dict())
        self.hyperparams = arr

    def save(self, file):
        df = pd.DataFrame.from_dict(self.to_dict())
        df.to_csv(file)

    def __getitem__(self, idx):
        return self.hyperparams[idx]

    def __len__(self):
        return len(self.hyperparams)

    def to_dict(self):
        hyperparam_dict = {}
        for config in self.hyperparams:
            for key, val in config.items():
                if key in hyperparam_dict.keys():
                    hyperparam_dict[key].append(val)
                else:
                    hyperparam_dict[key] = [val]
        return hyperparam_dict