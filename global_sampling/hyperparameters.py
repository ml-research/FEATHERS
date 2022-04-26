import numpy as np

class Hyperparameters:

    _instance = None

    def __init__(self, nr_configs) -> None:
        if Hyperparameters._instance is not None:
            raise RuntimeError("Hyperparameters is a singleton, use instance()")
        self.sample_hyperparams = lambda: {
            'learning_rate': np.sort(10 ** (np.random.uniform(-4, 0)))
        }
        self.hyperparams = [self.sample_hyperparams() for _ in range(nr_configs)]
        
    @classmethod
    def instance(cls, nr_configs):
        if Hyperparameters._instance is None:
            Hyperparameters._instance = Hyperparameters(nr_configs)
        return Hyperparameters._instance

    def __getitem__(self, idx):
        return self.hyperparams[idx]

    def __len__(self):
        return len(self.hyperparams)