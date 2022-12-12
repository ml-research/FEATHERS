from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
import torch
import pandas as pd
import numpy as np

class FraudDetectionData(Dataset):

    def __init__(self, file, train) -> None:
        super().__init__()
        data = pd.read_csv(file + 'ccFraud.csv')
        data = data.drop(columns=['custID'])
        np.random.seed(0) # ensure deterministic behavior
        train_inds = np.random.uniform(0, len(data), int(0.7 * len(len(data)))) # 70% of data is treated as train data
        val_inds = np.argwhere(np.arange(0, len(data)) != train_inds).reshape(1, -1)[0] # 30% of data is treated as validation data
        y = data['fraudRisk'].to_numpy()
        X = data.drop(columns=['fraudRisk'])
        y = y[train_inds] if train else y[val_inds]
        X = X[train_inds] if train else X[val_inds]
        scaler = StandardScaler()
        X = scaler.fit_transform(X.to_numpy())
        self.y = torch.from_numpy(y)
        self.X = torch.from_numpy(X)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        x, y = self.X[index], self.y[index]
        return x, y