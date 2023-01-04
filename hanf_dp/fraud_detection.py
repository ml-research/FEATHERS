from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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
        subsamples = np.random.randint(0, len(data), size=int(0.01*len(data)))
        data = data.iloc[subsamples, :]
        y = data['fraudRisk'].to_numpy()
        X = data.drop(columns=['fraudRisk']).to_numpy()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        if train:
            X, y = X_train, y_train
        else:
            X, y = X_test, y_test
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        self.y = torch.from_numpy(y).long()
        self.X = torch.from_numpy(X).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        x, y = self.X[index], self.y[index]
        return x, y