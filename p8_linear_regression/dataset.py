import torch
from torch.utils.data import Dataset
import pandas as pd


class Dataset_My(Dataset):
    def __init__(self, path=r'../dataset/boston_housing_data.csv', target=['MEDV'], flag="train"):
        super(Dataset_My, self).__init__()
        data = pd.read_csv(path)
        # drop the nan value of the MEDA columns
        data = data.dropna(subset=[data.columns[-1]])
        # fill the nan value with 0
        data = data.fillna(0)

        self.l = len(data)
        self.data_y = torch.tensor(data[target].values)
        self.data_x = torch.tensor(data.drop(columns=target, axis=1).values)

    def __getitem__(self, index):
        return self.data_x[index], self.data_y[index]

    def __len__(self):
        return self.l

