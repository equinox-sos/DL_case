import torch
from torch.utils.data import Dataset
import pandas as pd


class Dataset_My(Dataset):
    def __init__(self, path=r'../dataset/boston_housing_data/train_dataset.csv', target=['PRICE'], flag="train", board=[0.7, 0.2, 0.1]):
        super(Dataset_My, self).__init__()
        data = pd.read_csv(path)
        # drop the nan value of the MEDA columns
        data = data.dropna(subset=[data.columns[-1]])
        # fill the nan value with 0
        data = data.fillna(0)

        self.l = len(data)
        board_l = [0, int(self.l * board[0]), int(self.l * (board[0] + board[1]))]
        board_r = [int(self.l * board[0]), int(self.l * (board[0] + board[1])), self.l]

        if flag == "train":
            ind = 0
        elif flag == "val":
            ind = 1
        else:
            ind = 2

        data = data[board_l[ind]:board_r[ind]]    
        self.l = len(data)

        self.data_y = torch.tensor(data[target].values).float()
        self.data_x = torch.tensor(data.drop(columns=target, axis=1).values).float()

        self.in_size = len(self.data_x[0])
        self.out_size = len(self.data_y[0])

    def __getitem__(self, index):
        return self.data_x[index], self.data_y[index]

    def __len__(self):
        return self.l
    
    def getsize(self):
        return self.in_size, self.out_size


class Test_My(Dataset):
    def __init__(self, path=r'../dataset/boston_housing_data/test_dataset.csv'):
        super(Test_My, self).__init__()
        data = pd.read_csv(path)
        # drop the nan value of the MEDA columns
        data = data.dropna(subset=[data.columns[-1]])
        # fill the nan value with 0
        data = data.fillna(0)

        self.l = len(data)
        col = data.columns[1:]
        self.data_x = torch.tensor(data[col].values).float()

    def __getitem__(self, index):
        return self.data_x[index]

    def __len__(self):
        return self.l
