import tokenizer.tokenizer as tokenizer
import torch
from torch.utils.data import Dataset, DataLoader


class ErtakDataset(Dataset):
    def __init__(self, data, block_size:int=8, stride:int=8):
        self.data = data
        self.block_size = block_size
        self.stride = stride

        X = []
        Y = []

        for i in range(0, len(self.data) - block_size, stride):
            X.append(self.data[i:i+block_size])
            Y.append(self.data[i+1:i+block_size+1])

        self.X = torch.tensor(X, dtype=torch.long)
        self.Y = torch.tensor(Y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def get_data_loader(split:str='train', batch_size:int=16, block_size:int=8, stride:int=8, shuffle:bool=True):
    file_name = (
        'uzbek_ertak/uzbek_ertak_training.txt'
        if split == 'train' else
        'uzbek_ertak/uzbek_ertak_validation.txt'
    )

    text = open(file_name, 'rt', encoding='utf-8').read()
    encoded_data = tokenizer.encode(text)

    ds = ErtakDataset(data=encoded_data, block_size=block_size, stride=stride)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=True)
    return loader

