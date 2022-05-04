import torch
from torch.utils.data import Dataset


class IMDBDataset(Dataset):
    def __init__(self, text: list, label: list):
        self.text = text
        self.label = label

    def __getitem__(self, item):
        return {
            'feature': torch.tensor(self.text[item]),
            'label': torch.tensor(self.label[item])
        }

    def __len__(self):
        return len(self.text)
