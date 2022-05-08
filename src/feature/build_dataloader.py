import torch
from torch.utils.data import DataLoader, Dataset


def collate_pad(batch) -> dict:
    max_len = max(len(row["feature"]) for row in batch)

    feature = torch.empty((len(batch), max_len), dtype=torch.long)
    labels = torch.empty(len(batch), dtype=torch.long)

    for idx, row in enumerate(batch):
        to_pad = max_len - len(row["feature"])
        feature[idx] = torch.cat((row["feature"], torch.zeros(to_pad)))
        labels[idx] = row['label']
    return {
        'feature': feature,
        'label': labels,
    }


def collate_caps(batch) -> dict:
    max_len = 1024

    feature = torch.empty((len(batch), max_len), dtype=torch.long)
    labels = torch.empty(len(batch), dtype=torch.long)

    for idx, row in enumerate(batch):
        if len(row["feature"]) <= max_len:
            to_pad = max_len - len(row["feature"])
            feature[idx] = torch.cat((row["feature"], torch.zeros(to_pad)))
        else:
            feature[idx] = row["feature"][:max_len]
        labels[idx] = row['label']
    return {
        'feature': feature,
        'label': labels,
    }


def build_dataloader(dataset: Dataset, batch_size: int, collate_fn) -> DataLoader:
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    return loader
