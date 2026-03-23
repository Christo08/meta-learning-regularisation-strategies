import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, data, labels):
        # Ensure data and labels are tensors
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data.values, dtype=torch.float32)
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels.values, dtype=torch.int64)

        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = {
            'data': self.data[index],
            'label': self.labels[index]
        }
        return sample