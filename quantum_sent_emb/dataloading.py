import torch

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, data_key='data', label_key='label'):
        self.dataset = dataset
        self.data_key = data_key
        self.label_key = label_key

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        data = sample[self.data_key]
        label = sample[self.label_key]
        return {'idx': idx, 'data': data, 'label': label}