import pickle
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
    
class CustomMNISTDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        dataset_idx = self.indices[idx].item()
        data, label = self.dataset[dataset_idx]
        return {'idx': dataset_idx, 'data': data, 'label': label}
    

def decomposition_collate_fn(batch):
    collated_strings = []
    collated_floats = [] #torch.zeros(batch_size, sample_size)
    collated_labels = [] #torch.zeros(batch_size)
    for i, sample in enumerate(batch):
        dec, label = sample
        sample_strings = []
        sample_floats = torch.zeros(len(dec))
        collated_labels.append(label)
        for j, (a, b) in enumerate(dec):
            sample_strings.append(a)
            sample_floats[j] = b
        collated_floats.append(sample_floats)
        collated_strings.append(sample_strings)
    collated_labels = torch.stack(collated_labels)
    return collated_strings, collated_floats, collated_labels



class DecompositionDataset(torch.utils.data.IterableDataset):
    def __init__(self, file_path):
        self.file_path = file_path

    def __iter__(self):
        with open(self.file_path, 'rb') as f:
            while True:
                try:
                    item = pickle.load(f)
                    yield item
                except EOFError:
                    break