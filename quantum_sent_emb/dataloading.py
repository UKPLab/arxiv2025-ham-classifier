# Sentiment Analysis Dataset
# Builds from pandas dataframe
import torch


class SentimentDataset(torch.utils.data.Dataset):
    
    def __init__(self, dataframe):
        self.data = dataframe

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data.loc[idx, 'review'], self.data.loc[idx, 'label']