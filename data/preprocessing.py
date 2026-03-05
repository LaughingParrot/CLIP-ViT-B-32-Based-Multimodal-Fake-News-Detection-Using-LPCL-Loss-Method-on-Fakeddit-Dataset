import torch
from torch.utils.data import Dataset


class FakedditTextDataset(Dataset):

    def __init__(self, dataframe, tokenizer):

        self.texts = dataframe["text"].tolist()
        self.labels = dataframe["label"].tolist()

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):

        text = str(self.texts[idx])
        label = int(self.labels[idx])

        tokens = self.tokenizer([text])[0]

        return {
            "text_tokens": tokens,
            "label": torch.tensor(label, dtype=torch.long)
        }