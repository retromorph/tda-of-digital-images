from torch.utils.data import Dataset


class EncodedFeatureDataset(Dataset):
    """Adapter over PHT dataset that lazily applies a fixed encoder."""

    def __init__(self, base_dataset, encoder):
        self.base_dataset = base_dataset
        self.encoder = encoder

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        diagram, target = self.base_dataset[idx]
        feature = self.encoder(diagram)
        return feature, target
