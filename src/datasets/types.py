import torch

from torch.utils.data import Dataset


def collate_fn(batch):
    n_batch = len(batch)
    d_lengths = [int(torch.argmax(diagram[:, 0])) for diagram, _ in batch]

    max_len = max(d_lengths)
    diagrams = torch.ones([n_batch, max_len, 9]) * 0.0
    masks = torch.zeros([n_batch, max_len]).bool()
    labels = torch.zeros(n_batch).long()

    for i, (diagram, label) in enumerate(batch):
        diagrams[i][: d_lengths[i]] = diagram[: d_lengths[i]]
        masks[i][d_lengths[i] :] = True
        labels[i] = label

    return diagrams, masks, labels


class PersistenceDataset(Dataset):
    def __init__(self, diagrams, targets, idx=None, eps=None):
        self.targets = targets
        idx_tensor = torch.tensor(idx) if idx is not None else None

        data = torch.ones([len(diagrams), max(map(len, diagrams)) + 1, 9]) * torch.inf

        for i, dgm in enumerate(diagrams):
            if eps is not None:
                eps_idx = (dgm[:, 1] - dgm[:, 0]) >= eps
                dgm = dgm[eps_idx]

            dgm_direction = torch.clone(dgm[:, -1])
            if idx_tensor is not None:
                dgm_idx = torch.isin(dgm_direction, idx_tensor)
                dgm = dgm[dgm_idx]
                dgm_angle = torch.clone(dgm[:, -2])

                data[i, : len(dgm), :4] = dgm[:, :4]
                data[i, : len(dgm), 4] = torch.sin(dgm_angle * (torch.pi / 40))
                data[i, : len(dgm), 5] = torch.sin(dgm_angle * (torch.pi / 90))
                data[i, : len(dgm), 6] = torch.sin(dgm_angle * (torch.pi / 180))
                data[i, : len(dgm), 7] = torch.sin(dgm_angle * (torch.pi / 130))
                data[i, : len(dgm), 8] = torch.sin(dgm_angle * (torch.pi / 360))
            else:
                data[i, : len(dgm), :4] = dgm[:, :4]

        if idx_tensor is not None:
            max_len = torch.argmax(data[:, :, 0], axis=1).max()
            data = data[:, : max_len + 1]

        self.data = data

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


class ImageDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        image = self.data[idx]
        if self.transform:
            image = self.transform(image)
        label = self.targets[idx]
        return image, label
