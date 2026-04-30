import torch

from torch.utils.data import Dataset


def collate_fn(batch, task="classification", idx=None, eps=None, sin_encoding_config=None):
    idx_tensor = torch.as_tensor(idx, dtype=torch.float32) if idx is not None else None
    sin_denominators = sin_encoding_config or [40.0, 90.0, 180.0, 130.0, 360.0]

    processed = []
    lengths = []
    out_dim = None
    for diagram, label in batch:
        dgm = diagram
        if dgm.ndim != 2:
            raise ValueError("Each diagram must be a 2D tensor.")

        if eps is not None:
            keep = (dgm[:, 1] - dgm[:, 0]) >= float(eps)
            dgm = dgm[keep]

        if idx_tensor is not None and dgm.shape[0] > 0:
            direction_idx = dgm[:, -1]
            keep = torch.isin(direction_idx, idx_tensor.to(direction_idx.device))
            dgm = dgm[keep]

        if dgm.shape[0] > 0:
            base = dgm[:, : min(4, dgm.shape[1])]
            if idx_tensor is not None and len(sin_denominators) > 0 and dgm.shape[1] >= 2:
                angle = dgm[:, -2]
                sin_feats = [torch.sin(angle * (torch.pi / float(den))) for den in sin_denominators]
                feat = torch.cat([base] + [s.unsqueeze(1) for s in sin_feats], dim=1)
            else:
                feat = base
        else:
            width = min(4, dgm.shape[1]) + (len(sin_denominators) if idx_tensor is not None else 0)
            feat = torch.zeros((0, width), dtype=diagram.dtype)

        out_dim = feat.shape[1] if out_dim is None else out_dim
        processed.append((feat, label))
        lengths.append(feat.shape[0])

    n_batch = len(batch)
    max_len = max(lengths) if lengths else 0
    diagrams = torch.zeros([n_batch, max_len, out_dim or 0], dtype=torch.float32)
    masks = torch.zeros([n_batch, max_len]).bool()
    labels = torch.zeros(n_batch)

    for i, (diagram, label) in enumerate(processed):
        diagrams[i][: lengths[i]] = diagram[: lengths[i]]
        masks[i][lengths[i] :] = True
        labels[i] = label

    if task == "classification":
        labels = labels.long()
    else:
        labels = labels.float()

    return diagrams, masks, labels


class PersistenceDataset(Dataset):
    def __init__(self, diagrams, targets, schema=None):
        self.diagrams = [torch.as_tensor(dgm).float() for dgm in diagrams]
        self.targets = targets
        self.schema = schema or {}

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.diagrams[idx], self.targets[idx]


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
