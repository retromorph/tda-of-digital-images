import torch
import numpy as np
import gudhi as gd


def pht(image, image_base=None, pos=None, eps=None):
    if len(image.shape) != 3:
        raise ValueError("Image tensor dimensions should be (channel, width, height).")

    dim = 6 if pos is not None else 3
    dgm_pht = torch.zeros((0, dim))
    dgms = sublevel_persistence(image, eps=eps, pos=pos, sort="persistence")

    for _i, dgm in enumerate(dgms):
        dgm_pht = torch.cat([dgm_pht, dgm])

    if image_base is not None:
        dgms_base = sublevel_persistence(image_base, eps=eps, pos=None, sort="persistence")
        for _i, dgm_base_ in enumerate(dgms_base):
            dgm_base = torch.zeros(len(dgm_base_), dim)
            dgm_base[:, :3] = dgm_base_
            dgm_base[:, 0] /= 1  # scale
            dgm_base[:, 1] /= 1  # scale
            dgm_base[:, 3] = 1.0
            dgm_base[:, -1] = image.shape[0]  # n+1 direction
            dgm_pht = torch.cat([dgm_pht, dgm_base])

    return dgm_pht


def sublevel_persistence(image, eps=None, pos=None, inf="max", sort="birth"):
    if len(image.shape) != 3:
        raise ValueError("Image tensor dimensions should be (channel, width, height).")

    diagrams = []
    persistence = lambda x: x[:, 1] - x[:, 0]

    # for each channel
    for k, channel in enumerate(image):
        diagram_channel_gudhi = gd.CubicalComplex(top_dimensional_cells=channel).persistence()

        # convert a diagram from GUDHI format to n x 3 ndarray
        diagram_channel = np.zeros((len(diagram_channel_gudhi), 3))
        for i, (dim, (birth, death)) in enumerate(diagram_channel_gudhi):
            diagram_channel[i] = (birth, death, dim)

        # work with infs
        if inf == "max":
            diagram_channel = np.nan_to_num(diagram_channel, posinf=torch.max(channel))
        elif inf == "remove":
            diagram_channel = diagram_channel[~np.isinf(diagram_channel).any(axis=1)]
        else:
            raise ValueError("Inf should be 'max' or 'remove'.")

        # remove points w/ persistence less \eps
        if eps is not None:
            diagram_channel = diagram_channel[persistence(diagram_channel) > eps]

        # add positional encoding
        if pos is not None:
            pos_elements = np.repeat(pos[k], len(diagram_channel))[..., np.newaxis]
            pos_idx = np.repeat(k, len(diagram_channel))[..., np.newaxis]
            diagram_channel = np.concatenate([diagram_channel, np.zeros_like(pos_idx), pos_elements, pos_idx], axis=1)

        # sort by dim, then birth or persistence
        if sort == "birth":
            sort_idx = np.lexsort([diagram_channel[:, 0], diagram_channel[:, 2]])
        elif sort == "persistence":
            sort_idx = np.lexsort([persistence(diagram_channel), diagram_channel[:, 2]])
        else:
            raise ValueError("Sort should be 'birth' or 'persistence'.")
        diagram_channel = diagram_channel[sort_idx]

        diagrams.append(torch.tensor(diagram_channel))

    return diagrams
