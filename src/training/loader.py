import os
import random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import training.meshio as meshio


def get_blendweights(id_exp: int, nb_exp: int) -> torch.Tensor:
    '''
    Return the blendweight vector for the given expression.
    Ex: expr0 = [1, 0, 0, 0, ...]
        expr3 = [0, 0, 0, 1, ...]
    '''

    return F.one_hot(torch.tensor(id_exp), num_classes=nb_exp).double()


def get_all_filepaths(root: str, start_id: int, end_id: int, nb_shapes: int, device: str='cpu') -> list:
    '''
    Get all filepaths from the dataset root directory.
    EXPR_PATHS: all expression paths of every person in the dataset
    BLENDWEIGHTS: expression blendweights, one for each expression
    NEUTRAL_INDICES: neutral path ID in NEUTRAL_PATHS, one for each expression
    NEUTRAL_PATHS: all neutral expression paths
    '''

    expr_paths, neutrals_indices, blendweights = [], [], []
    neutrals_paths = []

    for i, _id in enumerate(range(start_id, end_id)):
        person_dir = os.path.join(root, f'Tester_{_id}/Blendshape')
        neu_path = os.path.join(person_dir, 'shape_0.obj')
        neutrals_paths.append(neu_path)

        if os.path.isdir(person_dir):
            for expr in range(nb_shapes):
                expr_path = os.path.join(person_dir, f'shape_{expr}.obj')
                weights = get_blendweights(expr, nb_shapes).to(device)

                neutrals_indices.append(i)
                expr_paths.append(expr_path)
                blendweights.append(weights)

    return expr_paths, neutrals_indices, blendweights, neutrals_paths


def split_data(expr_paths: list, neutrals_indices: list, blendweights: list, ratio: float) -> tuple:
    '''Shuffle and split data into train and validation splits.'''

    # Shuffle
    data = list(zip(expr_paths, neutrals_indices, blendweights))
    random.shuffle(data)
    expr_paths, neutrals_indices, blendweights = zip(*data)

    # Split
    split = int(len(expr_paths) * ratio)
    a, b, c = expr_paths[:split], neutrals_indices[:split], blendweights[:split]
    d, e, f = expr_paths[split:], neutrals_indices[split:], blendweights[split:]

    return a, b, c, d, e, f


class FaceWarehouseDataset(Dataset):
    def __init__(self, expr_paths: list, neutrals_indices: list, 
                 blendweights: list, neutrals_paths: list, 
                 device: str = 'cpu') -> None:
        super(FaceWarehouseDataset, self).__init__()

        self.cached_exp = {}
        self.cached_neu = {}
        self.device = device

        self.expr_paths = expr_paths
        self.neutrals_indices = neutrals_indices
        self.blendweights = blendweights
        self.neutrals_paths = neutrals_paths

    def __len__(self) -> int:
        return len(self.expr_paths)

    def __getitem__(self, idx: int) -> dict:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if idx in self.cached_exp:
            expr_mesh = self.cached_exp[idx]
            neu_mesh = self.cached_neu[self.neutrals_indices[idx]]
        else:
            # Load expression mesh
            expr_path = self.expr_paths[idx]
            expr_mesh, _ = meshio.load_obj(expr_path, self.device)

            # Load neutral mesh
            neu_path = self.neutrals_paths[self.neutrals_indices[idx]]
            neu_mesh, _ = meshio.load_obj(neu_path, self.device)

            # Save in cache
            self.cached_exp[idx] = expr_mesh
            self.cached_neu[self.neutrals_indices[idx]] = neu_mesh

        # Get blendweights
        weights = self.blendweights[idx]

        sample = {
            'expression': expr_mesh,
            'weights': weights,
            'neutral': neu_mesh
        }

        return sample
