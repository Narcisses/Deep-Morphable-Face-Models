import torch
import trimesh


def load_obj(filepath: str, device: str = 'cpu') -> tuple[torch.Tensor, torch.Tensor]:
    '''
    Load object from .obj file and convert to tensor.
    Output tensors are of shape (nb_vertices, 3).
    Return vertices and faces tensors.
    '''

    mesh = trimesh.load(filepath)
    vertices = torch.tensor(mesh.vertices, device=device, dtype=torch.float64)
    faces = torch.tensor(mesh.faces, device=device, dtype=torch.long)

    return vertices, faces


def save_obj(filepath: str, vertices: torch.Tensor, faces: torch.Tensor) -> None:
    '''
    Save object to .obj file.
    Input tensor must be of shape (nb_vertices, 3).
    '''

    mesh = trimesh.Trimesh(vertices=vertices.tolist(), faces=faces.tolist())
    mesh.export(filepath)


def load_neutral_mesh(path: str) -> tuple[torch.Tensor, torch.Tensor]:
    '''Load neutral mesh from disk.'''

    neutral_mesh = torch.load(path)
    vertices = neutral_mesh[0]
    faces = neutral_mesh[1]

    return vertices, faces