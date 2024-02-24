import io
import torch
import trimesh
import numpy as np
from PIL import Image


def average_meshes(meshes: list) -> torch.Tensor:
    '''Average all meshes.'''

    return meshes.mean(dim=0)


def mesh_to_image(mesh: trimesh.Trimesh, resolution: tuple = (256, 256)) -> torch.Tensor:
    '''Convert mesh to image.'''

    scene = mesh.scene()
    data = scene.save_image(resolution=resolution)
    image = np.array(Image.open(io.BytesIO(data)))

    return image
