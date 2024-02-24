import trimesh

from training.utils import mesh_to_image
from src.viz.debug import display_images


mesh = trimesh.load('../data/FaceWarehouse/Tester_1/Blendshape/shape_0.obj')
image = mesh_to_image(mesh, resolution=(256, 256))

images = [image, image, image, image, image]
display_images(images)
