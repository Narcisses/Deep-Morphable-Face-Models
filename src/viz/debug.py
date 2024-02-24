import torch
import trimesh
from PIL import Image
from matplotlib import pyplot as plt


def show_mesh(vertices: torch.Tensor, faces: torch.Tensor) -> None:
    '''Show a mesh given its vertices and faces.'''

    trimesh.Trimesh(vertices=vertices.tolist(), faces=faces.tolist()).show()


def display_images(image_list: list[Image.Image], title: str = 'Images') -> None:
    '''
    Display a list of images.
    Input: list of images (PIL.Image.Image or np.ndarray)
    '''

    num_images = len(image_list)

    # Create a subplot grid based on the number of images
    rows = len(image_list) // 3
    cols = len(image_list) // rows

    # Create a figure and set its size
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    fig.canvas.manager.set_window_title(title)

    # Flatten the axes array if it's a single row or column
    axes = axes.flatten() if num_images > 1 else [axes]

    # Display each image on the corresponding subplot
    for i, image in enumerate(image_list):
        axes[i].imshow(image)
        axes[i].axis('off')

    plt.show()
