import torch

from training.meshio import load_neutral_mesh, load_obj
from training.model import load_model
from viz.debug import show_mesh


model = 'model2.pt'
reference = 'reference_neutral.pt'
neutral = '../../FaceWarehouse/Tester_110/Blendshape/shape_0.obj'
example = '../../FaceWarehouse/Tester_110/Blendshape/shape_9.obj'

avg_ref, faces = load_neutral_mesh(reference)
neutral, _ = load_obj(neutral)
example, _ = load_obj(example)
vae, checkpoint = load_model(model)

neutral = neutral.to(dtype=torch.float32)
avg_ref = avg_ref.to(dtype=torch.float32)
example = example.to(dtype=torch.float32)

blendweights = torch.zeros(checkpoint['input_dim_exp'])
blendweights[9] = 1.0
blendweights = blendweights.view(1, -1)

neutral = neutral.view(1, 11510, 3)

# Reshape input
neutral -= avg_ref.view(1, 11510, 3)
neutral = neutral.view(-1, 11510 * 3)

# Forward pass
output, mean_id, logvar_id, mean_exp, logvar_exp = vae(neutral, blendweights)

# Reshape output
output += avg_ref.view(1, 11510 * 3)
output = output.view(-1, 11510, 3)

show_mesh(example, faces)
show_mesh(output[0], faces)