import os
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from training.loader import FaceWarehouseDataset
from training.loader import get_all_filepaths, split_data
from training.model import VAE, loss_fn
import training.meshio as meshio
from training.common import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_dtype(torch.float64)


def evaluate(model: VAE, optimizer: Adam,
             dataloader: DataLoader, 
             train: bool = True) -> float:
    '''Run network evaluation on a batch of data.'''

    global kl_id_factor, kl_exp_factor

    if train:
        model.train()
    else:
        model.eval()

    running_geo, running_klid, running_klexp = 0.0, 0.0, 0.0

    for _, batch in enumerate(dataloader):
        # Ground truth=true expression meshes (batch, 11510, 3)
        expressions = batch['expression']
        # Neutral meshes (batch, 11510, 3)
        neutrals = batch['neutral']
        # Expressions encoded as blendweights (batch, vec size)
        blendweightses = batch['weights']

        # Zero grad
        if train and optimizer is not None:
            optimizer.zero_grad()

        # Reshape input
        neutrals -= avg_ref.view(1, 11510, 3)
        neutrals = neutrals.view(-1, INPUT_DIM_ID)

        # Forward pass
        output, mean_id, logvar_id, mean_exp, logvar_exp = model(neutrals, blendweightses)

        # Reshape output
        output += avg_ref.view(1, INPUT_DIM_ID)
        output = output.view(-1, 11510, 3)

        # Loss
        geo, klid, klexp = loss_fn(expressions, output,
                       mean_id, logvar_id, 
                       mean_exp, logvar_exp,
                       klid=kl_id_factor, 
                       klexp=kl_exp_factor, 
                       wgeo=W_GEO)
        loss = geo + klid + klexp
        
        # Backward pass
        if train:
            loss.backward()
            optimizer.step()

        running_geo += geo.item()
        running_klid += klid.item()
        running_klexp += klexp.item()

    geo = running_geo / len(dataloader)
    klid = running_klid / len(dataloader)
    klexp = running_klexp / len(dataloader)

    return geo, klid, klexp


def train(model: VAE, optimizer: Adam, scheduler: 
          ReduceLROnPlateau, print_every: int = 100) -> None:
    '''Train the model.'''

    global kl_id_factor, kl_exp_factor

    best_val = float('inf')

    # Keep track of loss values
    tr_geo, tr_klid, tr_klexp = [], [], []
    val_geo, val_klid, val_klexp = [], [], []

    for epoch in range(EPOCHS):
        # Increase KL Loss weight
        if epoch % KL_STEP_INC == 0 and epoch >= KL_WAIT:
            kl_id_factor *= KL_FACTOR
            kl_exp_factor *= KL_FACTOR

            if kl_id_factor > KL_LIMIT:
                kl_id_factor = KL_LIMIT

            if kl_exp_factor > KL_LIMIT:
                kl_exp_factor = KL_LIMIT

        # Train
        geo, klid, klexp = evaluate(model, optimizer,
                            train_dataloader, 
                            train=True)

        # Validation
        with torch.no_grad():
            vgeo, vklid, vklexp = evaluate(model, None,
                            val_dataloader, 
                            train=False)

        # Update scheduler
        scheduler.step(geo)
        
        # Track stats
        tr_geo.append(geo)
        tr_klid.append(klid)
        tr_klexp.append(klexp)

        val_geo.append(vgeo)
        val_klid.append(vklid)
        val_klexp.append(vklexp)

        # Print advancement
        if (epoch + 1) % print_every == 0 or epoch == 0:
            epo = f'Epoch: {epoch + 1}'
            tr_geo_loss = f'Geo: {geo:.2f}'
            val_geo_loss = f'VGeo: {vgeo:.2f}'
            tr_klid_loss = f'KLID: {abs(klid):.2f}'
            tr_klexp_loss = f'KLEXP: {abs(vklexp):.2f}'
            lr = f'LR: {scheduler.get_last_lr()[0]:.3E}'
            print(f'{epo} \t | {lr} \t | {tr_geo_loss} \t | {val_geo_loss} \t | {tr_klid_loss} \t | {tr_klexp_loss}')

        # Save model if better
        if vgeo < best_val and epoch >= KL_WAIT:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'input_dim_id': INPUT_DIM_ID,
                'input_dim_exp': NB_EXPRESSIONS,
                'hidden_dim_id': HIDDEN_DIM_ID,
                'hidden_dim_exp': HIDDEN_DIM_EXP,
                'latent_dim_id': LATENT_DIM_ID,
                'latent_dim_exp': LATENT_DIM_EXP,
                'hidden_dim_dec': HIDDEN_DIM_DECODER,
                'lr': LR,
                'decay': DECAY,
                'patience': PATIENCE,
                'thresh': THRESHOLD,
                'reg': REG,
                'batch': BATCH,
                'epochs': EPOCHS,
                'ratioTr': RATIO_TRAIN,
                'tr_geo': tr_geo,
                'tr_klid': tr_klid,
                'tr_klexp': tr_klexp,
                'val_geo': val_geo,
                'val_klid': val_klid,
                'val_klexp': val_klexp,
            }, OUT_MODEL_PATH)
            best_val = vgeo


def number_parameters(model: VAE) -> int:
    '''Calculate total number of parameters in model.'''

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_avg_ref(filepath: str) -> torch.Tensor:
    '''
    Calculate mean of all neutral expressions and save it as file.
    If neutral reference mesh file already exists, load it.
    '''

    if os.path.isfile(filepath):
        print(f'Loading reference neutral from: {filepath}')

        avg_ref, faces = torch.load(filepath)
        avg_ref = avg_ref.to(device, dtype=torch.float64)    
    else:
        print(f'No reference neutral found, saving in: {filepath}')

        neutral_meshes = []
        faces = None

        # Load all neutral meshes
        for _id in range(1, NB_IDENTITIES + 1):
            person_dir = os.path.join(DATA_PATH, f'Tester_{_id}/Blendshape')

            if os.path.isdir(person_dir):
                neutral_path = os.path.join(person_dir, f'shape_0.obj')
                mesh, face = meshio.load_obj(neutral_path)
                neutral_meshes.append(mesh)

                if faces is None:
                    faces = face

        # Calculate average neutral mesh
        avg_ref = torch.stack(neutral_meshes).mean(dim=0)
        avg_ref = avg_ref.to(device, dtype=torch.float64)

        # Save reference to file
        save = (avg_ref.to('cpu'), faces)
        torch.save(save, filepath)

        print('Reference neutral mesh successfully loaded!')

    return avg_ref, faces


# Get all dataset (only load file paths for meshes)
data = get_all_filepaths(DATA_PATH, 1, NB_IDENTITIES, NB_EXPRESSIONS, device)
expr_paths, neutrals_indices, blendweights, neutrals_paths = data

# Split into train and validation sets
data = split_data(expr_paths, neutrals_indices, blendweights, RATIO_TRAIN)
tr_expr_paths, tr_neutrals_indices, tr_blendweights = data[:3]
val_expr_paths, val_neutrals_indices, val_blendweights = data[3:]

# Setup dataset and dataloaders
train_dataset = FaceWarehouseDataset(
    tr_expr_paths,
    tr_neutrals_indices,
    tr_blendweights,
    neutrals_paths,
    device)

val_dataset = FaceWarehouseDataset(
    val_expr_paths, 
    val_neutrals_indices, 
    val_blendweights,
    neutrals_paths,
    device)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH, shuffle=True)

# Get reference neutral mesh
avg_ref, faces = get_avg_ref('reference_neutral.pt')

# Build model
model = VAE(input_dim_id=INPUT_DIM_ID,
            latent_dim_id=LATENT_DIM_ID,
            input_dim_exp=NB_EXPRESSIONS,
            latent_dim_exp=LATENT_DIM_EXP,
            hidden_dim_id=HIDDEN_DIM_ID,
            hidden_dim_exp=HIDDEN_DIM_EXP,
            hidden_dim_decoder=HIDDEN_DIM_DECODER,
            output_dim=INPUT_DIM_ID).to(device=device, dtype=torch.float64)
optimizer = Adam(model.parameters(), lr=LR, weight_decay=REG)
scheduler = ReduceLROnPlateau(optimizer, factor=DECAY, threshold=THRESHOLD, patience=PATIENCE)

kl_id_factor = W_KL_ID
kl_exp_factor = W_KL_EXP

# Print info
print(f'Model: \t \t \t {OUT_MODEL_PATH}')
print(f'Device: \t \t {device}')
print(f'DB[ID|EX]: \t \t {NB_IDENTITIES} | {NB_EXPRESSIONS}')
print(f'Train set: \t \t {len(train_dataset)}')
print(f'Val set: \t \t {len(val_dataset)}')
print(f'Parameters: \t \t {number_parameters(model):,}')
print(f'Latent[ID|EX]: \t \t {LATENT_DIM_ID} | {LATENT_DIM_EXP}')
print(f'Batch size: \t \t {BATCH}')
print(f'Epochs: \t \t {EPOCHS}')
print(f'Reg: \t \t \t {REG}')
print(f'LR: \t \t \t {LR}')
print(f'Decay: \t \t \t {DECAY}')
print(f'Patience: \t \t {PATIENCE}')
print(f'Threshold: \t \t {THRESHOLD}')
print(f'Loss W[GEO|ID|EX]: \t {W_GEO} | {W_KL_ID} | {W_KL_EXP}')
print(f'KL [STEP|FACTOR]: \t {KL_STEP_INC} | {KL_FACTOR}')
print(f'KL [LIMIT|WAIT]: \t {KL_LIMIT} | {KL_WAIT}')

# Train model
print('Training...')
train(model, optimizer, scheduler, print_every=100)
