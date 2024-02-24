import torch
from matplotlib import pyplot as plt


def plot_loss(tr_geo, tr_klid, tr_klexp, val_geo, val_klid, val_klexp):
    '''Plot the loss values.'''
    plt.figure(figsize=(12, 6))
    plt.plot(tr_geo, label='Train Geometry Loss')
    plt.plot(tr_klid, label='Train KL ID Loss')
    plt.plot(tr_klexp, label='Train KL Exp Loss')
    plt.plot(val_geo, label='Val Geometry Loss')
    plt.plot(val_klid, label='Val KL ID Loss')
    plt.plot(val_klexp, label='Val KL Exp Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plt.close()


# Load model from checkpoint
checkpoint = torch.load('models/model1031.pt', map_location=torch.device('cpu'))

# Print information
# Training hyperparameters
print('>>>>>>>>>> Training hyperparameters:')
print(f'Ratio train: {checkpoint["ratioTr"]}')
print(f'Epochs: {checkpoint["epochs"]}')
print(f'Batch size: {checkpoint["batch"]}')
print(f'Regularization: {checkpoint["reg"]}')
print(f'LR: {checkpoint["lr"]}')
print(f'Decay: {checkpoint["decay"]}')
print(f'Patience: {checkpoint["patience"]}')
print(f'Threshold: {checkpoint["thresh"]}')

# Inputs
print('>>>>>>>>>> Inputs:')
print(f'Input dim ID: {checkpoint["input_dim_id"]}')
print(f'Input dim Exp: {checkpoint["input_dim_exp"]}')

# Latent space
print('>>>>>>>>>> Latent space:')
print(f'Latent dim ID: {checkpoint["latent_dim_id"]}')
print(f'Latent dim Exp: {checkpoint["latent_dim_exp"]}')

# Model parameters
print('>>>>>>>>>> Model parameters:')
print(f'Encoder ID: {checkpoint["hidden_dim_id"]}')
print(f'Encoder EXP: {checkpoint["hidden_dim_exp"]}')
print(f'Decoder: {checkpoint["hidden_dim_dec"]}')

# Plot loss
plot_loss(checkpoint["tr_geo"], checkpoint["tr_klid"], 
          checkpoint["tr_klexp"], checkpoint["val_geo"], 
          checkpoint["val_klid"], checkpoint["val_klexp"])
