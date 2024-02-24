import typing
import torch
import torch.nn as nn


class EncoderID(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int) -> None:
        super(EncoderID, self).__init__()

        self.fc1 = nn.Linear(input_dim,  hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)

        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.act_fn = nn.LeakyReLU(0.2)

        self.init_weights()

    def init_weights(self) -> None:
        with torch.no_grad():
            # Glorot initialization
            gain = nn.init.calculate_gain('leaky_relu')
            nn.init.xavier_normal_(self.fc1.weight, gain=gain)
            nn.init.xavier_normal_(self.fc2.weight, gain=gain)
            nn.init.xavier_normal_(self.fc3.weight, gain=gain)
            nn.init.xavier_normal_(self.fc4.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_mean.weight)
            nn.init.xavier_normal_(self.fc_logvar.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act_fn(self.fc1(x))
        x = self.act_fn(self.fc2(x)) + x
        x = self.act_fn(self.fc3(x)) + x
        x = self.act_fn(self.fc4(x)) + x

        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)

        return mean, logvar


class EncoderEXP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int) -> None:
        super(EncoderEXP, self).__init__()
        self.fc1 = nn.Linear(input_dim,  hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)

        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.act_fn = nn.LeakyReLU(0.2)

        self.init_weights()

    def init_weights(self) -> None:
        with torch.no_grad():
            # Glorot initialization
            gain = nn.init.calculate_gain('leaky_relu')
            nn.init.xavier_normal_(self.fc1.weight, gain=gain)
            nn.init.xavier_normal_(self.fc2.weight, gain=gain)
            nn.init.xavier_normal_(self.fc3.weight, gain=gain)
            nn.init.xavier_normal_(self.fc4.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_mean.weight)
            nn.init.xavier_normal_(self.fc_logvar.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act_fn(self.fc1(x))
        x = self.act_fn(self.fc2(x)) + x
        x = self.act_fn(self.fc3(x)) + x
        x = self.act_fn(self.fc4(x)) + x

        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)

        return mean, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int) -> None:
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

        self.act_fn = nn.LeakyReLU(0.2)

        self.init_weights()

    def init_weights(self) -> None:
        with torch.no_grad():
            # Glorot initialization
            gain = nn.init.calculate_gain('leaky_relu')
            nn.init.xavier_normal_(self.fc1.weight, gain=gain)
            nn.init.xavier_normal_(self.fc2.weight, gain=gain)
            nn.init.xavier_normal_(self.fc3.weight, gain=gain)
            nn.init.xavier_normal_(self.fc4.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_out.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act_fn(self.fc1(x))
        x = self.act_fn(self.fc2(x))
        x = self.act_fn(self.fc3(x))
        x = self.act_fn(self.fc4(x))

        x = self.fc_out(x)

        return x


class VAE(nn.Module):
    def __init__(self, input_dim_id: int, latent_dim_id: int,
                 input_dim_exp: int, latent_dim_exp: int,
                 hidden_dim_id: int, hidden_dim_exp: int, 
                 hidden_dim_decoder: int, output_dim: int) -> None:
        super(VAE, self).__init__()

        self.encoderID = EncoderID(input_dim_id, hidden_dim_id, latent_dim_id)
        self.encoderEXP = EncoderEXP(input_dim_exp, hidden_dim_exp, latent_dim_exp)
        self.decoder = Decoder(latent_dim_id + latent_dim_exp, hidden_dim_decoder, output_dim)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        mean_id, logvar_id = self.encoderID(a)
        z_id = self.reparameterize(mean_id, torch.exp(0.5 * logvar_id))

        mean_exp, logvar_exp = self.encoderEXP(b)
        z_exp = self.reparameterize(mean_exp, torch.exp(0.5 * logvar_exp))

        x = self.decoder(torch.concat((z_id, z_exp), dim=1))

        return x, mean_id, logvar_id, mean_exp, logvar_exp
    
    def reparameterize(self, mean: torch.Tensor, var: torch.Tensor)-> torch.Tensor:
        epsilon = torch.randn_like(var)
        z = mean + var * epsilon

        return z


def loss_fn(x: torch.Tensor, x_hat: torch.Tensor, 
            mean_id: torch.Tensor, logvar_id: torch.Tensor, 
            mean_exp: torch.Tensor, logvar_exp: torch.Tensor,
            klid: float, klexp: float, wgeo: float) -> list[torch.Tensor]:
    L1 = nn.functional.l1_loss(x_hat, x, reduction='sum')
    KLD_ID = -0.5 * torch.sum(1 + logvar_id - mean_id.pow(2) - logvar_id.exp())
    KLD_EXP = -0.5 * torch.sum(1 + logvar_exp - mean_exp.pow(2) - logvar_exp.exp())

    return wgeo * L1, klid * KLD_ID, klexp * KLD_EXP


def load_model(path: str, device: str = 'cpu') -> tuple[VAE, typing.Any]:
    '''Load model from disk.'''

    checkpoint = torch.load(path, map_location=torch.device(device))

    model = VAE(input_dim_id=checkpoint['input_dim_id'],
            latent_dim_id=checkpoint['latent_dim_id'],
            input_dim_exp=checkpoint['input_dim_exp'],
            latent_dim_exp=checkpoint['latent_dim_exp'],
            hidden_dim_id=checkpoint['hidden_dim_id'],
            hidden_dim_exp=checkpoint['hidden_dim_exp'],
            hidden_dim_decoder=checkpoint['hidden_dim_dec'],
            output_dim=checkpoint['input_dim_id'])

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, checkpoint
