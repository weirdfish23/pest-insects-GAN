from torch import nn
import torch.nn.functional as F
import torch

class Encoder(nn.Module):
    def __init__(self, input_dim=3, latent_dim=64, hidden_dim=32):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.enc = nn.Sequential(                                                                         # (3, 64, 64)
            self.make_enc_block(input_dim, hidden_dim, kernel_size=4, stride=2, padding=1),                 # (32, 32, 32)
            self.make_enc_block(hidden_dim, hidden_dim * 2, kernel_size=4, stride=2, padding=1),          # (64, 16, 16)
            self.make_enc_block(hidden_dim * 2, hidden_dim * 4, kernel_size=4, stride=2, padding=1),      # (128, 8, 8)
            self.make_enc_block(hidden_dim * 4, hidden_dim * 8, kernel_size=4, stride=2, padding=1),      # (256, 4, 4)
            self.make_enc_block(hidden_dim * 8, hidden_dim * 16, kernel_size=4, stride=1, padding=0, final_layer=True), # (512, 1, 1)
        )

        self.fc_mu = nn.Linear(hidden_dim * 16, latent_dim)
        self.fc_var = nn.Linear(hidden_dim * 16, latent_dim)

    def make_enc_block(self, input_channels, output_channels, kernel_size=4, stride=2, padding=0, final_layer=False):
        if not final_layer:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding),
                nn.BatchNorm2d(output_channels),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding),
            )

    def forward(self, image):
        result = self.enc(image)
        result = torch.flatten(result, start_dim=1)

        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

class Decoder(nn.Module):
    def __init__(self, input_dim=71, im_chan=3, hidden_dim=32):
        super(Decoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.decoder_input = nn.Linear(input_dim, hidden_dim * 16)                                                          # (64*16, 1, 1)

        self.dec = nn.Sequential(                                                                                           # (64*16, 1, 1)
            self.make_dec_block(hidden_dim * 16, hidden_dim * 8, kernel_size=3, stride=2, padding=1, output_padding=1),     # (64*8, 2, 2)
            self.make_dec_block(hidden_dim * 8, hidden_dim * 4, kernel_size=3, stride=2, padding=1, output_padding=1),      # (64*4, 4, 4)
            self.make_dec_block(hidden_dim * 4, hidden_dim * 2, kernel_size=3, stride=2, padding=1, output_padding=1),      # (64*2, 8, 8)
            self.make_dec_block(hidden_dim * 2, hidden_dim , kernel_size=3, stride=2, padding=1, output_padding=1),         # (64, 16, 16)
            self.make_dec_block(hidden_dim , hidden_dim , kernel_size=3, stride=2, padding=1, output_padding=1),            # (64, 32, 32)
        )

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1, output_padding=1),               # (64, 64, 64)
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dim, out_channels= 3,kernel_size= 3, padding= 1),
            nn.Tanh()
        )

    def make_dec_block(self, input_channels, output_channels, kernel_size=3, stride=2, padding=0, output_padding=1, final_layer=False):
    
        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride, padding, output_padding),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride, padding),
                nn.Tanh(),
            )

    def forward(self, x):
        x = self.decoder_input(x)
        x = x.view(-1, self.hidden_dim*16, 1, 1)
        x = self.dec(x)
        result = self.final_layer(x)
        return result

def get_noise(n_samples, input_dim, device='cpu'):
    return torch.randn(n_samples, input_dim, device=device)

def reparameterize(mu, log_var):
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return eps * std + mu


def get_one_hot_labels(labels, n_classes):
    return F.one_hot(labels, n_classes)

def combine_vectors(x, y):
    combined = torch.cat([x.float(), y.float()], dim=1)
    return combined

def get_input_dimensions(z_dim, mnist_shape, n_classes):
    decoder_input_dim = z_dim + n_classes
    encoder_im_chan = mnist_shape[0] + n_classes
    return decoder_input_dim, encoder_im_chan

def loss_function(fake, real, mu, log_var, kld_weight):
    # kld_weight: minibatch size    
    recons_loss =F.mse_loss(fake, real)

    kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

    loss = recons_loss + kld_weight * kld_loss
    return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':-kld_loss}