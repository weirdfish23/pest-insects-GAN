from torch import nn
import torch.nn.functional as F
import torch

class Generator(nn.Module):
    def __init__(self, input_dim=10, im_chan=3, hidden_dim=64):
        super(Generator, self).__init__()
        self.input_dim = input_dim

        # self.gen = nn.Sequential(                                                               # (71, 1, 1)
        #     self.make_gen_block(input_dim, hidden_dim * 2, kernel_size=5, stride=2),            # (64*2, 5, 5)
        #     self.make_gen_block(hidden_dim * 2, hidden_dim, kernel_size=5, stride=4),           # (64, 21, 21)
        #     self.make_gen_block(hidden_dim, im_chan, kernel_size=4, stride=3, final_layer=True),# (3, 64, 64)
        # )

        self.gen = nn.Sequential(                                                                           # (71, 1, 1)
            self.make_gen_block(input_dim, hidden_dim * 8, kernel_size=4, stride=1, padding=0),             # (64*8, 4, 4)
            self.make_gen_block(hidden_dim * 8, hidden_dim * 4, kernel_size=4, stride=2, padding=1),        # (64*4, 8, 8)
            self.make_gen_block(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=2, padding=1),        # (64*2, 16, 16)
            self.make_gen_block(hidden_dim * 2, hidden_dim, kernel_size=4, stride=2, padding=1),            # (64, 32, 32)
            self.make_gen_block(hidden_dim, im_chan, kernel_size=4, stride=2, padding=1, final_layer=True), # (3, 64, 64)
        )

    def make_gen_block(self, input_channels, output_channels, kernel_size=3, stride=2, padding=0, final_layer=False):
    
        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride, padding),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride, padding),
                nn.Tanh(),
            )

    def forward(self, noise):
        x = noise.view(len(noise), self.input_dim, 1, 1)
        return self.gen(x)

def get_noise(n_samples, input_dim, device='cpu'):
    return torch.randn(n_samples, input_dim, device=device)

class Discriminator(nn.Module):
    def __init__(self, im_chan=3, hidden_dim=64):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(                                                              # (3, 64, 64)
            self.make_disc_block(im_chan, hidden_dim, kernel_size=4, stride=2, padding=1),                 # (64, 32, 32)
            self.make_disc_block(hidden_dim, hidden_dim * 2, kernel_size=4, stride=2, padding=1),          # (64*2, 16, 16)
            self.make_disc_block(hidden_dim * 2, hidden_dim * 4, kernel_size=4, stride=2, padding=1),      # (64*4, 8, 8)
            self.make_disc_block(hidden_dim * 4, hidden_dim * 4, kernel_size=4, stride=2, padding=1),      # (64*4, 4, 4)
            self.make_disc_block(hidden_dim * 4, 1, kernel_size=4, stride=1, padding=0, final_layer=True), # (1, 1, 1)
        )

    def make_disc_block(self, input_channels, output_channels, kernel_size=4, stride=2, padding=0, final_layer=False):
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
        disc_pred = self.disc(image)
        return disc_pred.view(len(disc_pred), -1)


def get_one_hot_labels(labels, n_classes):
    return F.one_hot(labels, n_classes)

def combine_vectors(x, y):
    combined = torch.cat([x.float(), y.float()], dim=1)
    return combined

def get_input_dimensions(z_dim, mnist_shape, n_classes):
    generator_input_dim = z_dim + n_classes
    discriminator_im_chan = mnist_shape[0] + n_classes
    return generator_input_dim, discriminator_im_chan
