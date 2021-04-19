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

class Critic(nn.Module):
    def __init__(self, im_chan=3, hidden_dim=64):
        super(Critic, self).__init__()
        self.crit = nn.Sequential(                                                              # (3, 64, 64)
            self.make_crit_block(im_chan, hidden_dim, kernel_size=4, stride=2, padding=1),                 # (64, 32, 32)
            self.make_crit_block(hidden_dim, hidden_dim * 2, kernel_size=4, stride=2, padding=1),          # (64*2, 16, 16)
            self.make_crit_block(hidden_dim * 2, hidden_dim * 4, kernel_size=4, stride=2, padding=1),      # (64*4, 8, 8)
            self.make_crit_block(hidden_dim * 4, hidden_dim * 4, kernel_size=4, stride=2, padding=1),      # (64*4, 4, 4)
            self.make_crit_block(hidden_dim * 4, 1, kernel_size=4, stride=1, padding=0, final_layer=True), # (1, 1, 1)
        )

    def make_crit_block(self, input_channels, output_channels, kernel_size=4, stride=2, padding=0, final_layer=False):
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
        crit_pred = self.crit(image)
        return crit_pred.view(len(crit_pred), -1)


def get_one_hot_labels(labels, n_classes):
    return F.one_hot(labels, n_classes)

def combine_vectors(x, y):
    combined = torch.cat([x.float(), y.float()], dim=1)
    return combined

def get_input_dimensions(z_dim, mnist_shape, n_classes):
    generator_input_dim = z_dim + n_classes
    critic_im_chan = mnist_shape[0] + n_classes
    return generator_input_dim, critic_im_chan

def get_gradient(crit, real, fake, epsilon):
    mixed_images = real * epsilon + fake * (1 - epsilon)
    mixed_scores = crit(mixed_images)
    
    gradient = torch.autograd.grad(
        inputs=mixed_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores), 
        create_graph=True,
        retain_graph=True,
    )[0]
    return gradient

def gradient_penalty(gradient):
    gradient = gradient.view(len(gradient), -1)
    gradient_norm = gradient.norm(2, dim=1)
    penalty = ((gradient_norm - 1)**2).mean()
    return penalty

def get_gen_loss(crit_fake_pred):
    gen_loss = - crit_fake_pred.mean()
    return gen_loss

def get_crit_loss(crit_fake_pred, crit_real_pred, gp, c_lambda):
    crit_loss = -( crit_real_pred - crit_fake_pred - c_lambda*gp).mean()
    return crit_loss