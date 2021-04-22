import torch
import os
import random
import wandb
import numpy as np
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from datetime import datetime
import matplotlib.pyplot as plt

from utils.dataloader import InsectsDataset, ToTensorNorm
from utils.imgs import show_tensor_images
from utils.read import read_config
from models.conditionalVAE import Decoder, Encoder, get_noise, reparameterize, get_one_hot_labels, combine_vectors, get_input_dimensions, loss_function


# Load configuration

config = read_config()

base_path = os.path.join(config['base_data']['dest_dir'], 'augmented')
csv_file = os.path.join(base_path, 'data_info.csv')
root_dir = os.path.join(base_path, 'images')

cfg_model = config['conditional_VAE']

load_weights = cfg_model['load_weights']
weights_dir = cfg_model['weights_dir']
save_epoch = cfg_model['save_epoch']
img_shape = tuple(cfg_model['img_shape'])
n_classes = cfg_model['n_classes']
n_epochs = cfg_model['n_epochs']
z_dim = cfg_model['z_dim']
kld_weight = cfg_model['kld_weight']
display_step = cfg_model['display_step']
batch_size = cfg_model['batch_size']
lr = cfg_model['lr']
device = cfg_model['device']

print("Training config::", cfg_model)

# To save model weights

model_name = cfg_model['model_name']
weights_path = os.path.join(weights_dir, model_name+str(datetime.now().strftime('_%d_%m_%y__%H_%M_%S')))

if  model_name not in os.listdir(weights_dir):
    os.mkdir(weights_path)

# Load Dataset

dataset = InsectsDataset(csv_file=csv_file, root_dir=root_dir, transform=ToTensorNorm(), return_one_hot=True)
dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

# Instance Model

decoder_input_dim, encoder_im_chan = get_input_dimensions(z_dim, img_shape, n_classes)

dec = Decoder(input_dim=decoder_input_dim).to(device)
enc = Encoder(input_dim=encoder_im_chan).to(device)
params = list(enc.parameters()) + list(dec.parameters())
opt = torch.optim.Adam(params, lr=lr)
# opt = torch.optim.Adam(enc.parameters(), lr=lr)

# Initialize weights

def weights_init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
        if isinstance(m, nn.BatchNorm2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
            torch.nn.init.constant_(m.bias, 0)

if load_weights:
    dec_ws = torch.load(cfg_model['w_to_load']['dec'])
    enc_ws = torch.load(cfg_model['w_to_load']['enc'])
    
    dec.load_state_dict(dec_ws['state_dict'])
    opt.load_state_dict(dec_ws['optimizer'])

    enc.load_state_dict(enc_ws['state_dict'])
    opt.load_state_dict(enc_ws['optimizer'])

else:
    dec = dec.apply(weights_init)
    enc = enc.apply(weights_init)

# Initialize WandB

wandb.login()
wandb.init(project=cfg_model['model_name'], config=cfg_model)

wandb.watch(dec, log="all", log_freq=cfg_model['display_step'])
wandb.watch(enc, log="all", log_freq=cfg_model['display_step'])

# Train

cur_step = 0
losses = []
recons_losses = []
kld_losses = []

noise_and_labels = False
fake = False

fake_image_and_labels = False
real_image_and_labels = False
enc_fake_pred = False
enc_real_pred = False

for epoch in range(n_epochs):
    print("Epoch {}:".format(epoch))
    for batch in tqdm(dataloader):
        # Se obtienen las imagenes, las etiquetas y los one_hot labels para 
        real = batch['image']
        labels = batch['class_name']
        cur_batch_size = len(real)
        real = real.to(device)
        
        one_hot_labels = torch.as_tensor(batch['one_hot']).squeeze().to(device) 
        image_one_hot_labels = one_hot_labels[:, :, None, None] # agrega dos dimensiones mas (unsqueeze() ?)
        image_one_hot_labels = image_one_hot_labels.repeat(1, 1, img_shape[1], img_shape[2]) # repeat: size per dim

        enc.zero_grad()
        dec.zero_grad()

        ### Encoder
        real_image_and_labels = combine_vectors(real, image_one_hot_labels)

        mu, log_var = enc(real_image_and_labels)

        z = reparameterize(mu, log_var)

        # Se concatenan el vector latente y el vector one-hot label
        z_and_labels = combine_vectors(z, one_hot_labels)
        fake = dec(z_and_labels)
        
        obj_loss = loss_function(fake, real, mu, log_var, kld_weight)
        loss = obj_loss['loss']
        recons_loss = obj_loss['Reconstruction_Loss']
        kld_loss = obj_loss['KLD']
        
        loss.backward()
        opt.step() 

        # Se almacenan los valores del error del encoder
        losses += [loss.item()]

        # Se almacenan los valores del error del decoder
        recons_losses += [recons_loss.item()]
        kld_losses += [kld_loss.item()]
        #

        wandb.log({
            'loss': loss.item(),
            'recons_loss': recons_loss.item(),
            'kld_loss': kld_loss.item()
        }, step=cur_step)

        if cur_step % display_step == 0 and cur_step > 0:
            #dec_mean = sum(decoder_losses[-display_step:]) / display_step
            #enc_mean = sum(encoder_losses[-display_step:]) / display_step
            #print(f"Step {cur_step}: Decoder loss: {dec_mean}, encoder loss: {enc_mean}")

            if cfg_model['show_graphs']:
                show_tensor_images(fake)
                show_tensor_images(real)
                step_bins = 20
                x_axis = sorted([i * step_bins for i in range(len(decoder_losses) // step_bins)] * step_bins)
                num_examples = (len(decoder_losses) // step_bins) * step_bins
                plt.plot(
                    range(num_examples // step_bins), 
                    torch.Tensor(decoder_losses[:num_examples]).view(-1, step_bins).mean(1),
                    label="Decoder Loss"
                )
                plt.plot(
                    range(num_examples // step_bins), 
                    torch.Tensor(encoder_losses[:num_examples]).view(-1, step_bins).mean(1),
                    label="Encoder Loss"
                )
                plt.legend()
                plt.show()

            wandb.log({
                'fake_samples': wandb.Image(show_tensor_images(fake, show=False)),
                'real_samples': wandb.Image(show_tensor_images(real, show=False)) 
            }, step=cur_step)
        elif cur_step == 0:
            print("Assertions passed!!")
        cur_step += 1

    # Save weights
    if (epoch+1) % save_epoch == 0:
        state_dec = {
            'epoch': epoch,
            'state_dict': dec.state_dict(),
            'optimizer': opt.state_dict(),
        }
        torch.save(state_dec, os.path.join(weights_path, model_name+'_dec_state_epoch_{}'.format(str(epoch+1))))

        state_enc = {
            'epoch': epoch,
            'state_dict': enc.state_dict(),
            'optimizer': opt.state_dict(),
        }
        torch.save(state_enc, os.path.join(weights_path, model_name+'_enc_state_epoch_{}'.format(str(epoch+1))))

