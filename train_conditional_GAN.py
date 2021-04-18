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
from models.conditional_GAN import Generator, Discriminator, get_noise, get_one_hot_labels, combine_vectors, get_input_dimensions

torch.backends.cudnn.deterministic = True
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# Load configuration

config = read_config()

base_path = os.path.join(config['base_data']['dest_dir'], 'augmented')
csv_file = os.path.join(base_path, 'data_info.csv')
root_dir = os.path.join(base_path, 'images')

cfg_model = config['conditional_GAN']

load_weights = cfg_model['load_weights']
weights_dir = cfg_model['weights_dir']
save_epoch = cfg_model['save_epoch']
img_shape = tuple(cfg_model['img_shape'])
n_classes = cfg_model['n_classes']
n_epochs = cfg_model['n_epochs']
z_dim = cfg_model['z_dim']
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

generator_input_dim, discriminator_im_chan = get_input_dimensions(z_dim, img_shape, n_classes)

gen = Generator(input_dim=generator_input_dim).to(device)
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
disc = Discriminator(im_chan=discriminator_im_chan).to(device)
disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)

criterion = nn.BCEWithLogitsLoss()

# Initialize weights

def weights_init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
        if isinstance(m, nn.BatchNorm2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
            torch.nn.init.constant_(m.bias, 0)

if load_weights:
    gen_ws = torch.load(cfg_model['w_to_load']['gen'])
    disc_ws = torch.load(cfg_model['w_to_load']['disc'])
    
    gen.load_state_dict(gen_ws['state_dict'])
    gen_opt.load_state_dict(gen_ws['optimizer'])

    disc.load_state_dict(disc_ws['state_dict'])
    disc_opt.load_state_dict(disc_ws['optimizer'])

else:
    gen = gen.apply(weights_init)
    disc = disc.apply(weights_init)

# Initialize WandB

wandb.login()
wandb.init(project=cfg_model['model_name'], config=cfg_model)

wandb.watch(gen, criterion, log="all", log_freq=cfg_model['display_step'])
wandb.watch(disc, criterion, log="all", log_freq=cfg_model['display_step'])

# Train

cur_step = 0
generator_losses = []
discriminator_losses = []

noise_and_labels = False
fake = False

fake_image_and_labels = False
real_image_and_labels = False
disc_fake_pred = False
disc_real_pred = False

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

        ### Se Actualiza el discriminador ###
        # Limpiar las gradientes del discriminador
        disc_opt.zero_grad()
        # Obtener los vectores de ruido correspondientes al tamaño del batch
        fake_noise = get_noise(cur_batch_size, z_dim, device=device)
        
        # Se obtienen imagenes del generador
        # Se concatenan el vector de ruido y el one-hot label
        # Se generan las imágenes condicionadas en la clase
       
        noise_and_labels = combine_vectors(fake_noise, one_hot_labels)
        fake = gen(noise_and_labels)
        
        assert len(fake) == len(real)
        assert tuple(noise_and_labels.shape) == (cur_batch_size, fake_noise.shape[1] + one_hot_labels.shape[1])
        assert tuple(fake.shape) == (len(real), 3, 64, 64)

        # Se crea el input para el discriminador
        # Crear el input para el discriminador
        #     Se concatenan las imagenes falsas y reales con los one-hot labels
        #     Se obtiene la prediccion del discriminador con las imágenes falsas y reales

        fake_image_and_labels = combine_vectors(fake.detach(), image_one_hot_labels)
        real_image_and_labels = combine_vectors(real, image_one_hot_labels)
        disc_fake_pred = disc(fake_image_and_labels)
        disc_real_pred = disc(real_image_and_labels)
        
        assert tuple(fake_image_and_labels.shape) == (len(real), fake.detach().shape[1] + image_one_hot_labels.shape[1], 64 ,64)
        assert tuple(real_image_and_labels.shape) == (len(real), real.shape[1] + image_one_hot_labels.shape[1], 64 ,64)
        assert len(disc_real_pred) == len(real)
        assert torch.any(fake_image_and_labels != real_image_and_labels)
        assert tuple(fake_image_and_labels.shape) == tuple(real_image_and_labels.shape)
        assert tuple(disc_fake_pred.shape) == tuple(disc_real_pred.shape)
        
        
        disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
        disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
        disc_loss = (disc_fake_loss + disc_real_loss) / 2
        disc_loss.backward(retain_graph=True)
        disc_opt.step() 

        # Se almacenan los valores del error del discriminador
        discriminator_losses += [disc_loss.item()]

        ### Se actualiza el generador ###
        # Limpiar las gradientes del generador
        gen_opt.zero_grad()

        fake_image_and_labels = combine_vectors(fake, image_one_hot_labels)
        disc_fake_pred = disc(fake_image_and_labels)
        gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
        gen_loss.backward()
        gen_opt.step()

        # Se almacenan los valores del error del generador
        generator_losses += [gen_loss.item()]
        #

        wandb.log({
            'gen_loss': gen_loss.item(),
            'disc_loss': disc_loss.item()
        }, step=cur_step)

        if cur_step % display_step == 0 and cur_step > 0:
            gen_mean = sum(generator_losses[-display_step:]) / display_step
            disc_mean = sum(discriminator_losses[-display_step:]) / display_step
            print(f"Step {cur_step}: Generator loss: {gen_mean}, discriminator loss: {disc_mean}")

            if cfg_model['show_graphs']:
                show_tensor_images(fake)
                show_tensor_images(real)
                step_bins = 20
                x_axis = sorted([i * step_bins for i in range(len(generator_losses) // step_bins)] * step_bins)
                num_examples = (len(generator_losses) // step_bins) * step_bins
                plt.plot(
                    range(num_examples // step_bins), 
                    torch.Tensor(generator_losses[:num_examples]).view(-1, step_bins).mean(1),
                    label="Generator Loss"
                )
                plt.plot(
                    range(num_examples // step_bins), 
                    torch.Tensor(discriminator_losses[:num_examples]).view(-1, step_bins).mean(1),
                    label="Discriminator Loss"
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
        state_gen = {
            'epoch': epoch,
            'state_dict': gen.state_dict(),
            'optimizer': gen_opt.state_dict(),
        }
        torch.save(state_gen, os.path.join(weights_path, model_name+'_gen_state_epoch_{}'.format(str(epoch+1))))

        state_disc = {
            'epoch': epoch,
            'state_dict': disc.state_dict(),
            'optimizer': disc_opt.state_dict(),
        }
        torch.save(state_disc, os.path.join(weights_path, model_name+'_disc_state_epoch_{}'.format(str(epoch+1))))

