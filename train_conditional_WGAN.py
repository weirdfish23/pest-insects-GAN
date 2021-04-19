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
from models.conditional_WGAN import Generator, Critic, get_noise, get_one_hot_labels, combine_vectors, get_input_dimensions, get_gradient, gradient_penalty, get_crit_loss, get_gen_loss


# Load configuration

config = read_config()

base_path = os.path.join(config['base_data']['dest_dir'], 'augmented')
csv_file = os.path.join(base_path, 'data_info.csv')
root_dir = os.path.join(base_path, 'images')

cfg_model = config['conditional_WGAN']

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
beta_1 = cfg_model['beta_1']
beta_2 = cfg_model['beta_2']
c_lambda = cfg_model['c_lambda']
crit_repeats = cfg_model['crit_repeats']

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

generator_input_dim, critic_im_chan = get_input_dimensions(z_dim, img_shape, n_classes)

gen = Generator(input_dim=generator_input_dim).to(device)
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr, betas=(beta_1, beta_2))
crit = Critic(im_chan=critic_im_chan).to(device)
crit_opt = torch.optim.Adam(crit.parameters(), lr=lr, betas=(beta_1, beta_2))


# Initialize weights

def weights_init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
        if isinstance(m, nn.BatchNorm2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
            torch.nn.init.constant_(m.bias, 0)

if load_weights:
    gen_ws = torch.load(cfg_model['w_to_load']['gen'])
    crit_ws = torch.load(cfg_model['w_to_load']['crit'])
    
    gen.load_state_dict(gen_ws['state_dict'])
    gen_opt.load_state_dict(gen_ws['optimizer'])

    crit.load_state_dict(crit_ws['state_dict'])
    crit_opt.load_state_dict(crit_ws['optimizer'])

else:
    gen = gen.apply(weights_init)
    crit = crit.apply(weights_init)

# Initialize WandB

wandb.login()
wandb.init(project=cfg_model['model_name'], config=cfg_model)

wandb.watch(gen, log="all", log_freq=cfg_model['display_step'])
wandb.watch(crit, log="all", log_freq=cfg_model['display_step'])

# Train

cur_step = 0
generator_losses = []
critic_losses = []

noise_and_labels = False
fake = False

fake_image_and_labels = False
real_image_and_labels = False
crit_fake_pred = False
crit_real_pred = False

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

        ### Se Actualiza el critic ###
        # Limpiar las gradientes del critic
        mean_iteration_critic_loss = 0
        for _ in range(crit_repeats):
            crit_opt.zero_grad()
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

            # Se crea el input para el critic
            # Crear el input para el critic
            #     Se concatenan las imagenes falsas y reales con los one-hot labels
            #     Se obtiene la prediccion del critic con las imágenes falsas y reales

            fake_image_and_labels = combine_vectors(fake.detach(), image_one_hot_labels)
            real_image_and_labels = combine_vectors(real, image_one_hot_labels)
            crit_fake_pred = crit(fake_image_and_labels)
            crit_real_pred = crit(real_image_and_labels)
            
            assert tuple(fake_image_and_labels.shape) == (len(real), fake.detach().shape[1] + image_one_hot_labels.shape[1], 64 ,64)
            assert tuple(real_image_and_labels.shape) == (len(real), real.shape[1] + image_one_hot_labels.shape[1], 64 ,64)
            assert len(crit_real_pred) == len(real)
            assert torch.any(fake_image_and_labels != real_image_and_labels)
            assert tuple(fake_image_and_labels.shape) == tuple(real_image_and_labels.shape)
            assert tuple(crit_fake_pred.shape) == tuple(crit_real_pred.shape)
            
            
            epsilon = torch.rand(len(real), 1, 1, 1, device=device, requires_grad=True)
            gradient = get_gradient(crit, real_image_and_labels, fake_image_and_labels, epsilon)
            gp = gradient_penalty(gradient)
            crit_loss = get_crit_loss(crit_fake_pred, crit_real_pred, gp, c_lambda)
            # Se calcula el error promedio del critic durante el batch
            mean_iteration_critic_loss += crit_loss.item() / crit_repeats
            crit_loss.backward(retain_graph=True)
            crit_opt.step() 

        # Se almacenan los valores del error del critic
        critic_losses += [mean_iteration_critic_loss]

        ### Se actualiza el generador ###
        # Limpiar las gradientes del generador
        gen_opt.zero_grad()

        # Se vuelven a generar imágenes falsas que el critic no ha visto
        fake_noise_2 = get_noise(cur_batch_size, z_dim, device=device)
        noise_and_labels_2 = combine_vectors(fake_noise_2, one_hot_labels)
        fake_2 = gen(noise_and_labels_2)
        fake_image_and_labels_2 = combine_vectors(fake_2, image_one_hot_labels)

        crit_fake_pred = crit(fake_image_and_labels_2)
        gen_loss = get_gen_loss(crit_fake_pred)
        gen_loss.backward()
        gen_opt.step()

        # Se almacenan los valores del error del generador
        generator_losses += [gen_loss.item()]

        wandb.log({
            'gen_loss': gen_loss.item(),
            'crit_loss': crit_loss.item()
        }, step=cur_step)

        if cur_step % display_step == 0 and cur_step > 0:
            gen_mean = sum(generator_losses[-display_step:]) / display_step
            crit_mean = sum(critic_losses[-display_step:]) / display_step
            print(f"Step {cur_step}: Generator loss: {gen_mean}, critic loss: {crit_mean}")

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
                    torch.Tensor(critic_losses[:num_examples]).view(-1, step_bins).mean(1),
                    label="Critic Loss"
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

        state_crit = {
            'epoch': epoch,
            'state_dict': crit.state_dict(),
            'optimizer': crit_opt.state_dict(),
        }
        torch.save(state_crit, os.path.join(weights_path, model_name+'_crit_state_epoch_{}'.format(str(epoch+1))))

