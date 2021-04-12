from utils.dataloader import InsectsDataset, ToTensor
from utils.read import read_config
from utils.imgs import augment_img

from torchvision.utils import save_image
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import PIL
import os
import pandas as pd

config = read_config()
base_path = os.path.join(config['base_data']['dest_dir'], 'base')
csv_file = os.path.join(base_path, 'data_info.csv')
root_dir = os.path.join(base_path, 'images')

n_data_augmentation = config['n_data_augmentation']

new_imgs = []
lbls = []

for class_name in n_data_augmentation:
    n = n_data_augmentation[class_name]
    dataset = InsectsDataset(csv_file=csv_file, root_dir=root_dir, transform=ToTensor(), class_name=class_name)
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    
    print('Augmenting {} ...'.format(class_name))
    print('# of new augmented samples per original image: {}'.format((n)))
    for i, batch in enumerate(tqdm(dataloader)):
        img = batch['image']
        lbl = batch['class_name']
        nim, lb = augment_img(img.squeeze(), lbl=lbl[0], n=n, gray_scale=False)
        new_imgs = new_imgs + nim
        lbls = lbls + lb

new_imgs = torch.stack(new_imgs, dim=0)

filename = []
for i, lbl in enumerate(lbls):
    filename.append('img_{}.jpg'.format(i))

df_aug_data = pd.DataFrame.from_records({'filename': filename,'especie':lbls})
df_aug_data = df_aug_data[['filename', 'especie']]

dest_dir = config['base_data']['dest_dir']
path_dest_new_imgs = 'augmented'

if not path_dest_new_imgs in os.listdir(dest_dir):
    os.mkdir(os.path.join(dest_dir, path_dest_new_imgs))
    
if not 'images' in os.listdir(os.path.join(dest_dir, path_dest_new_imgs)):
    os.mkdir(os.path.join(dest_dir, path_dest_new_imgs, 'images'))
    
for i, f in enumerate(filename):
    torchvision.utils.save_image(new_imgs[i], os.path.join(dest_dir, path_dest_new_imgs, 'images', f))

df_aug_data.to_csv(os.path.join(dest_dir, path_dest_new_imgs, 'data_info.csv'), index=False)