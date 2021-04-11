from utils.dataloader import InsectsDataset, ToTensor
from utils.read import read_config
from torchvision.utils import save_image
import os

config = read_config()
base_path = os.path.join(config['base_data']['dest_dir'], 'base')
csv_file = os.path.join(base_path, 'data_info.csv')
root_dir = os.path.join(base_path, 'images')

dataset = InsectsDataset(csv_file=csv_file, root_dir=root_dir, transform=ToTensor(), class_name='prodiplosis longifila')




# sample = dataset[0]
# print(dataset.df_data['especie'].value_counts())
# print(len(dataset))