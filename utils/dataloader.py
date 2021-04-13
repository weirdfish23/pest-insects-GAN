import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelBinarizer

class ToTensor(object):
    """Covert ndarrays in sample to Tensors."""
    
    def __call__(self, sample):
        image, class_name = sample['image'], sample['class_name']

        # numpy image: [H x W x C]
        # torch image: [C x H x W]
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
               'class_name': class_name}

class InsectsDataset(Dataset):
    """Insects Dataset.
       Especies =['liriomyza huidobrensis',
                  'brevicoryne brassicae',
                  'prodiplosis longifila',
                  'trips tabaci',
                  'Bemisia tabaci',
                  'Macrolophus pygmaeus',
                  'Nesidiocoris tenuis'] 
    """
    
    def __init__(self, csv_file, root_dir, transform=None, class_name='all', return_one_hot=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        
        self.df_data = pd.read_csv(csv_file)
        # Create one-hot label encoding
        self.label_binarizer = LabelBinarizer().fit(self.df_data['especie'])
        self.return_one_hot = return_one_hot

        self.class_name = list(self.df_data['especie'].unique())
        # If class_name != 'all', filter for only one class
        if class_name != 'all':
            assert(class_name in self.class_name), "Must be a valid class_name"
            self.df_data = self.df_data[self.df_data['especie'] == class_name]
            self.df_data = self.df_data.reset_index(drop=True)
            self.class_name = [class_name]
            
        
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.df_data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_name = os.path.join(self.root_dir, 
                               self.df_data.iloc[idx, 0])
        image = io.imread(img_name)
        class_name = self.df_data.iloc[idx, 1]
        
        sample = {'image':image, 'class_name': class_name}
        
        if self.transform:
            sample = self.transform(sample)

        if self.return_one_hot:
            if type(class_name) == str:
                sample['one_hot'] = self.label_binarizer.transform([class_name])
            else:
                sample['one_hot'] = self.label_binarizer.transform(class_name.to_list())
        
        return sample

def show_samples(class_name='all'):
    insects_dataset = InsectsDataset(csv_file=path_data_csv,
                             root_dir=path_imgs,
                            class_name = class_name)
    
    fig = plt.figure(figsize=[12, 12])

    imgs_idx = np.random.randint(len(insects_dataset), size=5)

    for i, j in enumerate(imgs_idx):
        sample = insects_dataset[j]

        print(i, sample['image'].shape, sample['class_name'])

        ax = plt.subplot(2, 5, i+1)
        plt.tight_layout()
        ax.set_title('Sample #{}'.format(j))
        ax.axis('off')
        plt.imshow(sample['image'])

    plt.show()