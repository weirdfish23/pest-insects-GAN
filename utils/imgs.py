from torchvision.utils import save_image
from torch.utils.data import DataLoader
import torch
import torchvision.transforms as transforms
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import PIL

def show_img(img):
    plt.figure(figsize=(18,15))
    npimg = img.numpy()

    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28), nrow=5, show=True):
    image_tensor = (image_tensor + 1) / 2 
    image_unflat = image_tensor.detach().cpu()
    image_grid = torchvision.utils.make_grid(image_unflat[:num_images], nrow=nrow)
    if show:
        plt.imshow(image_grid.permute(1, 2, 0).squeeze())
        plt.show()
    return image_grid.permute(1, 2, 0).squeeze()

def augment_img(img, lbl, n=24, gray_scale=False):
    base_t = transforms.Compose([transforms.ToPILImage(),transforms.ToTensor()])
    img_mean = torch.mean(img.float(), dim=(1, 2))
    img_mean = (int(img_mean[0]), int(img_mean[0]), int(img_mean[0]))
    img = base_t(img)
                                
    transform = transforms.Compose([transforms.ToPILImage(),
                                    transforms.RandomHorizontalFlip(p=0.5),
                                    # transforms.RandomApply([transforms.RandomCrop(55)], p=0.8),
                                    transforms.Resize((64, 64)),
                                    transforms.RandomRotation(degrees=180, fill=img_mean),
                                    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.3, hue=0.05),
                                    transforms.ToTensor()])
    
    to_gray_scale = base_t = transforms.Compose([transforms.ToPILImage(),
                                                 transforms.Grayscale(num_output_channels=3),
                                                 transforms.ToTensor()])
    
    if gray_scale is True:
            img = to_gray_scale(img)
    
    new_imgs = [img]
    lbls = [lbl]
    for i in range(n):
        new_img = transform(img)
        if gray_scale is True:
            new_img = to_gray_scale(new_img)
        new_imgs.append(new_img)
        lbls.append(lbl)
        
    # result = torch.cat(new_imgs, dim=0)
    # print(result.shape)
    return new_imgs, lbls