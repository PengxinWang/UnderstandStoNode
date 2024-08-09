import torch

from model import *
from data import get_dataloader
from utils import unnormalize

import os
import numpy as np

import warnings
warnings.filterwarnings("ignore")

def generate_noise(input_dir, save_dir, unet_ck_path, in_channels, batch_size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    trainloader = get_dataloader(data_dir=input_dir,
                                 train=True, val=False,
                                 batch_size=batch_size,)

    imgs_clean = []
    labels_clean = []

    imgs_noisy = []
    labels_noisy = []

    unet = UNet(in_channels=in_channels, out_channels=in_channels).to(device)
    model_dict = torch.load(unet_ck_path)
    unet.load_state_dict(model_dict)

    for imgs, labels in trainloader:
        imgs_clean.append(imgs)
        labels_clean.append(labels)
        labels_noisy.append(labels)
        imgs = imgs.to(device)
        imgs_noisy_batch = unet(imgs)
        imgs_noisy.append(imgs_noisy_batch.detach().cpu())

    imgs_noisy = torch.cat(imgs_noisy, dim=0)
    imgs_noisy = unnormalize(imgs_noisy).permute(0,2,3,1)

    imgs_clean = torch.cat(imgs_clean, dim=0)
    imgs_clean = unnormalize(imgs_clean).permute(0,2,3,1)
    labels_clean =  torch.cat(labels_clean, dim=0)

    imgs_noisy_numpy = imgs_noisy.detach().cpu().numpy()
    imgs_clean_numpy = imgs_clean.detach().cpu().numpy()
    labels_clean_numpy = labels_clean.cpu().numpy()

    np.save(os.path.join(save_dir, f'noisy_imgs.npy'), imgs_noisy_numpy)
    np.save(os.path.join(save_dir, f'imgs.npy'), imgs_clean_numpy)
    np.save(os.path.join(save_dir, f'labels.npy'), labels_clean_numpy)

def main():
    input_dir = f'./data/CIFAR10'
    save_dir = f'./data/CIFAR10-aug'
    in_channels = 3
    unet_ck_dir = f'./checkpoints/unet'
    unet_epoch = 90
    batch_size = 512
    unet_ck_path = os.path.join(unet_ck_dir, f'unet_epoch{unet_epoch}.pt')
    os.makedirs(save_dir, exist_ok=True)
    generate_noise(input_dir=input_dir, save_dir=save_dir, unet_ck_path=unet_ck_path, in_channels=in_channels, batch_size=batch_size)

if __name__ == '__main__':
    main()
