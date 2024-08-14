import torch
from model import *
from data import get_dataloader
from utils import unnormalize
import os
import numpy as np
import warnings

warnings.filterwarnings("ignore")

def generate_noise(input_dir, save_dir, unet_ck_dir, beta_list, in_channels, batch_size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the entire dataset
    trainloader = get_dataloader(data_dir=input_dir, train=True, val=False, batch_size=batch_size)

    imgs_clean = []
    labels_clean = []
    imgs_noisy = []

    # Store all the clean images and labels first
    for imgs, labels in trainloader:
        imgs_clean.append(imgs)
        labels_clean.append(labels)
    
    # Concatenate clean images and labels
    imgs_clean = torch.cat(imgs_clean, dim=0)
    labels_clean = torch.cat(labels_clean, dim=0)

    # Calculate how many samples each UNet model should process
    total_samples = len(imgs_clean)
    samples_per_model = total_samples // len(beta_list)

    # Process each part of the dataset with the corresponding UNet model
    start_idx = 0
    for beta in beta_list:
        end_idx = start_idx + samples_per_model
        imgs_part = imgs_clean[start_idx:end_idx]

        # Load the UNet model for the current beta
        unet_ck_path = os.path.join(unet_ck_dir, f'beta_{beta}.pt')
        unet = UNet(in_channels=in_channels, out_channels=in_channels).to(device)
        model_dict = torch.load(unet_ck_path)
        unet.load_state_dict(model_dict)
        unet.eval()
        print(f'Model with beta {beta} loaded.')

        # Process images in smaller batches to avoid memory issues
        batch_size_internal = 512 # You can adjust this based on your GPU memory
        for batch_start in range(0, samples_per_model, batch_size_internal):
            batch_end = min(batch_start + batch_size_internal, samples_per_model)
            imgs_batch = imgs_part[batch_start:batch_end].to(device)
            
            with torch.no_grad():
                imgs_noisy_batch = unet(imgs_batch)

            imgs_noisy.append(imgs_noisy_batch.detach().cpu())

        start_idx = end_idx

    # Concatenate and unnormalize noisy images
    imgs_noisy = torch.cat(imgs_noisy, dim=0)
    imgs_noisy = unnormalize(imgs_noisy).permute(0, 2, 3, 1)
    imgs_noisy_numpy = imgs_noisy.detach().cpu().numpy()

    # Unnormalize clean images
    imgs_clean = unnormalize(imgs_clean).permute(0, 2, 3, 1)
    imgs_clean_numpy = imgs_clean.detach().cpu().numpy()

    # Save the clean images, noisy images, and labels
    print(f'Shape of imgs_noisy: {imgs_noisy.shape}')
    print(f'Shape of imgs_clean: {imgs_clean.shape}')
    np.save(os.path.join(save_dir, f'imgs.npy'), imgs_clean_numpy)
    np.save(os.path.join(save_dir, f'noisy_imgs.npy'), imgs_noisy_numpy)
    np.save(os.path.join(save_dir, f'labels.npy'), labels_clean.numpy())

def main():
    input_dir = f'./data/CIFAR10'
    save_dir = f'./data/CIFAR10-aug'
    in_channels = 3
    batch_size = 512
    os.makedirs(save_dir, exist_ok=True)

    # Directory containing the UNet checkpoints
    unet_ck_dir = '/u/48/wangp8/unix/Work/exp_cifar10/checkpoints/beta'
    beta_list = [0.8, 1.0, 1.5]

    generate_noise(input_dir=input_dir, save_dir=save_dir, unet_ck_dir=unet_ck_dir, beta_list=beta_list, in_channels=in_channels, batch_size=batch_size)

if __name__ == '__main__':
    main()
