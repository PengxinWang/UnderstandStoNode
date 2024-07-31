import torch

from model import *
from data import get_dataloader
from utils import cross_entropy, unnormalize

import os
import numpy as np
from tqdm import tqdm

import hydra
import logging
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
<<<<<<< HEAD
=======
from torch.utils.tensorboard import SummaryWriter
>>>>>>> origin/main

import warnings
warnings.filterwarnings("ignore")

log = logging.getLogger(__name__)

def generate_noise(input_dir, save_dir, save_dir_noisy, unet_ck_path, in_channels, downlaod, batch_size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    trainloader = get_dataloader(data_dir=input_dir,
                                 train=True, val=False,
                                 batch_size=batch_size,
                                 download=downlaod)

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
    imgs_noisy = unnormalize(imgs_noisy)
    labels_noisy =  torch.cat(labels_noisy, dim=0)
    log.info(f'Noisy Dataset shape: {imgs_noisy.shape}')

    imgs_clean = torch.cat(imgs_clean, dim=0)
    imgs_clean = unnormalize(imgs_clean)
    labels_clean =  torch.cat(labels_clean, dim=0)
    log.info(f'Clean Dataset shape: {imgs_noisy.shape}')

    imgs_noisy_numpy = imgs_noisy.detach().cpu().numpy()
<<<<<<< HEAD
    imgs_clean_numpy = imgs_clean.detach().cpu().numpy()
    labels_clean_numpy = labels_clean.cpu().numpy()

    np.save(os.path.join(save_dir, f'noisy_imgs.npy'), imgs_noisy_numpy)
    np.save(os.path.join(save_dir, f'imgs.npy'), imgs_clean_numpy)
    np.save(os.path.join(save_dir, f'labels.npy'), labels_clean_numpy)
=======
    labels_noisy_numpy = labels_noisy.cpu().numpy()
    
    imgs_clean_numpy = imgs_clean.detach().cpu().numpy()
    labels_clean_numpy = labels_clean.cpu().numpy()

    imgs_all_numpy = np.concatenate((imgs_clean_numpy, imgs_noisy_numpy), axis=0)
    labels_all_numpy = np.concatenate((labels_clean_numpy, labels_noisy_numpy), axis=0)
    log.info(f'Combined Dataset shape: {imgs_all_numpy.shape}')

    np.save(os.path.join(save_dir_noisy, f'imgs_noisy.npy'), imgs_noisy_numpy)
    np.save(os.path.join(save_dir_noisy, f'labels_noisy.npy'), labels_noisy_numpy)

    np.save(os.path.join(save_dir, f'imgs.npy'), imgs_all_numpy)
    np.save(os.path.join(save_dir, f'labels.npy'), labels_all_numpy)
>>>>>>> origin/main


@hydra.main(config_path='conf_generate_noise', config_name='generate_noise_v3_config')
def main(cfg: DictConfig):
    experiment_name = cfg.experiment.name
    seed = cfg.experiment.seed
    logdir = cfg.experiment.log_dir

    dataset_name = cfg.dataset.name
    input_dir = to_absolute_path(cfg.dataset.input_dir)
    save_dir = to_absolute_path(cfg.dataset.save_dir)
<<<<<<< HEAD
=======
    save_dir_noisy = to_absolute_path(cfg.dataset.save_dir_noisy)
>>>>>>> origin/main
    n_classes = cfg.dataset.n_classes
    in_channels = cfg.dataset.in_channels
    download = cfg.dataset.download

    unet_ck_dir = to_absolute_path(cfg.unet.ck_dir)
    unet_epoch = cfg.unet.epoch

    batch_size = cfg.params.batch_size

    torch.manual_seed(seed)
    unet_ck_path = os.path.join(unet_ck_dir, f'unet_epoch{unet_epoch}.pt')
    os.makedirs(save_dir, exist_ok=True)
<<<<<<< HEAD
=======
    os.makedirs(save_dir_noisy, exist_ok=True)
>>>>>>> origin/main

    log.info(f'Experiment: {experiment_name}')
    log.info(f'  -Seed: {seed}')
    log.info(f'  -Logdir: {logdir}')

    log.info(f'Dataset: {dataset_name}')    
    log.info(f'  -Input Directory: {input_dir}')
    log.info(f'  -Save Directory: {save_dir}')
<<<<<<< HEAD
=======
    log.info(f'  -Save Directory Noisy: {save_dir_noisy}')    
>>>>>>> origin/main
    log.info(f'  -Number of Classes: {n_classes}')
    log.info(f'  -Number of Input Channels: {in_channels}')
    log.info(f'  -Download: {download}')

    log.info(f'Unet:')
    log.info(f'  -Unet Checkpoint Path: {unet_ck_path}')

    log.info(f'Params:')
    log.info(f'  -Batch Size: {batch_size}')

<<<<<<< HEAD
    generate_noise(input_dir=input_dir, save_dir=save_dir, unet_ck_path=unet_ck_path, in_channels=in_channels, downlaod=download, batch_size=batch_size)
=======
    generate_noise(input_dir=input_dir, save_dir=save_dir, save_dir_noisy=save_dir_noisy, unet_ck_path=unet_ck_path, in_channels=in_channels, downlaod=download, batch_size=batch_size)
>>>>>>> origin/main

if __name__ == '__main__':
    main()
