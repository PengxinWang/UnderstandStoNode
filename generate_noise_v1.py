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
from torch.utils.tensorboard import SummaryWriter

import warnings
warnings.filterwarnings("ignore")

log = logging.getLogger(__name__)

def consistency_loss(pred_bnn, pred_det, epsilon, anta=1e-1):
    """
    loss to force consistency between bnn(input) and det(input+epsilon)
    Args:
        epsilon(torch.Tensor): noise added to the input image, shape=[batch_size, *img.shape]
        anta: regularization factor to constrain noise scale
    """

    epsilon_flatten = epsilon.view(epsilon.size(0), -1)
    loss = 0.5 * cross_entropy(pred_bnn, pred_det) + 0.5 * anta * torch.norm(epsilon_flatten, p=2, dim=1).mean()
    return loss

def generate_noise(dataset, model, log_dir, input_dir, save_dir, ck_path, n_classes, in_channel, downlaod, n_component, num_iter, batch_size, lr, anta):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter(log_dir=log_dir)

    trainloader = get_dataloader(dataset=dataset,
                                 data_dir=input_dir,
                                 train=True, val=False,
                                 batch_size=batch_size,
                                 download=downlaod)

    if model=='storesnet18':
        bnn_model = StoResNet18(num_classes=n_classes, in_channels=in_channel, n_components=n_component, stochastic=1).to(device)
        det_model = StoResNet18(num_classes=n_classes, in_channels=in_channel, n_components=n_component, stochastic=2).to(device)
    else:
        raise ValueError(f'{model} not supported')

    model_dict = torch.load(ck_path)
    bnn_model.load_state_dict(model_dict)
    det_model.load_state_dict(model_dict)
    bnn_model.eval()
    det_model.eval()

    imgs_clean = []
    labels_clean = []

    imgs_noisy = []
    labels_noisy = []

    for batch_id, (imgs, labels) in enumerate(trainloader):
        imgs_clean.append(imgs)
        labels_clean.append(labels)
        imgs = imgs.to(device)

        epsilon = torch.zeros_like(imgs, device=imgs.device, dtype=imgs.dtype, requires_grad=True)
        torch.nn.init.kaiming_normal_(epsilon)
        optimizer = torch.optim.Adam([epsilon], lr=lr)

        with tqdm(total=num_iter, desc=f'Batch id {batch_id}/{len(trainloader)}') as pbar:
            for i in range(num_iter):
                with torch.no_grad():
                    bnn_pred = bnn_model(imgs).mean(dim=1).exp()
                det_pred = det_model(imgs+epsilon).mean(dim=1).exp()
                loss = consistency_loss(bnn_pred, det_pred, epsilon, anta=anta)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                writer.add_scalar('Loss/train', loss.item(), i)

                pbar.set_postfix({'Loss': f' {loss.item():.4f}'})
                pbar.set_postfix({'Loss': f' {loss.item():.4f}'})
                pbar.update()

            imgs_aug = imgs + epsilon
            imgs_noisy.append(imgs_aug.detach().cpu())
            labels_noisy.append(labels)

    imgs_noisy = torch.cat(imgs_noisy, dim=0)
    imgs_noisy = unnormalize(imgs_noisy)
    labels_noisy =  torch.cat(labels_noisy, dim=0)
    log.info(f'Noisy Dataset shape: {imgs_noisy.shape}')

    imgs_clean = torch.cat(imgs_clean, dim=0)
    imgs_clean = unnormalize(imgs_clean)
    labels_clean =  torch.cat(labels_clean, dim=0)
    log.info(f'Clean Dataset shape: {imgs_noisy.shape}')


    imgs_noisy_numpy = imgs_noisy.detach().cpu().numpy()
    
    imgs_clean_numpy = imgs_clean.detach().cpu().numpy()
    labels_clean_numpy = labels_clean.cpu().numpy()

    np.save(os.path.join(save_dir, f'noisy_imgs.npy'), imgs_noisy_numpy)
    np.save(os.path.join(save_dir, f'imgs.npy'), imgs_clean_numpy)
    np.save(os.path.join(save_dir, f'labels.npy'), labels_clean_numpy)

@hydra.main(config_path='conf_generate_noise', config_name='generate_noise_v1_config')
def main(cfg: DictConfig):
    experiment_name = cfg.experiment.name
    seed = cfg.experiment.seed
    logdir = cfg.experiment.log_dir

    dataset_name = cfg.dataset.name
    input_dir = to_absolute_path(cfg.dataset.input_dir)
    save_dir = to_absolute_path(cfg.dataset.save_dir)
    n_classes = cfg.dataset.n_classes
    in_channel = cfg.dataset.in_channel
    download = cfg.dataset.download

    model_name = cfg.model.name
    ck_dir = to_absolute_path(cfg.model.ck_dir)
    epoch = cfg.model.epoch
    n_component = cfg.model.n_components

    num_iter = cfg.params.num_iter
    batch_size = cfg.params.batch_size
    lr = cfg.params.lr
    anta = cfg.params.anta

    torch.manual_seed(seed)
    ck_path = os.path.join(ck_dir, f'storesnet18_epoch{epoch}.pt')
    os.makedirs(save_dir, exist_ok=True)

    log.info(f'Experiment: {experiment_name}')
    log.info(f'  -Seed: {seed}')
    log.info(f'  -Logdir: {logdir}')

    log.info(f'Dataset: {dataset_name}')    
    log.info(f'  -Input Directory: {input_dir}')
    log.info(f'  -Save Directory: {save_dir}') 
    log.info(f'  -Number of Classes: {n_classes}')
    log.info(f'  -Number of Input Channels: {in_channel}')
    log.info(f'  -Download: {download}')

    log.info(f'Model: {model_name}')
    log.info(f'  -Checkpoint Directory: {ck_dir}')
    log.info(f'  -Epoch: {epoch}')
    log.info(f'  -Number of Components: {n_component}')

    log.info(f'Params:')
    log.info(f'  -Number of Iterations: {num_iter}')
    log.info(f'  -Batch Size: {batch_size}')
    log.info(f'  -Learning Rate: {lr}')
    log.info(f'  -Anta (regularization factor): {anta}')

    generate_noise(dataset=dataset_name, model=model_name, log_dir=logdir, input_dir=input_dir, save_dir=save_dir, ck_path=ck_path, n_classes=n_classes, in_channel=in_channel, downlaod=download,
                   n_component=n_component, num_iter=num_iter, batch_size=batch_size, lr=lr, anta=anta)

if __name__ == '__main__':
    main()
