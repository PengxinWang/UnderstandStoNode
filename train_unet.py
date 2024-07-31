import torch

from model import *
from data import get_dataloader
from utils import cross_entropy
import os
from tqdm import tqdm

import hydra
import logging
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from torch.utils.tensorboard import SummaryWriter

log = logging.getLogger(__name__)

def consistency_loss(pred_bnn, pred_det, epsilon=None, anta=1):
    """
    loss to force consistency between bnn(input) and det(input+epsilon)
    Args:
        epsilon(torch.Tensor): noise added to the input image, shape=[batch_size, *img.shape]
        anta: regularization factor to constrain noise scale
    """
    ce_loss = 0.5 * cross_entropy(pred_bnn, pred_det)
    if epsilon is not None:
        epsilon_flatten = epsilon.view(epsilon.size(0), -1) 
        l2_loss = 0.5 * anta * torch.norm(epsilon_flatten, p=2, dim=1).mean()
    else:
        l2_loss = 0.
    return ce_loss, l2_loss

def train_unet(unet_ck_dir, bnn_ck_path, input_dir, log_dir, download, n_classes, in_channels, batch_size, lr, anta, weight_decay, n_epochs, n_components):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter(log_dir=log_dir)
    trainloader, valloader = get_dataloader(data_dir=input_dir,
                                 train=True, val=True,
                                 download=download,
                                 batch_size=batch_size,
<<<<<<< HEAD
                                 train_unet_ratio=0.3)
=======
                                 train_unet=0.3)
>>>>>>> origin/main
    
    bnn_model = StoResNet18(num_classes=n_classes, in_channels=in_channels, n_components=n_components, stochastic=1).to(device)
    det_model = StoResNet18(num_classes=n_classes, in_channels=in_channels, n_components=n_components, stochastic=2).to(device)
    
    model_dict = torch.load(bnn_ck_path)
    bnn_model.load_state_dict(model_dict)
    det_model.load_state_dict(model_dict)
    bnn_model.eval()
    det_model.eval()

    unet = UNet(in_channels=in_channels, out_channels=in_channels).to(device)
    optimizer = torch.optim.Adam(unet.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(n_epochs):
        epoch_loss = 0.
        epoch_loss_val = 0.
        epoch_ce_loss = 0.
        epoch_ce_loss_val = 0.

        unet.train()
        with tqdm(total=len(trainloader), desc=f'Epoch {epoch}/{n_epochs}') as pbar:
            for batch_id, (imgs, _) in enumerate(trainloader):
                imgs = imgs.to(device)
                noisy_imgs = unet(imgs)
                epsilon = noisy_imgs - imgs
                bnn_pred = bnn_model(imgs).mean(dim=1).exp()
                det_pred = det_model(noisy_imgs).mean(dim=1).exp()
                ce_loss, l2_loss = consistency_loss(bnn_pred, det_pred, epsilon=epsilon ,anta=anta)
<<<<<<< HEAD
                loss = 0*ce_loss + l2_loss
=======
                loss = ce_loss + l2_loss
>>>>>>> origin/main
                epoch_loss += loss.item()
                epoch_ce_loss += ce_loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.set_postfix({'ce_loss': f'{ce_loss:.4f}', 'l2_loss':f'{l2_loss:4f}', 'loss': f'{epoch_loss/(1+batch_id):.4f}'})
                pbar.update()
        
        unet.eval()
        with tqdm(total=len(valloader), desc=f'Epoch {epoch}/{n_epochs}') as pbar:
            for batch_id, (imgs, _) in enumerate(valloader):
                imgs = imgs.to(device)
                noisy_imgs = unet(imgs)
                epsilon = noisy_imgs - imgs
                bnn_pred = bnn_model(imgs).mean(dim=1).exp()
                det_pred = det_model(noisy_imgs).mean(dim=1).exp()

                ce_loss_val, l2_loss_val = consistency_loss(bnn_pred, det_pred, epsilon=epsilon, anta=anta)
<<<<<<< HEAD
                loss_val = 0*ce_loss_val + l2_loss_val
=======
                loss_val = ce_loss_val + l2_loss_val
>>>>>>> origin/main
                epoch_loss_val += loss_val.item()
                epoch_ce_loss_val += ce_loss_val.item()

                pbar.set_postfix({'ce_loss_val': f'{ce_loss_val:.4f}', 'l2_loss_val':f'{l2_loss_val:4f}', 'loss_val':f'{epoch_loss_val/(1+batch_id):.4f}'})
                pbar.update()

        writer.add_scalars('Loss', {'train': epoch_loss/len(trainloader), 'val': epoch_loss_val/len(valloader)}, epoch)
        writer.add_scalars('ce loss', {'train': epoch_ce_loss/len(trainloader), 'val': epoch_ce_loss_val/len(valloader)}, epoch)

        if (epoch+1) % 10 == 0:
            unet_ck_path = os.path.join(unet_ck_dir, f'unet_epoch{epoch+1}.pt')
            torch.save(unet.state_dict(), unet_ck_path)
            log.info(f'Saved checkpoint: {unet_ck_path}')

    unet_ck_path_final = os.path.join(unet_ck_dir, f'unet_epoch{n_epochs}.pt')
    torch.save(unet.state_dict(), unet_ck_path_final)
    log.info('Training Done')
    writer.close()

@hydra.main(config_path='conf_unet', config_name='train_config')
def main(cfg: DictConfig):
    experiment_name = cfg.experiment.name
    seed = cfg.experiment.seed
    logdir = cfg.experiment.log_dir

    dataset_name = cfg.dataset.name
    input_dir = to_absolute_path(cfg.dataset.dir)
    n_classes = cfg.dataset.n_classes
    in_channels = cfg.dataset.in_channels
    download = cfg.dataset.download

    unet_ck_dir = to_absolute_path(cfg.unet.ck_dir)

    bnn_model_name = cfg.bnn.name
    bnn_ck_dir = to_absolute_path(cfg.bnn.ck_dir)
    bnn_epoch = cfg.bnn.epoch
    n_components = cfg.bnn.n_components

    n_epochs = cfg.params.n_epochs
    batch_size = cfg.params.batch_size
    lr = cfg.params.lr
    anta = cfg.params.anta
    weight_decay = cfg.params.weight_decay

    torch.manual_seed(seed)
    bnn_ck_path = os.path.join(bnn_ck_dir, f'storesnet18_epoch{bnn_epoch}.pt')
    os.makedirs(unet_ck_dir, exist_ok=True)

    log.info(f'Experiment: {experiment_name}')
    log.info(f'  -Seed: {seed}')
    log.info(f'  -Logdir: {logdir}')

    log.info(f'Dataset: {dataset_name}')    
    log.info(f'  -Input Directory: {input_dir}')  
    log.info(f'  -Number of Classes: {n_classes}')
    log.info(f'  -Number of Input Channels: {in_channels}')
    log.info(f'  -Download: {download}')

    log.info(f'Unet:')
    log.info(f'  -Unet Checkpoint Directory: {unet_ck_dir}') 

    log.info(f'bnn: {bnn_model_name}')
    log.info(f'  -Checkpoint Path: {bnn_ck_path}')
    log.info(f'  -Epoch: {bnn_epoch}')
    log.info(f'  -Number of Components: {n_components}')

    log.info(f'Params:')
    log.info(f'  -Number of Iterations: {n_epochs}')
    log.info(f'  -Batch Size: {batch_size}')
    log.info(f'  -Learning Rate: {lr}')
    log.info(f'  -Anta (regularization factor): {anta}')
    log.info(f'  -Weight Decay: {weight_decay}')

    train_unet(unet_ck_dir=unet_ck_dir, bnn_ck_path=bnn_ck_path, input_dir=input_dir, log_dir=logdir, n_components=n_components, download=download, n_classes=n_classes, in_channels=in_channels, batch_size=batch_size, lr=lr, anta=anta, weight_decay=weight_decay, n_epochs=n_epochs)

if __name__ == '__main__':
    main()
