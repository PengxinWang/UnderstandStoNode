import torch

from model import *
from data import get_dataloader
from utils import anneal_weight
import os
from tqdm import tqdm

import hydra
import logging
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from torch.utils.tensorboard import SummaryWriter

log = logging.getLogger(__name__)

def train_unet(unet_ck_dir, bnn_ck_path, layers, beta, input_weight_initial, input_weight_final,  
               input_dir, log_dir, n_classes, in_channels, batch_size, lr, weight_decay, n_epochs, n_components):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter(log_dir=log_dir)

    trainloader, valloader = get_dataloader(data_dir=input_dir,
                                 train=True, val=True,
                                 batch_size=batch_size,
                                 train_unet_ratio=0.5)
    
    bnn_model = StoResNet18(num_classes=n_classes, in_channels=in_channels, n_components=n_components, stochastic=1).to(device)
    det_model = StoResNet18(num_classes=n_classes, in_channels=in_channels, n_components=n_components, stochastic=2).to(device)
    
    model_dict = torch.load(bnn_ck_path)
    bnn_model.load_state_dict(model_dict)
    det_model.load_state_dict(model_dict)
    sto_features = ModelFeatures(bnn_model, layers, n_components=n_components)
    det_features = ModelFeatures(det_model, layers, n_components=n_components)
    sto_features.eval()
    det_features.eval()

    unet = UNet(in_channels=in_channels, out_channels=in_channels).to(device)
    optimizer = torch.optim.Adam(unet.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(n_epochs):
        epoch_loss = 0.
        epoch_loss_val = 0.
        epoch_input_loss = 0.
        epoch_pred_loss = 0.
        epoch_pred_loss_val = 0.
        epoch_input_loss_val = 0.
        input_weight = anneal_weight(epoch, initial_weight=input_weight_initial, final_weight=input_weight_final, last_epoch=int(n_epochs*2/3)+1)
        ds_loss = DistributionShiftLoss(sto_model_features=sto_features, det_model_features=det_features, beta=beta)

        unet.train()
        with tqdm(total=len(trainloader), desc=f'Epoch {epoch}/{n_epochs}') as pbar:
            for batch_id, (imgs, _) in enumerate(trainloader):
                imgs = imgs.to(device)
                noisy_imgs = unet(imgs)
                input_loss, intermediate_loss, pred_loss = ds_loss(noisy_imgs, imgs)
                loss = input_weight * input_loss + intermediate_loss + pred_loss

                epoch_loss += loss.item()
                epoch_pred_loss += pred_loss.item()
                epoch_input_loss += input_loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.set_postfix({'pred_loss': f'{epoch_pred_loss/(1+batch_id):.4f}', 'input_loss':f'{epoch_input_loss/(1+batch_id):.4f}', 'loss': f'{epoch_loss/(1+batch_id):.4f}'})
                pbar.update()
        
        unet.eval()
        with torch.no_grad():
            with tqdm(total=len(valloader), desc=f'Epoch {epoch}/{n_epochs}') as pbar:
                for batch_id, (imgs, _) in enumerate(valloader):
                    imgs = imgs.to(device)
                    noisy_imgs = unet(imgs)
                    input_loss_val, intermediate_loss_val, pred_loss_val = ds_loss(noisy_imgs, imgs)
                    loss_val = input_weight * input_loss_val + intermediate_loss_val + pred_loss_val

                    epoch_loss_val += loss_val.item()
                    epoch_pred_loss_val += pred_loss_val.item()
                    epoch_input_loss_val += input_loss_val.item()

                    pbar.set_postfix({'pred_loss_val': f'{epoch_pred_loss_val/(1+batch_id):.4f}', 'input_loss_val':f'{epoch_input_loss_val/(1+batch_id):.4f}', 'loss_val': f'{epoch_loss_val/(1+batch_id):.4f}'})
                    pbar.update()

        writer.add_scalars('Loss', {'train': epoch_loss/len(trainloader), 'val': epoch_loss_val/len(valloader)}, epoch)
        writer.add_scalars('pred loss', {'train': epoch_pred_loss/len(trainloader), 'val': epoch_pred_loss_val/len(valloader)}, epoch)
        writer.add_scalars('input loss', {'train': epoch_input_loss/len(trainloader), 'val': epoch_input_loss_val/len(valloader)}, epoch)

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

    unet_ck_dir = to_absolute_path(cfg.unet.ck_dir)

    intermediate_layers = cfg.loss.intermediate_layers
    input_weight_initial = cfg.loss.input_weight_initial
    input_weight_final = cfg.loss.input_weight_final 
    beta = cfg.loss.beta

    bnn_model_name = cfg.bnn.name
    bnn_ck_dir = to_absolute_path(cfg.bnn.ck_dir)
    bnn_epoch = cfg.bnn.epoch
    n_components = cfg.bnn.n_components

    n_epochs = cfg.params.n_epochs
    batch_size = cfg.params.batch_size
    lr = cfg.params.lr
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

    log.info(f'Loss:')
    log.info(f'  -Layers: {intermediate_layers}')
    log.info(f'  -Beta: {beta}')
    log.info(f'  -Input Weight: {input_weight_initial} to {input_weight_final}')

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
    log.info(f'  -Weight Decay: {weight_decay}')

    train_unet(unet_ck_dir=unet_ck_dir, bnn_ck_path=bnn_ck_path, layers=intermediate_layers, beta=beta, input_weight_initial=input_weight_initial, input_weight_final=input_weight_final,
               input_dir=input_dir, log_dir=logdir, n_components=n_components, n_classes=n_classes, in_channels=in_channels, 
               batch_size=batch_size, lr=lr, weight_decay=weight_decay, n_epochs=n_epochs)

if __name__ == '__main__':
    main()
