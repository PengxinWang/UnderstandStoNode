import os

import torch
import torch.optim as optim

from tqdm import tqdm
from data import get_dataloader
import hydra
import logging
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from torch.utils.tensorboard import SummaryWriter

from model import *
from utils import anneal_weight, lr_schedule
import warnings
warnings.filterwarnings("ignore")

log = logging.getLogger(__name__)

def stotrain(model, dataset, log_dir, data_dir, n_classes, in_channel, ck_dir, n_epoch, 
          lr, batch_size, weight_decay, milestones, final_factor, aug_type, n_component, n_sample, entropy_weight, kl_min, kl_max, prior_mean, prior_std, post_mean_init, post_std_init):
    """
    Trains the specified model on the given dataset and logs the training process.
    """
    writer = SummaryWriter(log_dir=log_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = StoResNet18(num_classes=n_classes, in_channels=in_channel, n_components=n_component, prior_mean=prior_mean, 
                        prior_std=prior_std, post_mean_init=post_mean_init, post_std_init=post_std_init).to(device)

    trainloader, valloader = get_dataloader(data_dir=data_dir, dataset=dataset,
                                            batch_size=batch_size,
                                            train=True, val=True,
                                            aug_type=aug_type)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: lr_schedule(epoch, n_epoch, milestones=milestones, final_factor=final_factor))

    entropy_weight = entropy_weight
    n_sample = n_sample
    size_trainset = len(trainloader) * batch_size

    for epoch in range(n_epoch):
        epoch_loss = 0.
        epoch_nll = 0.
        epoch_loss_val = 0.
        epoch_nll_val = 0.

        model.train()
        with tqdm(total=len(trainloader), desc=f'Training Epoch {epoch+1}/{n_epoch}') as pbar:
            for batch_id, (imgs, labels) in enumerate(trainloader):
                imgs, labels = imgs.to(device), labels.to(device)
                pred = model(imgs)

                nll, kl = model.vi_loss(pred, labels, n_sample, entropy_weight=entropy_weight)
                kl_weight = anneal_weight(epoch=epoch, initial_weight=kl_min, final_weight=kl_max, last_epoch=int(n_epoch*2/3)+1)
                loss = nll + kl_weight*kl/size_trainset
                epoch_loss += loss
                epoch_nll += nll

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.set_postfix({'Loss': f'{epoch_loss.item()/(batch_id+1):.4f}', 'nll': f'{epoch_nll.item()/(batch_id+1):.4f}'})
                pbar.update()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            with tqdm(total=len(valloader), desc=f'Validation Epoch {epoch+1}/{n_epoch}') as pbar:
                for batch_id, (imgs, labels) in enumerate(valloader):
                    imgs, labels = imgs.to(device), labels.to(device)
                    pred = model(imgs)

                    nll, kl = model.vi_loss(pred, labels, n_sample, entropy_weight=entropy_weight)
                    kl_weight = anneal_weight(epoch=epoch, initial_weight=kl_min, final_weight=kl_max, last_epoch=int(n_epoch*2/3)+1)
                    loss = nll + kl_weight*kl/size_trainset
                    epoch_loss_val += loss
                    epoch_nll_val += nll

                    pbar.set_postfix({'Loss_val': f'{epoch_loss_val.item()/(1+batch_id):.4f}', 'nll_val': f'{epoch_nll_val.item()/(1+batch_id):.4f}'})
                    pbar.update()

        writer.add_scalars('Loss', {'train': epoch_loss/len(trainloader), 'val': epoch_loss_val/len(valloader)}, epoch)
        writer.add_scalars('nll', {'train': epoch_nll/len(trainloader), 'val': epoch_nll_val/len(valloader)}, epoch)

        if (epoch+1) % 50 == 0:
            ck_path = os.path.join(ck_dir, f'storesnet18_epoch{epoch+1}.pt')
            torch.save(model.state_dict(), ck_path)
            log.info(f'Saved checkpoint: {ck_path}')

    ck_path_final = os.path.join(ck_dir, f'storesnet18_epoch{n_epoch}.pt')
    torch.save(model.state_dict(), ck_path_final)
    log.info('Training Done')
    writer.close()

@hydra.main(config_path='conf_storesnet18', config_name='train_storesnet_config')
def main(cfg: DictConfig):
    experiment_name = cfg.experiment.name
    log_dir = cfg.experiment.log_dir
    seed =cfg.experiment.seed

    model = cfg.model.name
    ck_dir = to_absolute_path(cfg.model.ck_dir)
    n_component = cfg.model.n_component

    dataset_name =cfg.dataset.name
    data_dir = to_absolute_path(cfg.dataset.dir)
    n_classes = cfg.dataset.n_classes
    in_channel = cfg.dataset.in_channel

    batch_size = cfg.params.batch_size
    lr = cfg.params.lr
    milestones = cfg.params.milestones
    final_factor = cfg.params.final_factor
    n_epoch = cfg.params.n_epoch
    weight_decay = cfg.params.weight_decay
    aug_type = cfg.params.aug_type
    entropy_weight = cfg.params.entropy_weight
    n_sample = cfg.params.n_sample
    kl_min = cfg.params.kl_min
    kl_max = cfg.params.kl_max
    prior_mean = cfg.params.prior_mean
    prior_std = cfg.params.prior_std
    post_mean_init = cfg.params.post_mean_init
    post_std_init = cfg.params.post_std_init

    os.makedirs(ck_dir, exist_ok=True)

    log.info(f'Experiment: {experiment_name}')
    log.info(f'  -Seed: {seed}')
    log.info(f'  -log_dir: {log_dir}')

    log.info(f'Dataset: {dataset_name}')
    log.info(f'  -Data directory: {data_dir}') 
    log.info(f'  -n_classes: {n_classes}')
    log.info(f'  -in_channel: {in_channel}')
    
    log.info(f'Model: {model}')
    log.info(f'  -ck_dir: {ck_dir}')

    log.info(f'Training with')
    log.info(f'  -batch size: {batch_size}')
    log.info(f'  -learning rate: {lr}')
    log.info(f'  -milestones: {milestones}')
    log.info(f'  -final factor: {final_factor}')
    log.info(f'  -epochs: {n_epoch}')

    log.info(f'  -data augmentation: {aug_type}')
    log.info(f'  -weight decay: {weight_decay}')

    log.info(f'  -entropy_weight: {entropy_weight}')
    log.info(f'  -n_component: {n_component}')
    log.info(f'  -n_sample: {n_sample}')
    log.info(f'  -kl_min: {kl_min}, kl_max: {kl_max}')
    log.info(f'  -prior_mean: {prior_mean} prior_std: {prior_std}')    
    log.info(f'  -post_mean_init: {post_mean_init} post_std_init: {post_std_init}')  
    
    torch.manual_seed(seed)
    stotrain(model=model,
            dataset=dataset_name,
            log_dir=log_dir,
            data_dir=data_dir,
            n_classes=n_classes,
            in_channel=in_channel,
            ck_dir=ck_dir,
            n_epoch=n_epoch,
            lr=lr,
            milestones=milestones,
            final_factor=final_factor,
            batch_size=batch_size,
            weight_decay=weight_decay,
            aug_type=aug_type,
            n_component=n_component,
            n_sample=n_sample,
            kl_min=kl_min,
            kl_max=kl_max,
            prior_mean=prior_mean,
            prior_std=prior_std,
            post_mean_init=post_mean_init,
            post_std_init=post_std_init,
            entropy_weight=entropy_weight
            )

if __name__ == "__main__":    
    main()

