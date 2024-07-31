import os
<<<<<<< HEAD
import json
=======
>>>>>>> origin/main

import torch

from data import get_dataloader

<<<<<<< HEAD
import matplotlib.pyplot as plt
=======
>>>>>>> origin/main
import hydra
import logging
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
<<<<<<< HEAD
=======
from torch.utils.tensorboard import SummaryWriter
>>>>>>> origin/main

from model import *
from utils import ECE
import warnings
warnings.filterwarnings("ignore")

log = logging.getLogger(__name__)

<<<<<<< HEAD
def eval(model, dataset, data_dir, device, test_bsize=512, download=False, intensity=0):
    """
    Evaluates the performance of the given model on the provided test data.

=======
def eval(model, dataset, data_dir, n_classes, in_channel, ck_dir, test_epoch, test_bsize=512, download=False):
    """
    Evaluates the performance of the given model on the provided test data.

    Args:
        model (str): The model to evaluate (e.g., 'resnet18').
        dataset (str): The dataset to use for evaluation (e.g., 'FashionMNIST').
        data_dir (str): Directory where the data is stored.
        n_classes (int): Number of classes in the dataset.
        in_channel (int): Number of input channels for the model.
        ck_dir (str): Directory where the model checkpoints are stored.
        n_components (int): Number of components for posterior.
        stochastic (int): 0,1,2 for different stochastic mode.
        test_epoch (int): Epoch of the checkpoint to load for evaluation.
        test_bsize (int, optional): The batch size used in the test DataLoader. Default is 512.
        download (bool, optional): Whether to download the dataset if not present. Default is False.

>>>>>>> origin/main
    Returns:
        acc (float): Accuracy of the model on the test set.
        ece (float): Expected Calibration Error of the model on the test set.
    """
<<<<<<< HEAD
    testloader = get_dataloader(data_dir=data_dir, dataset=dataset,
                                            batch_size=test_bsize,
                                            train=False,
                                            download=download,
                                            intensity=intensity)
=======
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model == 'resnet18':
        model = ResNet18(num_classes=n_classes, in_channels=in_channel).to(device)
        test_ck_path = os.path.join(ck_dir, f'resnet18_epoch{test_epoch}.pt')
    else:
        raise ValueError(f'{model} not supported')

    model.load_state_dict(torch.load(test_ck_path))

    testloader = get_dataloader(data_dir=data_dir, dataset=dataset,
                                            batch_size=test_bsize,
                                            train=False,
                                            download=download)
>>>>>>> origin/main
    
    ece_eval = ECE(n_bins=15)
    pred_total = []
    labels_total = []
    correct_count = 0
    size_testset = len(testloader) * test_bsize

    model.eval()
    with torch.no_grad():
        for imgs, labels in testloader:
            imgs, labels = imgs.to(device), labels.to(device)
            pred = model(imgs)
            pred = pred.exp()
            _, pred_id = torch.max(pred, dim=-1)
            correct_count += (pred_id==labels).sum().item()

            pred_total.append(pred)
            labels_total.append(labels)
    acc = correct_count/size_testset
    pred_total = torch.cat(pred_total, axis=0)
    labels_total = torch.cat(labels_total, axis=0)
    ece = ece_eval(pred_total, labels_total)
    return acc, ece

@hydra.main(config_path='conf_resnet18', config_name='eval_config')
def main(cfg: DictConfig):

<<<<<<< HEAD
    dataset_name = cfg.dataset.name
=======
>>>>>>> origin/main
    datadir_clean = to_absolute_path(cfg.dataset.dir_clean)
    datadir_corrupted = to_absolute_path(cfg.dataset.dir_corrupted)
    n_classes = cfg.dataset.n_classes
    in_channel = cfg.dataset.in_channel
    download = cfg.dataset.download
<<<<<<< HEAD

    experiment_name = cfg.experiment.name
    res_dir = to_absolute_path(cfg.experiment.res_dir)
    seed = cfg.experiment.seed

    model_name = f'{cfg.model.name}_v{cfg.model.version}'
    ck_dir = to_absolute_path(cfg.model.ck_dir)

    n_epochs = cfg.params.n_epoch
    test_bsize = cfg.params.batch_size
    os.makedirs(ck_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

    log.info(f'Experiment: {experiment_name}')
    log.info(f'  -Seed: {seed}')
    log.info(f'  -res_dir: {res_dir}')

    log.info(f'Dataset: {dataset_name}')
=======
    experiment_name = cfg.experiment.name
    log_dir = cfg.experiment.log_dir
    ck_dir = to_absolute_path(cfg.model.ck_dir)
    n_epochs = cfg.params.n_epoch
    test_bsize = cfg.params.batch_size
    os.makedirs(ck_dir, exist_ok=True)

    log.info(f'Experiment: {experiment_name}')
    log.info(f'Seed: {cfg.experiment.seed}')
    log.info(f'log_dir: {cfg.experiment.log_dir}')

    log.info('Dataset:')
>>>>>>> origin/main
    log.info(f'  - Clean data directory: {datadir_clean}')
    log.info(f'  - Corrupted data directory: {datadir_corrupted}')
    log.info(f'  - n_classes: {n_classes}')
    log.info(f'  - in_channel: {in_channel}')
    
<<<<<<< HEAD
    log.info(f'Model: {model_name}')
    log.info(f'  - ck_dir: {cfg.model.ck_dir}')

    log.info(f'Testing with batch size: {test_bsize}, epochs: {n_epochs}')
    
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    accuracies = {i: [] for i in range(6)}
    eces = {i: [] for i in range(6)}

    results = {}
    for epoch in n_epochs:
        model = ResNet18(num_classes=n_classes, in_channels=in_channel).to(device)
        test_ck_path = os.path.join(ck_dir, f'resnet18_epoch{epoch}.pt')
        model.load_state_dict(torch.load(test_ck_path))
        log.info(f'Evaluating model at epoch {epoch}')

        acc_clean, ece_clean = eval(model=model,
                                    dataset=dataset_name,
                                    data_dir=datadir_clean,
                                    test_bsize=test_bsize,
                                    download=download,
                                    device=device)
        
        accuracies[0].append(acc_clean)
        eces[0].append(ece_clean)
        if model_name not in results:
            results[model_name] = []
        results[model_name].append({
            'epoch': epoch,
            'intensity': 0,  
            'acc': acc_clean,
            'ece': ece_clean
        })

        for intensity in range(1, 6):
            acc_corrupted, ece_corrupted = eval(model=model,
                                                dataset=f'{dataset_name}-C',
                                                data_dir=datadir_corrupted,
                                                test_bsize=test_bsize,
                                                download=download,
                                                intensity=intensity,
                                                device=device)
            accuracies[intensity].append(acc_corrupted)
            eces[intensity].append(ece_corrupted)
            results[model_name].append({
                'epoch': epoch,
                'intensity': intensity,
                'acc': acc_corrupted,
                'ece': ece_corrupted
            })
=======
    log.info(f'Model: {cfg.model.name}')
    log.info(f'  - ck_dir: {cfg.model.ck_dir}')
    log.info(f'Testing with batch size: {test_bsize}, epochs: {n_epochs}')
    
    torch.manual_seed(cfg.experiment.seed)
    writer = SummaryWriter(log_dir=log_dir)

    accuracies_clean = []
    accuracies_corrupted = []
    eces_clean = []
    eces_corrupted = []

    for epoch in n_epochs:
        log.info(f'Evaluating model at epoch {epoch}')
        acc_clean, ece_clean = eval(model='resnet18',
                                    dataset='FashionMNIST',
                                    data_dir=datadir_clean,
                                    n_classes=n_classes,
                                    in_channel=in_channel,
                                    ck_dir=ck_dir,
                                    test_epoch=epoch,
                                    test_bsize=test_bsize,
                                    download=download)
        
        acc_corrupted, ece_corrupted = eval(model='resnet18',
                                            dataset='FashionMNIST-c',
                                            data_dir=datadir_corrupted,
                                            n_classes=n_classes,
                                            in_channel=in_channel,
                                            ck_dir=ck_dir,
                                            test_epoch=epoch,
                                            test_bsize=test_bsize,
                                            download=download)

        accuracies_clean.append(acc_clean)
        accuracies_corrupted.append(acc_corrupted)
        eces_clean.append(ece_clean)
        eces_corrupted.append(ece_corrupted)

        hparams = {
            'epoch': epoch,
            'b_size': test_bsize}
        metrics = {
            'accuracy_clean': acc_clean,
            'ece_clean': ece_clean,
            'accuracy_corrupted': acc_corrupted,
            'ece_corrupted': ece_corrupted
        }
        writer.add_hparams(hparams, metrics)
>>>>>>> origin/main

        log.info("Evaluation Results:")
        log.info("+--------------------+----------+----------+")
        log.info("| Dataset            | Accuracy | ECE      |")
        log.info("+--------------------+----------+----------+")
<<<<<<< HEAD
        for intensity in range(6):
            log.info(f"| Corrupted Intensity {intensity} | {accuracies[intensity][-1]:.4f} | {eces[intensity][-1]:.4f} |")
        log.info("+--------------------+----------+----------+")

    # Save results to a JSON file
    results_file = os.path.join(res_dir, 'evaluation_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
        log.info(f'Results saved to {results_file}')

        # Plot figures for evaluation
        intensities = list(range(1, 6))    
        plt.figure(figsize=(12, 6))
        
        # Accuracy plot
        plt.subplot(1, 2, 1)
        for k, epoch in enumerate(n_epochs):
            plt.plot(intensities, [accuracies[i][k] for i in intensities], marker='o', label=f'Epoch {epoch+1}')
        plt.xlabel('Corruption Intensity')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs. Corruption Intensity')
        plt.xticks(intensities)
        plt.legend()
        plt.grid(True)

        # ECE plot
        plt.subplot(1, 2, 2)
        for k, epoch in enumerate(n_epochs):
            plt.plot(intensities, [eces[i][k] for i in intensities], marker='o', label=f'Epoch {epoch+1}')
        plt.xlabel('Corruption Intensity')
        plt.ylabel('ECE')
        plt.title('ECE vs. Corruption Intensity')
        plt.xticks(intensities)
        plt.legend()
        plt.grid(True)

        plt.suptitle('Evaluation Results Across Epochs')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(res_dir, 'evaluation_results_all_epochs.png'))

if __name__ == "__main__":    
    main()
=======
        log.info(f"| Clean Test Set     | {acc_clean:.4f} | {ece_clean:.4f} |")
        log.info("+--------------------+----------+----------+")
        log.info(f"| Corrupted Test Set | {acc_corrupted:.4f} | {ece_corrupted:.4f} |")
        log.info("+--------------------+----------+----------+")
    writer.close()
if __name__ == "__main__":    
    main()

>>>>>>> origin/main
