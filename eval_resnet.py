import os
import json

import torch
import torch.nn.functional as F

from data import get_dataloader

import matplotlib.pyplot as plt
import hydra
import logging
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from model import *
from utils import ECE
import warnings
warnings.filterwarnings("ignore")

log = logging.getLogger(__name__)

def eval(model, dataset, data_dir, device, test_bsize=512, intensity=0, ece_bins=15):
    """
    Evaluates the performance of the given model on the provided test data.

    Returns:
        acc (float): Accuracy of the model on the test set.
        ece (float): Expected Calibration Error of the model on the test set.
        nll (float): Negative Log-Likelihood of the model on the test set.
    """
    testloader = get_dataloader(data_dir=data_dir, dataset=dataset,
                                batch_size=test_bsize,
                                train=False,
                                intensity=intensity)
    
    ece_eval = ECE(n_bins=ece_bins) 
    pred_total = []
    labels_total = []
    correct_count = 0
    nll_total = 0.
    size_testset = len(testloader) * test_bsize

    model.eval()
    with torch.no_grad():
        for imgs, labels in testloader:
            imgs, labels = imgs.to(device), labels.to(device)
            pred = model(imgs)
            nll = F.nll_loss(pred, labels) # the input of nll_loss should be log_probability
            nll_total += nll.item() * labels.size(0)

            pred = pred.exp()
            _, pred_id = torch.max(pred, dim=-1)
            correct_count += (pred_id==labels).sum().item()

            pred_total.append(pred)
            labels_total.append(labels)
    acc = correct_count/size_testset
    nll = nll_total/size_testset

    pred_total = torch.cat(pred_total, axis=0)
    labels_total = torch.cat(labels_total, axis=0)
    ece = ece_eval(pred_total, labels_total) # the input of ece_eval should be probability
    return acc, ece, nll

@hydra.main(config_path='conf_resnet18', config_name='eval_v2_config')
def main(cfg: DictConfig):

    dataset_name = cfg.dataset.name
    datadir_clean = to_absolute_path(cfg.dataset.dir_clean)
    datadir_corrupted = to_absolute_path(cfg.dataset.dir_corrupted)
    n_classes = cfg.dataset.n_classes
    in_channel = cfg.dataset.in_channel

    experiment_name = cfg.experiment.name
    res_dir = to_absolute_path(cfg.experiment.res_dir)
    seed = cfg.experiment.seed

    model_name = f'{cfg.model.name}_v{cfg.model.version}'
    ck_dir = to_absolute_path(cfg.model.ck_dir)

    n_epochs = cfg.params.n_epoch
    test_bsize = cfg.params.batch_size
    ece_bins = cfg.params.ece_bins
    os.makedirs(ck_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

    log.info(f'Experiment: {experiment_name}')
    log.info(f'  -Seed: {seed}')
    log.info(f'  -res_dir: {res_dir}')

    log.info(f'Dataset: {dataset_name}')
    log.info(f'  - Clean data directory: {datadir_clean}')
    log.info(f'  - Corrupted data directory: {datadir_corrupted}')
    log.info(f'  - n_classes: {n_classes}')
    log.info(f'  - in_channel: {in_channel}')
    
    log.info(f'Model: {model_name}')
    log.info(f'  - ck_dir: {cfg.model.ck_dir}')

    log.info(f'Testing with batch size: {test_bsize}, epochs: {n_epochs}, ece_bins: {ece_bins}')
    
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    accuracies = {i: [] for i in range(6)}
    eces = {i: [] for i in range(6)}
    nlls = {i: [] for i in range(6)}

    results = {}
    for epoch in n_epochs:
        model = ResNet18(num_classes=n_classes, in_channels=in_channel).to(device)
        test_ck_path = os.path.join(ck_dir, f'resnet18_epoch{epoch}.pt')
        model.load_state_dict(torch.load(test_ck_path))
        log.info(f'Evaluating model at epoch {epoch}')

        acc_clean, ece_clean, nll_clean = eval(model=model,
                                               dataset=dataset_name,
                                               data_dir=datadir_clean,
                                               test_bsize=test_bsize,
                                               device=device,
                                               ece_bins=ece_bins)
        
        accuracies[0].append(acc_clean)
        eces[0].append(ece_clean)
        nlls[0].append(nll_clean)

        if model_name not in results:
            results[model_name] = []
        results[model_name].append({
            'epoch': epoch,
            'intensity': 0,  
            'acc': acc_clean,
            'ece': ece_clean,
            'nll': nll_clean
        })

        for intensity in range(1, 6):
            acc_corrupted, ece_corrupted, nll_corrupted = eval(model=model,
                                                               dataset=f'{dataset_name}-C',
                                                               data_dir=datadir_corrupted,
                                                               test_bsize=test_bsize,
                                                               intensity=intensity,
                                                               device=device,
                                                               ece_bins=ece_bins)
            accuracies[intensity].append(acc_corrupted)
            eces[intensity].append(ece_corrupted)
            nlls[intensity].append(nll_corrupted)
            results[model_name].append({
                'epoch': epoch,
                'intensity': intensity,
                'acc': acc_corrupted,
                'ece': ece_corrupted,
                'nll': nll_corrupted
            })

        log.info("Evaluation Results:")
        log.info("+--------------------+----------+----------+----------+")
        log.info("| Dataset            | Accuracy | ECE      | NLL      |")
        log.info("+--------------------+----------+----------+----------+")
        for intensity in range(6):
            log.info(f"| Corrupted Intensity {intensity} | {accuracies[intensity][-1]:.4f} | {eces[intensity][-1]:.4f} | {nlls[intensity][-1]:.4f} |")
        log.info("+--------------------+----------+----------+----------+")


    # Save results to a JSON file
    results_file = os.path.join(res_dir, 'evaluation_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
        log.info(f'Results saved to {results_file}')

        # Plot figures for evaluation
        # Plot figures for evaluation
        intensities = list(range(1, 6))    
        plt.figure(figsize=(15, 6))
        
        # Accuracy plot
        plt.subplot(1, 3, 1)
        for k, epoch in enumerate(n_epochs):
            plt.plot(intensities, [accuracies[i][k] for i in intensities], marker='o', label=f'Epoch {epoch+1}')
        plt.xlabel('Corruption Intensity')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs. Corruption Intensity')
        plt.xticks(intensities)
        plt.legend()
        plt.grid(True)

        # ECE plot
        plt.subplot(1, 3, 2)
        for k, epoch in enumerate(n_epochs):
            plt.plot(intensities, [eces[i][k] for i in intensities], marker='o', label=f'Epoch {epoch+1}')
        plt.xlabel('Corruption Intensity')
        plt.ylabel('ECE')
        plt.title('ECE vs. Corruption Intensity')
        plt.xticks(intensities)
        plt.legend()
        plt.grid(True)

        # NLL plot
        plt.subplot(1, 3, 3)
        for k, epoch in enumerate(n_epochs):
            plt.plot(intensities, [nlls[i][k] for i in intensities], marker='o', label=f'Epoch {epoch+1}')
        plt.xlabel('Corruption Intensity')
        plt.ylabel('NLL')
        plt.title('NLL vs. Corruption Intensity')
        plt.xticks(intensities)
        plt.legend()
        plt.grid(True)

        plt.suptitle('Evaluation Results Across Epochs')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(res_dir, 'evaluation_results_all_epochs.png'))

if __name__ == "__main__":    
    main()
