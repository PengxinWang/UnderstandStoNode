import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

<<<<<<< HEAD
# Data
def unnormalize(tensor):
    """ Unnormalize from [0, 1] to [0, 255] """
    return (tensor * 255).clamp(0, 255).byte()

class Augmented_Dataset(Dataset):
    """
    Create customed dataset for BNN augmented data
    """
    def __init__(self, data_dir, p=0.5, transform=None):
        self.clean_imgs = np.load(os.path.join(data_dir, 'imgs.npy'))
        self.noisy_imgs = np.load(os.path.join(data_dir, 'noisy_imgs.npy'))
        self.labels = np.load(os.path.join(data_dir, 'labels.npy'))

        self.p = p
        self.transform = transform

    def __len__(self):
        return len(self.clean_imgs)

    def __getitem__(self, idx):
        if np.random.rand() < self.p:
            img = self.noisy_imgs[idx]
        else:
            img = self.clean_imgs[idx]
        
        label = torch.tensor(self.labels[idx]).long()
        if self.transform:
            img = self.transform(img)
        return img, label
    
class CorruptDataset(torch.utils.data.Dataset):
    """
    create customed dataset for corrupted data from .npy file
    """
    def __init__(self, data_dir, corrupt_types, intensity, transform=None):
        self.imgs = np.concatenate(
            [np.load(os.path.join(data_dir, f'{corrupt_type}.npy'))[(intensity-1)*10000:intensity*10000] for corrupt_type in corrupt_types], axis=0
        )
        self.labels = np.concatenate(
            [np.load(os.path.join(data_dir, 'labels.npy'))[(intensity-1)*10000:intensity*10000]] * len(corrupt_types), axis=0
        )
=======
class Customized_Dataset(Dataset):
    """
    create customized dataset in pytorch
    """
    def __init__(self, imgs, labels, transform=None):
        self.imgs = imgs
        self.labels = labels
>>>>>>> origin/main
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
<<<<<<< HEAD
        img = self.imgs[idx]
        label = torch.tensor(self.labels[idx]).long()
        if self.transform:
            img = self.transform(img)
        return img, label

# Training
=======
        x = self.imgs[idx].squeeze()
        if self.transform:
            x = self.transform(x)
        y = self.labels[idx]
        return x, y
    
class ECE(nn.Module):
    """
    expected calibration error, measure how reliable confidence score is.
    """
    def __init__(self, n_bins=15):
        super().__init__()
        bins = torch.linspace(0, 1, n_bins+1)
        self.bin_lowers = bins[:-1]
        self.bin_uppers = bins[1:]
    
    def forward(self, preds, labels):
        # preds.shape = [len(testset), n_classes], labels.shape[len(testset)], 
        # note: preds need to be normalized
        confidences, preds = torch.max(preds, dim=1)
        correct_preds = preds.eq(labels)

        ece = 0.
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            sample_in_bin = confidences.gt(bin_lower.item())*confidences.le(bin_upper.item())
            prop_in_bin = sample_in_bin.float().mean()
            if  prop_in_bin.item() > 0:
                acc_in_bin = correct_preds[sample_in_bin].float().mean()
                avg_confidence_in_bin = confidences[sample_in_bin].mean()
                diff_in_bin = torch.abs(avg_confidence_in_bin - acc_in_bin) * prop_in_bin
                ece += diff_in_bin.item()
        return ece

>>>>>>> origin/main
def anneal_weight(epoch, initial_weight=1e-2, final_weight=1e-1, last_epoch=200, mode='linear'):
    """
    Change weight for certain loss in different epoch.
    
    Args:
        epoch (int): Current epoch number.
        initial_weight (float): Weight at the start of training.
        final_weight (float): Weight at the end of the specified epochs.
        last_epoch (int): The epoch number at which the final weight is reached.

    Returns:
        float: Adjusted weight for the current epoch.
    """
    if epoch <= last_epoch:
        if mode == 'linear':
            weight = initial_weight + (final_weight - initial_weight) * (epoch / last_epoch)
        else:
            raise ValueError(f'{mode} not supported')
    else:
        weight = final_weight
    return weight

def lr_schedule(epoch, num_epochs, milestones=[0.5, 0.9], final_factor=1/3):
    """
    linear learning rate scheduling from [1.0, m1] to [lr_ratio, m2]
    """
    t = epoch/num_epochs
    m1, m2 = milestones
    if t <= m1:
        factor = 1.0
    elif t <= m2:
        factor = 1.0 - (1.0 - final_factor) * (t - m1) / (m2 - m1)
    else:
        factor = final_factor
    return factor

def cross_entropy(input, target):
    """
    Compute the cross-entropy loss between the input probabilities and target probabilities.
    Formula: sigma(target*log(input))
    
    Args:
    - input (torch.Tensor): Predicted probabilities with shape (batch_size, num_classes).
    - target (torch.Tensor): True probabilities with shape (batch_size, num_classes).
    
    Returns:
    - torch.Tensor: Mean cross-entropy loss.
    """
    input = torch.clamp(input=input, min=1e-32)
    res = -torch.sum(target * torch.log(input), dim=-1)
    return torch.mean(res)
<<<<<<< HEAD
    
# Evaluation
class ECE(nn.Module):
    """
    expected calibration error, measure how reliable confidence score is.
    """
    def __init__(self, n_bins=15):
        super().__init__()
        bins = torch.linspace(0, 1, n_bins+1)
        self.bin_lowers = bins[:-1]
        self.bin_uppers = bins[1:]
    
    def forward(self, preds, labels):
        # preds.shape = [len(testset), n_classes], labels.shape[len(testset)], 
        # note: preds need to be normalized
        confidences, preds = torch.max(preds, dim=1)
        correct_preds = preds.eq(labels)

        ece = 0.
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            sample_in_bin = confidences.gt(bin_lower.item())*confidences.le(bin_upper.item())
            prop_in_bin = sample_in_bin.float().mean()
            if  prop_in_bin.item() > 0:
                acc_in_bin = correct_preds[sample_in_bin].float().mean()
                avg_confidence_in_bin = confidences[sample_in_bin].mean()
                diff_in_bin = torch.abs(avg_confidence_in_bin - acc_in_bin) * prop_in_bin
                ece += diff_in_bin.item()
        return ece

=======
          
def unnormalize(tensor):
    """ Unnormalize from [-1, 1] to [0, 255] """
    return ((tensor + 1) * 127.5).clamp(0, 255).byte()

def mix_dataset(dataloader_1, dataloader_2, save_dir=f'./data/FashionMNIST-bnnaug'):
    """
    Mix two dataset together
    """
    imgs_mixed, labels_mixed = [], []
    for imgs, labels in dataloader_1:
        imgs_mixed.append(imgs)
        labels_mixed.append(labels)
    for imgs, labels in dataloader_2:
        imgs_mixed.append(imgs)
        labels_mixed.append(labels)
    imgs_mixed = torch.cat(imgs_mixed, axis=0).squeeze()
    labels_mixed = torch.cat(labels_mixed, axis=0)
    imgs_mixed = unnormalize(imgs_mixed)

    np.save(os.path.join(save_dir, f'imgs.npy'), imgs_mixed.numpy())
    np.save(os.path.join(save_dir, f'labels.npy'), imgs_mixed.numpy())
    print(f'mixed dataset saved')
>>>>>>> origin/main


















