import os
import glob
from PIL import Image
import numpy as np

import torch
import torchvision
import torch.utils
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split

# Data
def unnormalize(tensor):
    """ Unnormalize from [0, 1] to [0, 255] """
    return (tensor * 255).clamp(0, 255).byte()

class Augmented_Dataset(Dataset):
    """
    Create customed dataset for BNN augmented data
    """
    def __init__(self, data_dir, p=0.865, transform=None):
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
        if corrupt_types is None:
            raise ValueError(f'Corrupt types should be indicated')
        self.imgs = np.concatenate(
            [np.load(os.path.join(data_dir, f'{corrupt_type}.npy'))[(intensity-1)*10000:intensity*10000] for corrupt_type in corrupt_types], axis=0
        )
        self.labels = np.concatenate(
            [np.load(os.path.join(data_dir, 'labels.npy'))[(intensity-1)*10000:intensity*10000]] * len(corrupt_types), axis=0
        )
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        label = torch.tensor(self.labels[idx]).long()
        if self.transform:
            img = self.transform(img)
        return img, label

class TinyImageNetDataset(torch.utils.data.Dataset):
    """
    create customed dataset for TinyImageNet dataset
    n_classes = 200
    --data_dir
      --train
        --nid/images/nid_index.JPEG
      --val
        --images/val_index.JPEG
        --val_annotations.txt: val_index.JPEG <--> nid
      --test
        --images/test_index.JPEG (no label)
      --wnids.txt: nids occured in TinyImageNet
      --words.txt: nid <--> readable label
    """
    def __init__(self, data_dir, transform=None, train=True):
        self.wnid_to_id = {}
        self.train = train
        self.transform = transform
        with open(os.path.join(data_dir, 'wnids.txt'), 'r') as f:
            for i, line in enumerate(f):
                nid = line.strip()
                self.wnid_to_id[nid] = i

        if train:
            self.filenames = glob.glob(os.path.join(data_dir, 'train/*/images/*.JPEG'))
            self.labels = [self.wnid_to_id[os.path.basename(os.path.dirname(os.path.dirname(fp)))] 
                           for fp in self.filenames]
        else:
            val_images_dir = os.path.join(data_dir, 'val/images')
            self.filenames = []
            self.labels = []
            val_annotations_file = os.path.join(data_dir, 'val/val_annotations.txt')
            with open(val_annotations_file, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split('\t')
                    img_name, wnid = parts[0], parts[1]
                    self.filenames.append(os.path.join(val_images_dir, img_name))
                    self.labels.append(self.wnid_to_id[wnid])
    def __len__(self):
        return len(self.filenames)
        
    def __getitem__(self, idx):
        img = Image.open(self.filenames[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label
    
    @staticmethod
    def get_id_to_name_mapping(data_dir):
        wnid_to_id = {}
        id_to_name = {}
        with open(os.path.join(data_dir, 'wnids.txt'), 'r') as f:
            for i, line in enumerate(f):
                nid = line.strip()
                wnid_to_id[nid] = i
        with open(os.path.join(data_dir, 'words.txt'), 'r') as f:
            for line in f:
                nid, name = line.strip().split('\t')
                if nid in wnid_to_id:
                    id = wnid_to_id[nid]
                    id_to_name[id] = name
        return id_to_name

class CorruptTinyImageNetDataset(torch.utils.data.Dataset):
    """
    create customed dataset for corrupted TinyImageNet dataset.
    """
    def __init__(self, data_dir, corrupt_types, intensity, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.filenames = []
        self.labels = []
        self.corruption_types = corrupt_types

        for corrupt_type in corrupt_types:
            corrupt_dir = os.path.join(data_dir, corrupt_type, str(intensity))
            class_dirs = [d for d in os.listdir(corrupt_dir) if os.path.isdir(os.path.join(corrupt_dir, d))]
            for class_dir in class_dirs:
                class_idx = self.get_class_index(class_dir)
                if class_idx is not None:
                    image_paths = glob.glob(os.path.join(corrupt_dir, class_dir, '*.JPEG'))
                    self.filenames.extend(image_paths)
                    self.labels.extend([class_idx] * len(image_paths))

    def get_class_index(self, class_dir):
        wnids_path = os.path.join(os.path.dirname(self.data_dir), 'wnids.txt')
        with open(wnids_path, 'r') as f:
            wnids = [line.strip() for line in f]
        if class_dir in wnids:
            return wnids.index(class_dir)
        return None

    def __len__(self):
        return len(self.filenames)
        
    def __getitem__(self, idx):
        print(f'check running {idx}')
        img = Image.open(self.filenames[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

def get_dataloader(data_dir,
                   dataset='CIFAR10',
                   batch_size=64,
                   aug_type=None,
                   train=True,
                   val=False,
                   val_ratio=0.1,
                   train_unet_ratio=None,
                   intensity=0,
                   corrupt_types=None):
    """
    Creates dataloaders for the specified dataset.
    dataset currently supported: CIFAR10, CIFAR100, TinyImageNet, CIFAR10-C, CIFAR100-C, TinyImageNet-C
    """
    if 'ImageNet' in dataset:
        base_transform = transforms.Compose([transforms.Resize((224, 224)),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                         ])
    else:
        base_transform = transforms.Compose([transforms.ToTensor(),])
        
    if aug_type == 'geometric':
        train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(0.05),
                transforms.RandomVerticalFlip(0.05),
                transforms.RandomAffine(translate=(0.05, 0.05), scale=(0.95, 1.15), degrees=5, shear=5),
                transforms.RandomResizedCrop(size=(32, 32), scale=(0.9, 1.0)), 
                ])      
    elif aug_type == 'gaussian':
        train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(0.05),
                transforms.RandomVerticalFlip(0.05),
                transforms.RandomAffine(translate=(0.05, 0.05), scale=(0.95, 1.15), degrees=5, shear=5),
                transforms.RandomApply([transforms.Lambda(lambda x: x + 0.05 * torch.randn_like(x))], p=0.1),
                transforms.RandomResizedCrop(size=(32, 32), scale=(0.9, 1.0)), 
                ])
    elif aug_type == 'full':
        train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(0.05),
                transforms.RandomVerticalFlip(0.05),
                transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.15), shear=5),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.1),
                transforms.RandomApply([transforms.Lambda(lambda x: x + 0.05 * torch.randn_like(x))], p=0.1),
                transforms.RandomResizedCrop(size=(32, 32), scale=(0.9, 1.0)), 
                ])
    else:
        train_transform = base_transform

    if dataset == 'CIFAR10':
        if train_unet_ratio is None:
            trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
            testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=base_transform)
        else:
            dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
            train_size = int(len(dataset)*train_unet_ratio) 
            trainset, _ = random_split(dataset, [train_size, len(dataset)-train_size]) 
    
    elif dataset == 'CIFAR100':
        if train_unet_ratio is None:
            trainset = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True, transform=train_transform)
            testset = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True, transform=base_transform)
        else:
            dataset = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True, transform=train_transform)
            train_size = int(len(dataset)*train_unet_ratio) 
            trainset, _ = random_split(dataset, [train_size, len(dataset)-train_size]) 

    elif dataset == 'TinyImageNet':
        if train_unet_ratio is None:
            trainset = TinyImageNetDataset(data_dir=data_dir, transform=train_transform)
            testset = TinyImageNetDataset(data_dir=data_dir,  transform=base_transform, train=False)
        else:
            dataset = TinyImageNetDataset(root=data_dir, train=True, transform=train_transform)
            train_size = int(len(dataset)*train_unet_ratio) 
            trainset, _ = random_split(dataset, [train_size, len(dataset)-train_size]) 
    
    elif dataset == 'ImageNet-1k':
        raise NotImplementedError()

    elif dataset == 'CIFAR10-C':
        if intensity == 0:
            raise ValueError(f'need to indicate intensity(from 1 to 5) for corrupted dataset')
        testset = CorruptDataset(data_dir, corrupt_types=corrupt_types, intensity=intensity, transform=base_transform)
    
    elif dataset == 'CIFAR100-C':
        if intensity == 0:
            raise ValueError(f'need to indicate intensity(from 1 to 5) for corrupted dataset')
        testset = CorruptDataset(data_dir, corrupt_types=corrupt_types, intensity=intensity, transform=base_transform)

    elif dataset == 'TinyImageNet-C':
        if intensity == 0:
            raise ValueError(f'need to indicate intensity(from 1 to 5) for corrupted dataset')
        testset = CorruptTinyImageNetDataset(data_dir, corrupt_types=corrupt_types, intensity=intensity, transform=base_transform)

    elif dataset in ('CIFAR10-bnnaug', 'TinyImageNet-bnnaug'):
        trainset = Augmented_Dataset(data_dir=data_dir, transform=train_transform)

    else:
        raise(ValueError(f'Dataset not supported'))
            
    if train:
        if val:
            val_size = int(len(trainset) * val_ratio)
            train_size = len(trainset) - val_size
            trainset, valset = random_split(trainset, [train_size, val_size])
            trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
            valloader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=4)
            return trainloader, valloader
        else:
            trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
            return trainloader
    else:
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
        return testloader

# Training
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
        # note: preds need to be normalized to [0,1]
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



















