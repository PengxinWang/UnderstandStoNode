import os
<<<<<<< HEAD
import numpy as np
import torchvision
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from utils import Augmented_Dataset, CorruptDataset
    
def get_dataloader(data_dir,
                   dataset='CIFAR10',
                   download=False,
                   batch_size=64,
=======
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
from datasets import load_dataset
from utils import Customized_Dataset
    
def get_dataloader(data_dir,
                   dataset='FashionMNIST',
                   download=False,
                   batch_size=64,
                   test_batch_size=512,
>>>>>>> origin/main
                   geo_aug=False,
                   train=True,
                   val=False,
                   val_ratio=0.2,
<<<<<<< HEAD
                   train_unet_ratio=None,
                   intensity=0):
    """
    Creates dataloaders for the specified dataset.

    Returns:
    DataLoader(s): The data loader(s) as specified.
    """
    base_transform = transforms.Compose([transforms.ToTensor()])
=======
                   train_unet=None):
    """
    Creates dataloaders for the specified dataset.
    
    Args:
    data_dir (str): The directory where the data is stored.
    batch_size (int): Batch size for the training data loader.
    test_batch_size (int): Batch size for the test data loader.
    train_unet (float): Ratio to get a subset of trainset to train unet. 
    
    Returns:
    DataLoader(s): The data loader(s) as specified.
    """
    base_transform = transforms.Compose([
            transforms.ToTensor()
        ])
>>>>>>> origin/main
        
    if geo_aug:
        train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(0.05),
                transforms.RandomVerticalFlip(0.05),
                transforms.RandomAffine(translate=(0.05, 0.05), scale=(0.95, 1.15), degrees=5, shear=5),
                ])
    else:
        train_transform = base_transform

<<<<<<< HEAD
    if dataset in ('CIFAR10', 'TinyImageNet'):
        if train_unet_ratio is None:
            trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=download, transform=train_transform)
            testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=download, transform=base_transform)
        else:
            dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=download, transform=train_transform)
            train_size = int(len(dataset)*train_unet_ratio) 
            trainset, _ = random_split(dataset, [train_size, len(dataset)-train_size]) 

    elif dataset in ('CIFAR10-bnnaug', 'TinyImageNet-bnnaug'):
=======
    if dataset == 'FashionMNIST':
        if not train_unet:
            trainset = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=train_transform)
            testset = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=base_transform)
        else:
            dataset = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=train_transform)
            train_size = int(len(dataset)*train_unet) 
            trainset, _ = random_split(dataset, [train_size, len(dataset)-train_size])

    elif dataset == 'FashionMNIST-c':
        if download:
            # download the dataset and transform to proper format
            fmnist_c = load_dataset("mweiss/fashion_mnist_corrupted")
            imgs = np.array([np.array(x) for x in fmnist_c['test']['image']])
            labels = np.array(fmnist_c['test']['label'])
            save_dir = f'./data/FashionMNIST-c'
            os.makedirs(save_dir, exist_ok=True)
            np.save(os.path.join(save_dir, f'test_imgs.npy'), imgs)
            np.save(os.path.join(save_dir, f'test_labels.npy'), labels)
            print('test data downloaded')

        imgs_path = os.path.join(data_dir, f'test_imgs.npy')
        labels_path = os.path.join(data_dir, f'test_labels.npy')
        imgs = np.load(imgs_path)
        labels = np.load(labels_path)
        testset = Customized_Dataset(imgs, labels, transform=base_transform)

    elif dataset == 'FashionMNIST-bnnaug':
>>>>>>> origin/main
        imgs_path = os.path.join(data_dir, f'imgs.npy')
        labels_path = os.path.join(data_dir, f'labels.npy')
        imgs = np.load(imgs_path)
        labels = np.load(labels_path)
<<<<<<< HEAD
        trainset = Augmented_Dataset(imgs, labels, transform=train_transform)

    elif dataset in ('CIFAR10-C', 'TinyImageNet-C'):
        corrupt_types = ['saturate', 'shot_noise', 'gaussian_noise', 'zoom_blur', 'glass_blur', 'brightness', 'contrast', 'motion_blur', 'pixelate', 'snow', 'speckle_noise', 'spatter', 'gaussian_blur', 'frost', 'defocus_blur',
                        'elastic_transform', 'impulse_noise', 'jpeg_compression', 'fog']
        if intensity == 0:
            raise ValueError(f'need to indicate intensity(from 1 to 5) for corrupted dataset')
        testset = CorruptDataset(data_dir, corrupt_types=corrupt_types, intensity=intensity, transform=base_transform)

=======
        trainset = Customized_Dataset(imgs, labels, transform=train_transform)
>>>>>>> origin/main
    else:
        raise(ValueError(f'Dataset not supported'))
            
    if train:
        if val:
            val_size = int(len(trainset) * val_ratio)
            train_size = len(trainset) - val_size
            trainset, valset = random_split(trainset, [train_size, val_size])
            trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
            valloader = DataLoader(valset, batch_size=batch_size, shuffle=False)
            return trainloader, valloader
        else:
            trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
            return trainloader
    else:
<<<<<<< HEAD
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
        return testloader

def get_id_label_dict(dataset='CIFAR10'):
    """
    Get dictionaries for ID to label and label to ID mappings for a given dataset.
    Args:
        dataset (str): Name of the dataset.
=======
        testloader = DataLoader(testset, batch_size=test_batch_size, shuffle=False)
        return testloader

def get_id_label_dict(dataset='FashionMNIST'):
    """
    Get dictionaries for ID to label and label to ID mappings for a given dataset.
    Args:
        dataset (str): Name of the dataset. Currently, only 'FashionMNIST' is supported.
>>>>>>> origin/main
    Returns:
        tuple: id_to_label dict and label_to_id dict.
    Example:
        id_to_label, label_to_id = get_id_label_dict('FashionMNIST')
    """
<<<<<<< HEAD
    if dataset in ('CIFAR10', 'CIFAR10-C'):
        raise NotImplementedError
=======
    if dataset=='FashionMNIST':
        labels = datasets.mnist.FashionMNIST.classes
>>>>>>> origin/main
    else:
        raise ValueError(f'dataset {dataset} not supported')
    id_to_label = {id: label for (id,label) in enumerate(labels)}
    label_to_id = {label: id for (id,label) in enumerate(labels)}
<<<<<<< HEAD
    return id_to_label, label_to_id
=======
    return id_to_label, label_to_id
datasets.mnist.FashionMNIST.classes
>>>>>>> origin/main
