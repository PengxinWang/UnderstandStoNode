import torchvision
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from utils import Augmented_Dataset, CorruptDataset
    
def get_dataloader(data_dir,
                   dataset='CIFAR10',
                   batch_size=64,
                   geo_aug=False,
                   train=True,
                   val=False,
                   val_ratio=0.2,
                   train_unet_ratio=None,
                   intensity=0):
    """
    Creates dataloaders for the specified dataset.

    Returns:
    DataLoader(s): The data loader(s) as specified.
    """
    base_transform = transforms.Compose([transforms.ToTensor()])
        
    if geo_aug:
        train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(0.05),
                transforms.RandomVerticalFlip(0.05),
                transforms.RandomAffine(translate=(0.05, 0.05), scale=(0.95, 1.15), degrees=5, shear=5),
                ])
    else:
        train_transform = base_transform

    if dataset in ('CIFAR10', 'TinyImageNet'):
        if train_unet_ratio is None:
            trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
            testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=base_transform)
        else:
            dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
            train_size = int(len(dataset)*train_unet_ratio) 
            trainset, _ = random_split(dataset, [train_size, len(dataset)-train_size]) 

    elif dataset in ('CIFAR10-bnnaug', 'TinyImageNet-bnnaug'):
        trainset = Augmented_Dataset(data_dir=data_dir, transform=train_transform)

    elif dataset in ('CIFAR10-C', 'TinyImageNet-C'):
        corrupt_types = ['saturate', 'shot_noise', 'gaussian_noise', 'zoom_blur', 'glass_blur', 'brightness', 'contrast', 'motion_blur', 'pixelate', 'snow', 'speckle_noise', 'spatter', 'gaussian_blur', 'frost', 'defocus_blur',
                        'elastic_transform', 'impulse_noise', 'jpeg_compression', 'fog']
        if intensity == 0:
            raise ValueError(f'need to indicate intensity(from 1 to 5) for corrupted dataset')
        testset = CorruptDataset(data_dir, corrupt_types=corrupt_types, intensity=intensity, transform=base_transform)

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
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
        return testloader

# def get_id_label_dict(dataset='CIFAR10'):
#     """
#     Get dictionaries for ID to label and label to ID mappings for a given dataset.
#     Args:
#         dataset (str): Name of the dataset.
#     Returns:
#         tuple: id_to_label dict and label_to_id dict.
#     Example:
#         id_to_label, label_to_id = get_id_label_dict('FashionMNIST')
#     """
#     if dataset in ('CIFAR10', 'CIFAR10-C'):
#         raise NotImplementedError
#     else:
#         raise ValueError(f'dataset {dataset} not supported')
#     id_to_label = {id: label for (id,label) in enumerate(labels)}
#     label_to_id = {label: id for (id,label) in enumerate(labels)}
#     return id_to_label, label_to_id
