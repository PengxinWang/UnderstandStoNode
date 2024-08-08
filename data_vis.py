import os
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import random

# Define the path to the datasets
cifar10_data_dir = '/u/48/wangp8/unix/Work/exp_cifar10/data/CIFAR10'
cifar10_c_data_dir = '/u/48/wangp8/unix/Work/exp_cifar10/data/CIFAR10-C'

# CIFAR-10 label names
cifar10_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Load CIFAR-10 test dataset (original images)
transform = transforms.ToTensor()
cifar10_testset = torchvision.datasets.CIFAR10(root=cifar10_data_dir, train=False, download=False, transform=transform)

# Corruption types to visualize
corruption_types = ['brightness', 'fog', 'frost', 'snow', 'spatter', 'motion_blur']

# Severity levels to visualize
severity_levels = [1, 3, 5]

def plot_corruptions_grid(original_image, corruption_images, corruption_types, severity_levels):
    """
    Function to plot all corruptions and severities in a grid.
    """
    num_corruptions = len(corruption_types)
    num_severities = len(severity_levels)
    
    fig, axes = plt.subplots(num_severities, num_corruptions, figsize=(15, 15))
    
    # Plot the corrupted images
    for i, severity in enumerate(severity_levels):
        for j, corruption_type in enumerate(corruption_types):
            axes[i, j].imshow(corruption_images[severity][corruption_type])
            # axes[i, j].set_title(f"{corruption_type.capitalize()}\nSeverity {severity}")
            axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.show()

def visualize_all_corruptions(cifar10_testset, corruption_types, severity_levels, cifar10_c_data_dir, img_index=None):
    """
    Function to visualize all corruption types with their severity levels.
    """
    # Select an image by index or randomly if not provided
    if img_index is None:
        img_index = random.randint(0, len(cifar10_testset) - 1)
        
    img, label = cifar10_testset[img_index]
    img_np = img.numpy().transpose(1, 2, 0)

    corruption_images = {severity: {} for severity in severity_levels}
    
    for corruption_type in corruption_types:
        corruption_path = os.path.join(cifar10_c_data_dir, f'{corruption_type}.npy')
        corrupted_data = np.load(corruption_path)
        
        for severity in severity_levels:
            start_idx = (severity - 1) * 10000
            corrupted_img = corrupted_data[start_idx + img_index]  # Index corresponding to the same test image
            corruption_images[severity][corruption_type] = corrupted_img
    
    # Plot the original image separately
    plt.figure(figsize=(4, 4))
    plt.imshow(img_np)
    # plt.title(f"Original Image: {cifar10_labels[label]}")
    plt.axis('off')
    plt.show()

    # Plot all corruptions and severities in a grid
    plot_corruptions_grid(img_np, corruption_images, corruption_types, severity_levels)

# Specify the index of the image you want to visualize (or None for random selection)
img_index = 4

# Call the function to visualize all corruptions and severities
visualize_all_corruptions(cifar10_testset, corruption_types, severity_levels, cifar10_c_data_dir, 168)
