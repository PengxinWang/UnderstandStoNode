import numpy as np
import matplotlib.pyplot as plt
import os

# Load the image data from the .npy files
clean_imgs_path = '/u/48/wangp8/unix/Work/exp_cifar10/data/CIFAR10-aug/imgs.npy'
noisy_imgs_path = '/u/48/wangp8/unix/Work/exp_cifar10/data/CIFAR10-aug/noisy_imgs.npy'
labels_path = '/u/48/wangp8/unix/Work/exp_cifar10/data/CIFAR10-aug/labels.npy'

beginning_index = 10000
clean_imgs = np.load(clean_imgs_path).squeeze().astype(np.float32) / 255
noisy_imgs = np.load(noisy_imgs_path).squeeze().astype(np.float32) / 255
labels = np.load(labels_path)

# CIFAR-10 class names
CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Create a directory to save the images if it doesn't exist
output_dir = 'qualitative_results'
os.makedirs(output_dir, exist_ok=True)

# Set up the matplotlib figure with a grid of subplots
fig, axs = plt.subplots(10, 3, figsize=(12, 40))

for i in range(10):
    clean_img = clean_imgs[i + beginning_index]
    noisy_img = noisy_imgs[i + beginning_index]
    noise = np.abs(noisy_img - clean_img)
    label = CIFAR10_CLASSES[labels[i + beginning_index]]

    # Plot the clean image
    axs[i, 0].imshow(clean_img)
    axs[i, 0].axis('off')
    
    # Plot the noisy image
    axs[i, 1].imshow(noisy_img)
    axs[i, 1].axis('off')
    
    # Plot the noise
    axs[i, 2].imshow(noise)
    axs[i, 2].axis('off')

# Save the figure containing all the images
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'comparison_images.png'))
plt.show()
