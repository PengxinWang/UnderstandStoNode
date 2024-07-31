import numpy as np
import matplotlib.pyplot as plt

# Load the image data from the .npy file
<<<<<<< HEAD
clean_imgs_path = '/u/48/wangp8/unix/Work/exp_cifar10/data/CIFAR10-aug_v1/imgs.npy'
noisy_imgs_path = '/u/48/wangp8/unix/Work/exp_cifar10/data/CIFAR10-aug_v1/noisy_imgs.npy'
clean_imgs = np.load(clean_imgs_path).squeeze()
noisy_imgs = np.load(noisy_imgs_path).squeeze()

# Verify the shape of the data
print(f"Shape of clean imgs: {clean_imgs.shape}")
print(f"Shape of clean imgs: {noisy_imgs.shape}")

# Set up the matplotlib figure and axis
fig, axes = plt.subplots(2, 3, figsize=(12, 16))

for i in range(2):
    clean_img = clean_imgs[i].transpose(1,2,0)
    noisy_img = noisy_imgs[i].transpose(1,2,0)
    noise = noisy_img - clean_img
    img_norm = np.linalg.norm(clean_img)/clean_img.size
    noise_norm = np.linalg.norm(noise)/noise.size
    print("Clean Image Min/Max:", clean_img.min(), clean_img.max())
    print("Noisy Image Min/Max:", noisy_img.min(), noisy_img.max())
    print("Noise Min/Max:", noise.min(), noise.max())

    
    # Plot the clean image
    axes[i, 0].imshow(clean_img)
=======
data_path = '/u/48/wangp8/unix/Work/BNN_resnet18/data/FashionMNIST-aug_v3/imgs.npy'
imgs = np.load(data_path).squeeze()
dataset_size = imgs.shape[0] // 2  # Assuming the dataset is split evenly between clean and noisy images

# Verify the shape of the data
print(f"Shape of imgs: {imgs.shape}")

# Set up the matplotlib figure and axis
fig, axes = plt.subplots(8, 3, figsize=(8, 16))

for i in range(8):
    clean_img = imgs[i]
    noisy_img = imgs[i + dataset_size]
    noise = noisy_img - clean_img
    img_norm = np.linalg.norm(clean_img)/clean_img.size
    noise_norm = np.linalg.norm(noise)/noise.size
    
    # Plot the clean image
    axes[i, 0].imshow(clean_img, cmap='gray')
>>>>>>> origin/main
    axes[i, 0].set_title(f'Clean Image {i+1}\nNorm: {img_norm :2f}')
    axes[i, 0].axis('off')
    
    # Plot the noisy image
<<<<<<< HEAD
    axes[i, 1].imshow(noisy_img)
=======
    axes[i, 1].imshow(noisy_img, cmap='gray')
>>>>>>> origin/main
    axes[i, 1].set_title(f'Noisy Image {i+1}')
    axes[i, 1].axis('off')
    
    # Plot the noise
<<<<<<< HEAD
    axes[i, 2].imshow(noise)
=======
    axes[i, 2].imshow(noise, cmap='gray')
>>>>>>> origin/main
    axes[i, 2].set_title(f'Noise {i+1}\nNorm: {noise_norm:.2f}')
    axes[i, 2].axis('off')

# Adjust layout
plt.tight_layout()
plt.show()
