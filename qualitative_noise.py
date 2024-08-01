import numpy as np
import matplotlib.pyplot as plt

# Load the image data from the .npy file
clean_imgs_path = '/u/48/wangp8/unix/Work/exp_cifar10/data/CIFAR10-aug_v3/imgs.npy'
noisy_imgs_path = '/u/48/wangp8/unix/Work/exp_cifar10/data/CIFAR10-aug_v3/noisy_imgs.npy'
clean_imgs = np.load(clean_imgs_path).squeeze().astype(np.float32)/255
noisy_imgs = np.load(noisy_imgs_path).squeeze().astype(np.float32)/255

# Set up the matplotlib figure and axis
fig, axes = plt.subplots(4, 3, figsize=(12, 16))

for i in range(4):
    clean_img = clean_imgs[i].transpose(1,2,0)
    noisy_img = noisy_imgs[i].transpose(1,2,0)
    noise = np.abs(noisy_img - clean_img)
    pixel_x, pixel_y = 0, 0
    img_norm = np.linalg.norm(clean_img)
    noise_norm = np.linalg.norm(noise)
    print("Clean Image Min/Max:", clean_img.min(), clean_img.max())
    print("Noisy Image Min/Max:", noisy_img.min(), noisy_img.max())
    print("Noise Min/Max:", noise.min(), noise.max())

    
    # Plot the clean image
    axes[i, 0].imshow(clean_img)
    axes[i, 0].set_title(f'Clean Image {i+1}\nNorm: {img_norm :2f}')
    axes[i, 0].axis('off')
    
    # Plot the noisy image
    axes[i, 1].imshow(noisy_img)
    axes[i, 1].set_title(f'Noisy Image {i+1}')
    axes[i, 1].axis('off')
    
    # Plot the noise
    axes[i, 2].imshow(noise)
    axes[i, 2].set_title(f'Noise {i+1}\nNorm: {noise_norm:.2f}')
    axes[i, 2].axis('off')

# Adjust layout
plt.tight_layout()
plt.show()
