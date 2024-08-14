import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from model import *

plt.style.use('ggplot')

def extract_layer_latent_variables(model, layer_names):
    layer_stats = {}

    for layer_name in layer_names:
        post_mean_values = []
        post_std_values = []

        # Iterate through the model's named parameters to find post_mean and post_std for specific layers
        for name, param in model.named_parameters():
            if layer_name in name:
                if 'post_mean' in name:
                    post_mean_values.append(param.detach().cpu().numpy().flatten())
                elif 'post_std' in name:
                    post_std_values.append(torch.abs(param).detach().cpu().numpy().flatten())

        # Concatenate all values for the layer
        if post_mean_values:
            post_mean_values = np.concatenate(post_mean_values)
        if post_std_values:
            post_std_values = np.concatenate(post_std_values)

        layer_stats[layer_name] = {
            'post_mean': post_mean_values,
            'post_std': post_std_values
        }

    return layer_stats

def visualize_layer_distributions(layer_stats):
    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    for i, (layer_name, stats) in enumerate(layer_stats.items()):
        # Plot post_mean distribution
        sns.histplot(stats['post_mean'], kde=True, ax=axes[i, 0], color='blue', bins=20)
        # axes[i, 0].set_title(f'{layer_name} post_mean')
        axes[i, 0].get_yaxis().set_visible(False)
        axes[i, 0].set_ylabel('')

        # # Plot mean line for post_mean
        # mean_post_mean = np.mean(stats['post_mean'])
        # axes[i, 0].axvline(mean_post_mean, color='red', linestyle='dashed', linewidth=1)
        # axes[i, 0].text(mean_post_mean, axes[i, 0].get_ylim()[1] * 0.9, f'Mean: {mean_post_mean:.2f}', color='red')

        # Plot post_std distribution
        sns.histplot(stats['post_std'], kde=True, ax=axes[i, 1], color='green', bins=20)
        # axes[i, 1].set_title(f'{layer_name} post_std')
        axes[i, 1].set_ylabel('')
        axes[i, 1].get_yaxis().set_visible(False)

        # # Plot mean line for post_std
        # mean_post_std = np.mean(stats['post_std'])
        # axes[i, 1].axvline(mean_post_std, color='red', linestyle='dashed', linewidth=1)
        # axes[i, 1].text(mean_post_std, axes[i, 1].get_ylim()[1] * 0.9, f'Mean: {mean_post_std:.2f}', color='red')

    plt.tight_layout()
    plt.show()


# Assuming you have an instance of StoResNet18 model loaded
model = StoResNet18()
ck_path = f'checkpoints/storesnet18/storesnet18_epoch100.pt'
model.load_state_dict(torch.load(ck_path))

# Specify the layers you are interested in
layer_names = ['layer1', 'layer2', 'layer3', 'layer4']

# Extract the latent variables for each layer
layer_stats = extract_layer_latent_variables(model, layer_names)

# Visualize the distributions
visualize_layer_distributions(layer_stats)