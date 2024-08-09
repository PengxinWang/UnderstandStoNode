import torch
import matplotlib.pyplot as plt
from data import get_dataloader
from model import *

# CIFAR-10 class names
CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Load the data and models
ck_path = f'checkpoints/storesnet18/storesnet18_epoch50.pt'
data_dir = f'data/CIFAR10'
dataloader = get_dataloader(data_dir=data_dir, dataset='CIFAR10', batch_size=16, train=False)
imgs, labels = next(iter(dataloader))

bnn_model = StoResNet18(stochastic=1, n_samplesw=4)
det_model = StoResNet18(stochastic=2)
model_dict = torch.load(ck_path)
bnn_model.load_state_dict(model_dict)
det_model.load_state_dict(model_dict)

bnn_model.eval()
det_model.eval()
bnn_features, det_features = {}, {}

# Register hooks
def get_features(name, features_dict):
    def hook(module, input, output):
        features_dict[name] = output.detach().cpu()
    return hook

def register_hooks(model, feature_dict):
    for name, block in model.named_children():
        if isinstance(block, StoSequential):
            block.register_forward_hook(get_features(name, feature_dict))

register_hooks(bnn_model, bnn_features)
register_hooks(det_model, det_features)

# Pass the images through the models
with torch.no_grad():
    bnn_output = bnn_model(imgs).mean(dim=1)
    det_output = det_model(imgs)

# Convert output probabilities
bnn_probabilities = torch.exp(bnn_output)
det_probabilities = torch.exp(det_output)

# Plotting function
def plot_feature_shift(imgs, labels, bnn_features, det_features, layer_names):
    num_layers = len(layer_names)
    for i, img in enumerate(imgs):
        label = labels[i].item()
        label_name = CIFAR10_CLASSES[label]
        
        fig, axs = plt.subplots(2, num_layers + 2, figsize=(14, 6))
        
        # Plot the input image
        input_img = img.permute(1, 2, 0).cpu().numpy()  # Convert to HWC for plotting
        axs[0, 0].imshow(input_img)
        axs[0, 0].set_title(f'DET Input Image: {label_name}({label})')
        axs[0, 0].axis('off')
        axs[1, 0].imshow(input_img)
        axs[1, 0].set_title(f'STO Input Image: {label_name}({label})')
        axs[1, 0].axis('off')

        for idx, layer_name in enumerate(layer_names):
            # Compute mean across feature map channels
            fmap_bnn = bnn_features[layer_name][i]
            fmap_det = det_features[layer_name][i]

            fmap_bnn_mean = torch.mean(fmap_bnn, dim=0)
            fmap_det_mean = torch.mean(fmap_det, dim=0)

            # Plot deterministic model feature map
            axs[0, idx + 1].imshow(fmap_det_mean, cmap='coolwarm')
            axs[0, idx + 1].set_title(f'Det - {layer_name}')
            axs[0, idx + 1].axis('off')

            # Plot stochastic model feature map
            axs[1, idx + 1].imshow(fmap_bnn_mean, cmap='coolwarm')
            axs[1, idx + 1].set_title(f'Sto - {layer_name}')
            axs[1, idx + 1].axis('off')

        # Plot output distributions
        bnn_probabilities_np = bnn_probabilities[i].detach().cpu().numpy().flatten()
        det_probabilities_np = det_probabilities[i].detach().cpu().numpy().flatten()
        classes = list(range(len(bnn_probabilities_np)))

        # Plot deterministic model output distribution
        axs[0, -1].bar(classes, det_probabilities_np)
        axs[0, -1].set_title('Det Output Distribution')
        axs[0, -1].set_xlabel('Class')
        axs[0, -1].set_ylabel('Probability')

        # Plot stochastic model output distribution
        axs[1, -1].bar(classes, bnn_probabilities_np)
        axs[1, -1].set_title('BNN Output Distribution')
        axs[1, -1].set_xlabel('Class')
        axs[1, -1].set_ylabel('Probability')

        plt.tight_layout()
        plt.show()
        input()  # Wait for user input before continuing to next image
        plt.close()

# List of layer names to visualize
layer_names = ['layer1', 'layer2', 'layer3', 'layer4']
plot_feature_shift(imgs, labels, bnn_features, det_features, layer_names)
