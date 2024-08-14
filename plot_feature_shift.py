import torch
import matplotlib.pyplot as plt
from data import get_dataloader
from model import *

CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
plt.style.use('ggplot')

# Load the data and models
ck_path = f'checkpoints/storesnet18/storesnet18_epoch300.pt'
data_dir = f'data/CIFAR10'
b_size = 1
dataloader = get_dataloader(data_dir=data_dir, dataset='CIFAR10', batch_size=b_size, train=False)
imgs, labels = next(iter(dataloader))

bnn_model = StoResNet18(stochastic=1, n_samples=4)  # Stochastic model with 4 samples
det_model = StoResNet18(stochastic=2)  # Deterministic model
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
    bnn_output = bnn_model(imgs)
    det_output = det_model(imgs)

# Convert output probabilities
bnn_probabilities = torch.exp(bnn_output)
det_probabilities = torch.exp(det_output)

# Plotting function
def plot_feature_shift(imgs, labels, bnn_features, det_features, layer_names, bnn_probabilities, det_probabilities):
    num_layers = len(layer_names)
    n_samples = bnn_features[layer_names[0]].shape[0] // b_size  # Number of samples in BNN

    for i, img in enumerate(imgs):
        label = labels[i].item()
        label_name = CIFAR10_CLASSES[label]
        
        # Plot the input image separately
        plt.figure(figsize=(6, 6))
        input_img = img.permute(1, 2, 0).cpu().numpy()
        plt.imshow(input_img)
        plt.title(f'Input Image: {label_name} ({label})')
        plt.axis('off')
        # plt.show()

        fig, axs = plt.subplots(n_samples + 2, num_layers + 1, figsize=(16, 16))

        # Plot output distributions
        bnn_probabilities_np = bnn_probabilities[i].squeeze().detach().cpu().numpy()
        bnn_probabilities_mean_np = torch.mean(bnn_probabilities[i], dim=0).squeeze().detach().cpu().numpy()
        det_probabilities_np = det_probabilities[i].squeeze().detach().cpu().numpy()
        classes = list(range(len(det_probabilities_np)))

        axs[0, -1].bar(classes, det_probabilities_np)
        # axs[0, -1].set_title('Det Output')
        axs[0, -1].set_xticks(range(10))
        axs[0, -1].set_xticklabels([str(i) for i in range(10)])
        # axs[0, -1].set_xlabel('Class')
        axs[0, -1].get_yaxis().set_visible(False)  # Hide y-axis

        axs[-1, -1].bar(classes, bnn_probabilities_mean_np)
        # axs[-1, -1].set_title('Sto Output(mean)')
        axs[-1, -1].set_xticks(range(10))
        axs[-1, -1].set_xticklabels([str(i) for i in range(10)])
        # axs[-1, -1].set_xlabel('Class')
        axs[-1, -1].get_yaxis().set_visible(False)  # Hide y-axis

        for sample_idx in range(n_samples):
            axs[sample_idx+1, -1].bar(classes, bnn_probabilities_np[sample_idx])
            # axs[sample_idx+1, -1].set_title(f'BNN sample {sample_idx}')
            axs[sample_idx+1, -1].set_xticks(range(10))
            axs[sample_idx+1, -1].set_xticklabels([str(i) for i in range(10)])
            # axs[sample_idx+1, -1].set_xlabel('Class')
            axs[sample_idx+1, -1].get_yaxis().set_visible(False)  # Hide y-axis
            for idx, layer_name in enumerate(layer_names):
                fmap_bnn = bnn_features[layer_name][sample_idx * b_size + i]
                fmap_det = det_features[layer_name][i]

                fmap_bnn_mean = torch.mean(fmap_bnn, dim=0)
                fmap_det_mean = torch.mean(fmap_det, dim=0)

                if sample_idx == 0:  # Plot deterministic model's feature maps in the first row
                    axs[0, idx].imshow(fmap_det_mean, cmap='coolwarm')
                    axs[0, idx].axis('off')

                axs[sample_idx + 1, idx].imshow(fmap_bnn_mean, cmap='coolwarm')
                # axs[sample_idx + 1, idx].set_title(f'Sto Sample {sample_idx + 1}')
                axs[sample_idx + 1, idx].axis('off')

                # Compute and plot mean feature maps for stochastic samples
                if sample_idx == n_samples - 1:
                    mean_fmap_bnn = torch.mean(torch.mean(bnn_features[layer_name], dim=1), dim=0)
                    axs[-1, idx].imshow(mean_fmap_bnn, cmap='coolwarm')
                    axs[-1, idx].axis('off')

        plt.tight_layout()
        plt.show()

# List of layer names to visualize
layer_names = ['layer1', 'layer2', 'layer3', 'layer4']
plot_feature_shift(imgs, labels, bnn_features, det_features, layer_names, bnn_probabilities, det_probabilities)
