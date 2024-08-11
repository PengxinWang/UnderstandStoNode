import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
from model import UNet, ResNet18
from data import get_dataloader
from utils import unnormalize

def generate_noisy_data(input_dir, save_dir, unet_ck_path, in_channels, batch_size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    trainloader = get_dataloader(data_dir=input_dir, train=True, val=False, batch_size=batch_size)

    imgs_clean = []
    labels_clean = []

    imgs_noisy = []
    labels_noisy = []

    unet = UNet(in_channels=in_channels, out_channels=in_channels).to(device)
    unet.load_state_dict(torch.load(unet_ck_path))

    for imgs, labels in trainloader:
        imgs_clean.append(imgs)
        labels_clean.append(labels)
        labels_noisy.append(labels)
        imgs = imgs.to(device)
        imgs_noisy_batch = unet(imgs)
        imgs_noisy.append(imgs_noisy_batch.detach().cpu())

    imgs_noisy = torch.cat(imgs_noisy, dim=0)
    imgs_noisy = unnormalize(imgs_noisy).permute(0,2,3,1)

    imgs_clean = torch.cat(imgs_clean, dim=0)
    imgs_clean = unnormalize(imgs_clean).permute(0,2,3,1)
    labels_clean =  torch.cat(labels_clean, dim=0)

    imgs_noisy_numpy = imgs_noisy.detach().cpu().numpy()
    imgs_clean_numpy = imgs_clean.detach().cpu().numpy()
    labels_clean_numpy = labels_clean.cpu().numpy()

    np.save(os.path.join(save_dir, f'noisy_imgs.npy'), imgs_noisy_numpy)
    np.save(os.path.join(save_dir, f'imgs.npy'), imgs_clean_numpy)
    np.save(os.path.join(save_dir, f'labels.npy'), labels_clean_numpy)

def train_resnet(model, data_dir, n_classes, in_channel, ck_dir, n_epoch, lr, batch_size, weight_decay):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNet18(num_classes=n_classes, in_channels=in_channel).to(device)

    trainloader = get_dataloader(data_dir=data_dir, train=True, val=False, batch_size=batch_size)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.NLLLoss()

    for epoch in range(n_epoch):
        model.train()
        epoch_loss = 0.0

        with tqdm(total=len(trainloader), desc=f'Epoch {epoch+1}/{n_epoch}') as pbar:
            for imgs, labels in trainloader:
                imgs, labels = imgs.to(device), labels.to(device)
                optimizer.zero_grad()

                preds = model(imgs)
                loss = criterion(preds, labels)
                loss.backward()

                optimizer.step()

                epoch_loss += loss.item()
                pbar.set_postfix({'Loss': f'{epoch_loss/(pbar.n+1):.4f}'})
                pbar.update()

    os.makedirs(ck_dir, exist_ok=True)
    ck_path_final = os.path.join(ck_dir, f'resnet18_epoch{n_epoch}.pt')
    torch.save(model.state_dict(), ck_path_final)

def evaluate_resnet(model, data_dir, n_classes, in_channel):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNet18(num_classes=n_classes, in_channels=in_channel).to(device)
    model.load_state_dict(torch.load(f'{data_dir}/resnet18_epoch300.pt'))
    model.eval()

    testloader = get_dataloader(data_dir=data_dir, train=False, val=False, batch_size=128)

    correct = 0
    total = 0
    criterion = nn.NLLLoss()

    ece_loss = 0.0
    with torch.no_grad():
        for imgs, labels in testloader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loss = criterion(outputs, labels)
            ece_loss += loss.item()

    accuracy = 100 * correct / total
    ece_loss /= len(testloader)

    return accuracy, ece_loss

def main():
    input_dir = './data/CIFAR10'
    save_dir_base = './data/beta_experiment'
    in_channels = 3
    unet_ck_dir = './checkpoints/beta'
    batch_size = 512
    n_epoch = 1
    lr = 0.001
    weight_decay = 1e-4
    n_classes = 10
    beta_values = [0.5, 0.8, 1, 1.2, 1.5]

    results = []
    for beta in beta_values:
        save_dir = os.path.join(save_dir_base, f'beta_{beta}.pt')
        unet_ck_path = os.path.join(unet_ck_dir, f'beta_{beta}.pt')
        os.makedirs(save_dir, exist_ok=True)

        # Step 1: Generate Noisy Data
        generate_noisy_data(input_dir=input_dir, save_dir=save_dir, unet_ck_path=unet_ck_path, in_channels=in_channels, batch_size=batch_size)

        # Step 2: Train ResNet
        train_resnet(model=ResNet18, data_dir=save_dir, n_classes=n_classes, in_channel=in_channels,
                     ck_dir=save_dir, n_epoch=n_epoch, lr=lr, batch_size=batch_size, weight_decay=weight_decay)

        # Step 3: Evaluate ResNet
        accuracy, ece_loss = evaluate_resnet(model=ResNet18, data_dir=save_dir, n_classes=n_classes, in_channel=in_channels)

        results.append({'Model': 'ResNet18', 'Beta': beta, 'Accuracy': accuracy, 'ECE': ece_loss})

    # Output the results as a table
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    # Save results to CSV
    results_df.to_csv('./results/processed_results.csv', index=False)

    # Optional: Plot the results
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(results_df['Beta'], results_df['Accuracy'], marker='o', label='Accuracy')
    plt.xlabel('Beta')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs Beta')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(results_df['Beta'], results_df['ECE'], marker='o', label='ECE')
    plt.xlabel('Beta')
    plt.ylabel('ECE')
    plt.title('ECE vs Beta')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
