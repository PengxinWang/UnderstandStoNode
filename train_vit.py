import os

import numpy as np
from einops import rearrange

import timm
import torch
import torch.nn as nn
import torch.optim as optim
from model import *
from utils import get_dataloader, TinyImageNetDataset

from tqdm import tqdm
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(42)

# Load Pretrained ViT model
model = timm.create_model('vit_tiny_patch16_224.augreg_in21k', pretrained=True)
for param in model.parameters():
    param.requires_grad = False

# Attach linear head
n_features = model.num_features
n_classes = 200
classification_head = nn.Linear(n_features, n_classes)
model.head = nn.Sequential(classification_head,
                           nn.LogSoftmax(dim=-1))
model = model.to(device)
# Load data
data_dir = f'./data/TinyImageNet'
trainloader = get_dataloader(data_dir=data_dir, dataset='TinyImageNet', batch_size=64, train=True)

# Setting optimizer
n_epochs = 10
lr = 1e-2
weight_decay = 0.
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
criterion = nn.NLLLoss()

# Training
model.train()
for epoch in range(n_epochs):
    epoch_loss = 0.
    with tqdm(total=len(trainloader), desc=f'{epoch}/{n_epochs}') as pbar:
        for batch_id, (imgs, labels) in enumerate(trainloader):
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs)
            pred_loss = criterion(preds, labels)
            epoch_loss += pred_loss

            optimizer.zero_grad()
            pred_loss.backward()
            optimizer.step()

            pbar.set_postfix({'Loss': f'{epoch_loss.item()/(batch_id+1):.4f}'})
            pbar.update()

# save model checkpoint
ck_dir = f'./checkpoints/vit'
os.makedirs(ck_dir, exist_ok=True)
ck_path = os.path.join(ck_dir, f'vit_epoch{n_epochs}.pt')
torch.save(model.state_dict(),ck_path)
print(f'Training down')

# Evaluation
testloader = get_dataloader(data_dir=data_dir, dataset='TinyImageNet', batch_size=64, train=False)
model.eval()  # Set model to evaluation mode
correct = 0
total = 0
with torch.no_grad():
    for imgs, labels in testloader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total * 100
print(f'Accuracy on the test set: {accuracy:.2f}%')