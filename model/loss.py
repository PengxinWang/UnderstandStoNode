import torch
import torch.nn as nn
import torch.nn.functional as F
from .storesnet import *
from utils import cross_entropy

class ModelFeatures(nn.Module):
    def __init__(self, model, layers=None, n_components=4):
        super().__init__()
        self.features = nn.Sequential(*list(model.children())[:-2])
        self.layers = layers if layers is not None else []
        self.avgpool = model.avgpool
        self.fc = model.fc
        self.n_components = n_components

    def forward(self, x, indices=None):
        if indices is None:
            indices = torch.arange(x.size(0), dtype=torch.long, device=x.device) % self.n_components
        features = []
        features.append(('input layer', x))
        for name, layer in enumerate(self.features):
            if isinstance(layer, StoSequential):
                x = layer(x, indices)
            else:
                x = layer(x)
            if name in self.layers:
                features.append((f'layer {name}', x))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        fc_out = self.fc(x, indices)
        pred = F.softmax(fc_out, dim=-1)
        # pred: probability vector
        features.append((f'output layer', pred))
        return features

class DistributionShiftLoss(nn.Module):
    def __init__(self, sto_model_features, det_model_features, input_weight=0.2, prediction_weight=1, intermediate_weight=0.25):
        super().__init__()
        self.sto_model_features = sto_model_features
        self.det_model_features = det_model_features
        self.input_weight = input_weight
        self.intermediate_weight = intermediate_weight
        self.prediction_weight = prediction_weight

    def forward(self, noisy_imgs, clean_imgs):
        clean_features = self.sto_model_features(clean_imgs)
        noisy_features = self.det_model_features(noisy_imgs)
        
        input_loss = 0.
        pred_loss = 0.
        intermediate_loss = 0.
        for c_f, n_f in zip(clean_features, noisy_features):
            if c_f[0] == 'output layer':
                pred_loss += self.prediction_weight * cross_entropy(input=n_f[1], target=c_f[1])
            elif c_f[0] == 'input layer':
                input_loss += self.input_weight * F.mse_loss(n_f[1], c_f[1])
            else:
                intermediate_loss += self.intermediate_weight * F.mse_loss(n_f[1], c_f[1])
        return input_loss, intermediate_loss, pred_loss