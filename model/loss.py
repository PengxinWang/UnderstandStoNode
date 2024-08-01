import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import cross_entropy

class ModelFeatures(nn.Module):
    def __init__(self, pretrained_model, layers=None, include_fc=True, include_input=True):
        super().__init__()
        self.features = nn.Sequential(*list(pretrained_model.children())[:-2])
        self.layers = layers if layers is not None else []
        self.include_fc = include_fc
        self.include_input = include_input
        if self.include_fc:
            self.avgpool = pretrained_model.avgpool
            self.fc = pretrained_model.fc

    def forward(self, x):
        features = []
        if self.include_input:
            features.append(('input layer', x))
        for name, layer in enumerate(self.features):
            x = layer(x)
            if name in self.layers:
                features.append((f'layer {name}', x))
        if self.fc:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            fc_out = self.fc(x)
            pred = fc_out.exp()
            # pred: probability vector
            features.append((f'output layer', pred))
        return features

class PerceptualLoss(nn.Module):
    def __init__(self, model_features, input_weight=0.2, prediction_weight=1, intermediate_weight=0.25):
        super().__init__()
        self.model_features = model_features
        self.input_weight = input_weight
        self.intermediate_weight = intermediate_weight
        self.prediction_weight = prediction_weight

    def forward(self, output_imgs, input_imgs):
        output_features = self.model_features(output_imgs)
        input_features = self.model_features(input_imgs)
        
        loss = 0.
        for in_f, o_f in zip(output_features, input_features):
            if in_f[0] == 'output layer':
                loss += self.prediction_weight * cross_entropy(input=in_f[1], target=o_f[1])
            elif in_f[0] == 'input layer':
                loss += self.input_weight * F.mse_loss(in_f[1], o_f[1])
            else:
                loss += self.intermediate_weight * F.mse_loss(in_f[1], o_f[1])
        return loss