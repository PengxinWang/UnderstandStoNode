from typing import Any, Callable, List, Optional, Type, Union

import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import Tensor

from .layers import *

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution: used to change num_of_channels"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def sto_conv3x3(in_planes,
               out_planes,
               stride=1,
               n_components=4,
               prior_mean=1.0,
               prior_std=0.40,
               post_mean_init=(1.0, 0.05),
               post_std_init=(0.40, 0.02),
               mode='kernel',
               ):
    """3x3 stochastic convolution with padding"""
    return StoConv2d(in_planes, out_planes, kernel_size=3, stride=stride, bias=False, padding=1,
                     n_components=n_components, prior_mean=prior_mean, 
                     prior_std=prior_std, post_mean_init=post_mean_init, post_std_init=post_std_init, mode=mode)

def sto_conv1x1(in_planes,
               out_planes,
               stride=1,
               n_components=4,
               prior_mean=1.0,
               prior_std=0.40,
               post_mean_init=(1.0, 0.05),
               post_std_init=(0.40, 0.02),
               mode='in',
               ):
    """1x1 stochastic convolution"""
    return StoConv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False,
                     n_components=n_components, prior_mean=prior_mean, 
                     prior_std=prior_std, post_mean_init=post_mean_init, post_std_init=post_std_init, mode=mode)

class StoBasicBlock(StoLayer):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
        self.conv1 = sto_conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = sto_conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor, indices: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x, indices)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out, indices)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x, indices)
        out += identity
        out = self.relu(out)
        return out

class StoSequential(nn.Sequential, StoLayer):
    def __init__(self, *args):
        super().__init__(*args)
    
    def forward(self, input, indices):
        for module in self:
            if isinstance(module, StoLayer):
                input = module(input, indices)
            else:
                input = module(input)
        return input

class StoResNet(StoModel):
    def __init__(
        self,
        block,
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        in_channels:int = 3,
        n_components=4,
        prior_mean=1.0, 
        prior_std=0.40, 
        post_mean_init=(1.0, 0.05), 
        post_std_init=(0.40, 0.02),
        mode='in',
        stochastic=1,
        n_samples=1,
        freeze_post_learning=False
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.n_components = n_components
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = conv3x3(in_channels, self.inplanes)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AvgPool2d((4, 4))
        self.T = torch.nn.Parameter(torch.ones(1) * 1.0, requires_grad=True)
        self.fc = StoLinear(512*block.expansion, num_classes, n_components=n_components, mode = "inout")
        self.n_samples = n_samples

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, StoBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
        self.sto_layers_init(n_components)
        self.stochastic = stochastic
        if self.stochastic == 0:
            self.to_determinstic()
        else:
            self.to_stochastic(stochastic_mode=self.stochastic)
        if freeze_post_learning:
            for name, param in self.named_parameters():
                if 'post_mean' in name or 'post_std' in name:
                    param.requires_grad = False
             

    def _make_layer(self, block: Type[Union[StoBasicBlock]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> StoSequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = StoSequential(
                sto_conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return StoSequential(*layers)
    
    def forward(self, x, indices=None):
        """
        input: x.shape=[batch_size, in_channel, h, w]
        output: x.shape=[batch_size, n_sample, n_classes]
        note: log_softmax is applied to model's output 
        """
        if self.n_samples > 1:
            x = torch.repeat_interleave(x, self.n_samples, dim=0)
        if indices is None:
            indices = torch.arange(x.size(0), dtype=torch.long, device=x.device) % self.n_components
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x, indices)
        x = self.layer2(x, indices)
        x = self.layer3(x, indices)
        x = self.layer4(x, indices)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x, indices)
        x = x/self.T
        x = F.log_softmax(x, dim=-1)
        x = x.view(-1, self.n_samples, x.size(1))
        return x
    
def StoResNet18(num_classes=10, in_channels=3, n_components=4, stochastic=1, prior_mean=1.0, prior_std=0.32, post_mean_init=[1.0, 0.05], post_std_init=[0.40, 0.02], n_samples=1, freeze_post_learning=False):
    return StoResNet(StoBasicBlock, [2,2,2,2], num_classes=num_classes, in_channels=in_channels, n_components=n_components, freeze_post_learning=freeze_post_learning, stochastic=stochastic, prior_mean=prior_mean, prior_std=prior_std, post_mean_init=post_mean_init, post_std_init=post_std_init, n_samples=n_samples)