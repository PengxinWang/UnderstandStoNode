import torch
import torch.nn as nn
from collections import OrderedDict

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=32):
        super(UNet, self).__init__()
        features = init_features
        # Define the encoder blocks
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")  # New encoder
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # New pooling layer

        # Bottleneck
        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        # Define the decoder blocks
        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)  # New upconv
        self.decoder4 = UNet._block(features * 8, features * 8, name="dec4")  # New decoder
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = UNet._block(features * 4, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = UNet._block(features * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = UNet._block(features, features, name="dec1")

        self.conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)
        self._initialize_weights()

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3)) 

        bottleneck = self.bottleneck(self.pool4(enc4))  

        dec4 = self.upconv4(bottleneck)
        if dec4.size() != enc4.size():
            dec4 = nn.functional.pad(dec4, [0, enc4.size(3) - dec4.size(3), 0, enc4.size(2) - dec4.size(2)])
        dec4 = dec4+enc4
        dec4 = self.decoder4(dec4) 

        dec3 = self.upconv3(dec4)
        if dec3.size() != enc3.size():
            dec3 = nn.functional.pad(dec3, [0, enc3.size(3) - dec3.size(3), 0, enc3.size(2) - dec3.size(2)])
        dec3 = dec3+enc3
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        if dec2.size() != enc2.size():
            dec2 = nn.functional.pad(dec2, [0, enc2.size(3) - dec2.size(3), 0, enc2.size(2) - dec2.size(2)])
        dec2 = dec2+enc2
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        if dec1.size() != enc1.size():
            dec1 = nn.functional.pad(dec1, [0, enc1.size(3) - dec1.size(3), 0, enc1.size(2) - dec1.size(2)])
        dec1 = dec1+enc1
        dec1 = self.decoder1(dec1)

        output = torch.sigmoid(self.conv(dec1))
        return output

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
