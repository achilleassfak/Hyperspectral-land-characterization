"""U-Net architectures used for hyperspectral crop-health segmentation."""

import torch
import torch.nn as nn


class HSI_UNet(nn.Module):
    """
    U-Net variant for hyperspectral segmentation, with BatchNorm and
    dropout added to the standard encoder/decoder blocks.
    """

    def __init__(self, in_channels, num_classes, dropout_rate=0.3):
        super().__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            )

        def bottleneck_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ELU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ELU(inplace=True),
            )

        self.enc1 = conv_block(in_channels, 64);  self.pool1 = nn.MaxPool2d(2)
        self.enc2 = conv_block(64, 128);           self.pool2 = nn.MaxPool2d(2)
        self.enc3 = conv_block(128, 256);          self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = bottleneck_block(256, 512)
        self.drop_bn    = nn.Dropout2d(dropout_rate)
        self.up3   = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3  = conv_block(512, 256);  self.drop3 = nn.Dropout2d(dropout_rate)
        self.up2   = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2  = conv_block(256, 128);  self.drop2 = nn.Dropout2d(dropout_rate)
        self.up1   = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1  = conv_block(128, 64)
        self.out_conv = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        c1 = self.enc1(x);  p1 = self.pool1(c1)
        c2 = self.enc2(p1); p2 = self.pool2(c2)
        c3 = self.enc3(p2); p3 = self.pool3(c3)
        bn = self.drop_bn(self.bottleneck(p3))
        d3 = self.drop3(self.dec3(torch.cat([self.up3(bn), c3], dim=1)))
        d2 = self.drop2(self.dec2(torch.cat([self.up2(d3), c2], dim=1)))
        d1 = self.dec1(torch.cat([self.up1(d2), c1], dim=1))
        return self.out_conv(d1)


class Original_UNet(nn.Module):
    """
    Baseline U-Net (Ronneberger et al., 2015), adapted for multi-class
    HSI input. No BatchNorm or dropout, 4 encoder levels.
    """

    def __init__(self, in_channels, num_classes):
        super().__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1), nn.ReLU(inplace=True),
            )

        self.enc1 = conv_block(in_channels, 64);  self.pool1 = nn.MaxPool2d(2)
        self.enc2 = conv_block(64, 128);           self.pool2 = nn.MaxPool2d(2)
        self.enc3 = conv_block(128, 256);          self.pool3 = nn.MaxPool2d(2)
        self.enc4 = conv_block(256, 512);          self.pool4 = nn.MaxPool2d(2)
        self.bottleneck = conv_block(512, 1024)
        self.up4  = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = conv_block(1024, 512)
        self.up3  = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = conv_block(512, 256)
        self.up2  = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = conv_block(256, 128)
        self.up1  = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = conv_block(128, 64)
        self.out_conv = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        c1 = self.enc1(x);  p1 = self.pool1(c1)
        c2 = self.enc2(p1); p2 = self.pool2(c2)
        c3 = self.enc3(p2); p3 = self.pool3(c3)
        c4 = self.enc4(p3); p4 = self.pool4(c4)
        bn = self.bottleneck(p4)
        d4 = self.dec4(torch.cat([self.up4(bn), c4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), c3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), c2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), c1], dim=1))
        return self.out_conv(d1)
