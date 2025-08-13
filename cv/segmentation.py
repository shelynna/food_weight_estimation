import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_resnet50

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch,out_ch,3,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch,out_ch,3,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self,x): return self.net(x)

class UNet(nn.Module):
    def __init__(self, in_ch=3, n_classes=2, base=32):
        super().__init__()
        self.d1 = DoubleConv(in_ch, base)
        self.d2 = DoubleConv(base, base*2)
        self.d3 = DoubleConv(base*2, base*4)
        self.d4 = DoubleConv(base*4, base*8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(base*8, base*16)
        self.up4 = nn.ConvTranspose2d(base*16, base*8,2,2)
        self.c4 = DoubleConv(base*16, base*8)
        self.up3 = nn.ConvTranspose2d(base*8, base*4,2,2)
        self.c3 = DoubleConv(base*8, base*4)
        self.up2 = nn.ConvTranspose2d(base*4, base*2,2,2)
        self.c2 = DoubleConv(base*4, base*2)
        self.up1 = nn.ConvTranspose2d(base*2, base,2,2)
        self.c1 = DoubleConv(base*2, base)
        self.outc = nn.Conv2d(base, n_classes,1)
    def forward(self,x):
        d1 = self.d1(x)
        d2 = self.d2(self.pool(d1))
        d3 = self.d3(self.pool(d2))
        d4 = self.d4(self.pool(d3))
        bn = self.bottleneck(self.pool(d4))
        u4 = self.up4(bn)
        u4 = self.c4(torch.cat([u4,d4],1))
        u3 = self.up3(u4)
        u3 = self.c3(torch.cat([u3,d3],1))
        u2 = self.up2(u3)
        u2 = self.c2(torch.cat([u2,d2],1))
        u1 = self.up1(u2)
        u1 = self.c1(torch.cat([u1,d1],1))
        return self.outc(u1)

def build_segmentation_model(name="unet", in_channels=3, num_classes=2):
    if name == "unet":
        return UNet(in_ch=in_channels, n_classes=num_classes)
    elif name == "deeplabv3":
        model = deeplabv3_resnet50(weights="DEFAULT")
        model.classifier[-1] = torch.nn.Conv2d(256, num_classes, 1)
        return model
    else:
        raise ValueError(name)

def dice_loss(pred, target, eps=1e-6):
    pred = torch.softmax(pred, dim=1)
    target_oh = torch.nn.functional.one_hot(target, pred.shape[1]).permute(0,3,1,2).float()
    inter = (pred*target_oh).sum((0,2,3))
    union = pred.sum((0,2,3)) + target_oh.sum((0,2,3))
    dice = (2*inter+eps)/(union+eps)
    return 1 - dice.mean()