import argparse
import os
import cv2
import numpy as np
import torch
from PIL import Image
import time
import torch.nn.functional as F
import torch.nn as nn
mse = nn.MSELoss(reduction='mean')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GaussianBlurConv(nn.Module):
    def __init__(self, channels=1):
        super(GaussianBlurConv, self).__init__()
        self.channels = channels
        kernel = [[0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633],
                  [0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965],
                  [0.01330373, 0.11098164, 0.22508352, 0.11098164, 0.01330373],
                  [0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965],
                  [0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633]]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel = np.repeat(kernel, self.channels, axis=0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def __call__(self, x):
        weight = self.weight
        weight = weight.to(device)
        x = F.conv2d(x, weight, padding=2, groups=self.channels)
        return x

def loss_ec(img,F):
    a = 2.5
    b = 1.2
    gaussian_conv = GaussianBlurConv()
    Gf = gaussian_conv(F)
    G = gaussian_conv(img)
    Cf = Gf
    C = G

    xf = F - Gf
    x = img - G

    E = torch.abs(x)
    Ef = torch.abs(xf)

    y = a*C + b*E
    yf = a*Cf + b*Ef

    loss = mse(y, yf)
    return loss