import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

class SAM(nn.Module):
    def __init__(self):

        super(SAM, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

    def forward(self, a, b):
        b_, c_, h_, w_ = b.shape
        map = self.conv(b)
        attention_map = a*map

        return attention_map



