import torch.nn as nn
import torch
# from Attention import ChannelAttention
# from Attention import SpatialAttention
# from Attention import CA
from Supervise_Attention import SAM
from Subspace_Attention import SubspaceAttention as SSA

class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()

        self.conv1_1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv1_3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv2_3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv1_4 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv2_4 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        self.conv1_5 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv2_5 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv1_6 = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )
        self.conv2_6 = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

        self.conv9 = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )
        self.SAM = SAM()
        self.SA1 = SSA(64, 32)
        self.SA2 = SSA(128, 64)
        self.SA3 = SSA(256, 128)


    def forward(self, x1, x2):
        out1_1 = self.conv1_1(x1)
        out2_1 = self.conv2_1(x2)
        outssa1_1 = self.SA1(out1_1, out2_1)
        outssa2_1 = self.SA1(out2_1, out1_1)

        out1_2 = self.conv1_2(out1_1)
        out2_2 = self.conv2_2(out2_1)
        outssa1_2 = self.SA2(out1_2, out2_2)
        outssa2_2 = self.SA2(out2_2, out1_2)


        out1_3 = self.conv1_3(out1_2)
        out2_3 = self.conv2_3(out2_2)
        outssa1_3 = self.SA3(out1_3, out2_3)
        outssa2_3 = self.SA3(out2_3, out1_3)

        # 以上为encoder
        out1_4 = self.conv1_4(out1_3)
        out2_4 = self.conv2_4(out2_3)

        out1_5 = self.conv1_5(out1_4)
        out2_5 = self.conv2_5(out2_4)
        out1_6 = self.conv1_6(out1_5)
        out2_6 = self.conv2_6(out2_5)
        # out1 = self.SAM(out1_6, x1)
        # out2 = self.SAM(out2_6, x2)
        # out = torch.cat((out1, out2), 1)
        # F = self.conv9(out)

        return out1_1, out2_1, out1_2, out2_2, out1_3, out2_3, out1_6, out2_6


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

