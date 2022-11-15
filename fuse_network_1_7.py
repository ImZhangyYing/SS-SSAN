import torch.nn as nn
import torch
# from Attention import ChannelAttention
# from Attention import SpatialAttention
# from Attention import CA
from Supervise_Attention import SAM
from Subspace_Attention import SubspaceAttention as SSA

class fuse_net(nn.Module):
    def __init__(self):
        super(fuse_net, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(192, 32, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(96, 16, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        # self.SAM = SAM()
        # self.SSA = SSA()

    def forward(self, outssa1_1, outssa2_1, outssa1_2, outssa2_2, outssa1_3, outssa2_3):

        y1 = torch.cat((outssa1_3, outssa2_3), 1)
        out1 = self.conv1(y1)
        y2 = torch.cat((outssa1_2, outssa2_2, out1), 1)
        out2 = self.conv2(y2)
        y3 = torch.cat((outssa1_1, outssa2_1, out2), 1)
        out3 = self.conv3(y3)
        out4 = self.conv4(out3)

        return out4


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

