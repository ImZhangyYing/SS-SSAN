import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from Inverse import invmat

class SubspaceAttention(nn.Module):
    def __init__(self, intput_channel, output_channel):

        super(SubspaceAttention, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(intput_channel, output_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(output_channel, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )


    def forward(self, x1, x2):
        b_, c_, h_, w_ = x1.shape

        ##############SSAB
        y1 = torch.cat((x1, x2), dim=1)
        y2 = torch.cat((x2, x1), dim=1)
        out1 = self.conv1(y1)
        out2 = self.conv1(y2)
        out1 = self.conv2(out1)
        out2 = self.conv2(out2)
        # _________________________________
        Vt1 = out1.reshape(b_, 16, h_ * w_)
        Vt2 = out2.reshape(b_, 16, h_ * w_)
        V_t1 = Vt1 / (1e-6 + torch.abs(Vt1).norm(2))
        V_t2 = Vt2 / (1e-6 + torch.abs(Vt2).norm(2))
        # V = V_t.transpose(0, 2, 1)
        V1 = V_t1.permute(0, 2, 1)
        V2 = V_t2.permute(0, 2, 1)
        mat1 = torch.matmul(V_t1, V1)
        mat2 = torch.matmul(V_t2, V2)
        # mat = mat1 + mat2
        mat_inv1 = invmat(mat1)
        mat_inv2 = invmat(mat2)
        ##############
        project_mat1 = torch.matmul(mat_inv1, V_t1)
        project_mat2 = torch.matmul(mat_inv2, V_t2)
        project_mat = project_mat1+project_mat2

        x_1 = x1.reshape(b_, c_, h_*w_).permute(0, 2, 1)
        x_2 = x2.reshape(b_, c_, h_ * w_).permute(0, 2, 1)
        project_feature1 = torch.matmul(project_mat, x_1)
        project_feature2 = torch.matmul(project_mat, x_2)
        Y1 = torch.matmul(V1, project_feature1).permute(0, 2, 1).reshape(b_, c_, h_, w_)
        Y2 = torch.matmul(V2, project_feature2).permute(0, 2, 1).reshape(b_, c_, h_, w_)


        ############SSA
        # y1 = torch.cat((x1, x2), dim=1)
        # out1 = self.conv1(y1)
        # out = self.conv2(out1)
        # # _________________________________
        # Vt = out.reshape(b_, 16, h_ * w_)
        # V_t = Vt / (1e-6 + torch.abs(Vt).norm(2))
        # # V = V_t.transpose(0, 2, 1)
        # V = V_t.permute(0, 2, 1)
        # mat = torch.matmul(V_t, V)

        #
        #         # ########张量的逆运算
        #         # a = torch.arange(0, b_).view(-1, 2, 2)
        #         # b = [mat.inverse() for mat in torch.unbind(a)]
        #         # mat_inv = torch.stack(b)

        ## mat_inv = mat.inverse()
        # mat_inv = invmat(mat)
        # ##############
        # project_mat = torch.matmul(mat_inv, V_t)
        # x = x1.reshape(b_, c_, h_*w_).permute(0, 2, 1)
        # project_feature = torch.matmul(project_mat, x)
        # Y = torch.matmul(V, project_feature).permute(0, 2, 1).reshape(b_, c_, h_, w_)
        # _________________________________


        return Y1,Y2



