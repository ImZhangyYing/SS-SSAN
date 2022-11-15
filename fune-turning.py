from dataset import fusiondata
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import numpy as np
import matplotlib.pyplot as plt
from fuse_network_1_7 import fuse_net
from PIL import Image
import torch.nn.functional as F
from torch.autograd import Variable
import cv2
from Loss_EC import loss_ec



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
root_path = './train_datasets/'
ae_model_path = './AE/netG_' + str(1000) + '.pth'
net = torch.load(ae_model_path)
dataset = fusiondata(os.path.join(root_path))
training_data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)

f_net = fuse_net().to(device)
optimizer1 = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))
optimizer2 = optim.Adam(f_net.parameters(), lr=0.001, betas=(0.9, 0.999))
mse_loss = nn.MSELoss(reduction='mean')


#训练
def train(epoch):
    for iteration, batch in enumerate(training_data_loader, 1):

        imgA1, imgA2, detected_img1, detected_img2, pc_img1, pc_img2, g_img1, g_img2 = batch[0], batch[1], batch[2], batch[3], batch[4], batch[5], batch[6], batch[7]

        img_A = imgA1.to(device)
        img_B = imgA2.to(device)
        detected_img1 = detected_img1.to(device)
        detected_img2 = detected_img2.to(device)
        pc_img1 = pc_img1.to(device)
        pc_img2 = pc_img2.to(device)
        g_img1 = g_img1.to(device)
        g_img2 = g_img2.to(device)
        l_img1 = img_A
        l_img2 = img_B

        outssa1_1, outssa2_1, outssa1_2, outssa2_2, outssa1_3, outssa2_3, out1, out2 = net(img_A, img_B)
        out_image = f_net(outssa1_1, outssa2_1, outssa1_2, outssa2_2, outssa1_3, outssa2_3)

        a = 1
        b = 10
        eps = 1e-8
        #################################
        u1 = detected_img1 * l_img1
        u2 = detected_img2 * l_img2
        v1 = pc_img1 * g_img1
        v2 = pc_img2 * g_img2
        w_1 = (u1 ** a + v1 ** b + eps) / ((u1 ** a + v1 ** b) + (u2 ** a + v2 ** b) + eps)
        w_2 = 1 - w_1
        # w_1 = 1
        # w_2 = 1


        # loss_i1 = ((out_image - img_CT)*(w1+eps)).norm(2)
        # loss_i2 = ((out_image - img_MR)*(w2+eps)).norm(2)
        loss_i1 = ((out_image - img_A)*w_1).norm(2)
        loss_i2 = ((out_image - img_B)*w_2).norm(2)
        loss1 = loss_ec(img_A, out_image)
        loss2 = loss_ec(img_B, out_image)
        loss_EC = loss1+loss2

        # loss_ssim_i1 = ssim_loss(out_image, img_A)
        # loss_ssim_i2 = ssim_loss(out_image, img_B)


        loss_sum = (loss_i1 + loss_i2) + 0.8*loss_EC #- (loss_ssim_i1 + loss_ssim_i2)*5p4;nnm.

        optimizer1.zero_grad()
        optimizer2.zero_grad()
        loss_sum.backward()
        optimizer1.step()
        optimizer2.step()
        print("===> Epoch[{}]/({}/{}): loss_mse: {:.4f}".format(epoch, iteration, len(training_data_loader), loss_sum.item()))
       # print('===> Epoch[', epoch, ']', 'MEElossFUSE:', loss_sum.item())

def checkpoint(epoch):

    # f_net_model_out_path = 'F:/medical3/parameter_fuse/parameter_fuse_7/f_net_{}.pth'.format(epoch)
    # torch.save(f_net, f_net_model_out_path)
    # print('checkpoint', str(epoch), 'has saved!')

    net_ae_model_out_path = './last_model/ae/netG_{}.pth'.format(epoch)
    torch.save(net, net_ae_model_out_path)
    print('checkpoint_AE', str(epoch), 'has saved!')

    f_net_model_out_path = './last_model/fuse/f_net_{}.pth'.format(epoch)
    torch.save(f_net, f_net_model_out_path)
    print('checkpoint', str(epoch), 'has saved!')

if __name__ == '__main__':
    for epoch in range(1001):
        train(epoch)

        if epoch % 100 == 0:
            checkpoint(epoch)