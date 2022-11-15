from dataset import fusiondata
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
# import pytorch_ssim
import torch.backends.cudnn as cudnn
import numpy as np
import matplotlib.pyplot as plt
from network_1_7 import AE
from PIL import Image
import torch.nn.functional as F
from torch.autograd import Variable
import cv2



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
root_path = './train_datasets/'
net = AE().to(device)

dataset = fusiondata(os.path.join(root_path))
training_data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
optimizer = optim.Adam(net.parameters(), lr=0.0001, betas=(0.9, 0.999))

mse_loss = nn.MSELoss(reduction='mean')

train_loss = []
train_ssim = []
#训练
def train(epoch):
    for iteration, batch in enumerate(training_data_loader, 1):
        # imgA1, imgA2  = batch[0], batch[1]
        # img_A = imgA1.to(device)
        # img_B = imgA2.to(device)

        imgA1, imgA2, detected_img1, detected_img2, pc_img1, pc_img2, g_img1, g_img2 = batch[0], batch[1], batch[2], batch[3], batch[4], batch[5],batch[6], batch[7]

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

        ##############################################
        a = 1
        b = 10
        eps = 1e-8
        ##############################################
        u1 = detected_img1 * l_img1
        u2 = detected_img2 * l_img2
        v1 = pc_img1 * g_img1
        v2 = pc_img2 * g_img2
        w_1 = (u1 ** a + v1 ** b + eps) / ((u1 ** a + v1 ** b) + (u2 ** a + v2 ** b) + eps)
        w_2 = 1 - w_1
        # w_1 = (img_A+eps)/(img_A+img_B+eps)
        # w_2 = 1-w_1
        ##############################################
        # out_image = net(img_A, img_B)
        outssa1_1, outssa2_1, outssa1_2, outssa2_2, outssa1_3, outssa2_3, out1, out2 = net(img_A, img_B)
        ##############################################
        loss_a_1 = ((out1 - img_A)*w_1).norm(2)
        loss_b_1 = ((out2 - img_B)*w_2).norm(2)
        LOSS1 = (loss_a_1 + loss_b_1)

        loss_AE = LOSS1
        ##############################################
        optimizer.zero_grad()
        loss_AE.backward()
        optimizer.step()
        print("===> Epoch[{}]/({}/{}): loss_mse: {:.4f}".format(epoch, iteration, len(training_data_loader),
                                                                loss_AE.item()))
        train_loss.append(loss_AE.item())
        # train_ssim.append(SSIM.item())

def save_data(epoch):

    train_loss_out_path = './AE/loss_{}.npy'.format(epoch)
    m = np.array(train_loss)
    np.save(train_loss_out_path, m)
    print('loss has save')

def checkpoint(epoch):

    net_ae_model_out_path = './AE/netG_{}.pth'.format(epoch)
    torch.save(net, net_ae_model_out_path)
    print('checkpoint_AE', str(epoch), 'has saved!')

if __name__ == '__main__':
    for epoch in range(501):
        train(epoch)

        if epoch % 100 == 0:
            checkpoint(epoch)
           # save_data(epoch)

