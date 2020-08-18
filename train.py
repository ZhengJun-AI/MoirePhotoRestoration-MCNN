import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import math
from torch.utils.data import TensorDataset, DataLoader
import torch.utils.data as data
from torchvision import transforms
from torch.autograd import Variable
from torch.nn import functional as F
from torch import nn
import torch
from random import randint
from PIL import Image
import numpy as np
import struct
import gzip
import os


torch.cuda.set_device(4)


class MoireCNN(nn.Module):

    def conv(self, channels):
        x=nn.Sequential(
            nn.Conv2d(channels, channels, 1, 1, 0),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 1, 1, 0),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 1, 1, 0),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 1, 1, 0),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 1, 1, 0),
            nn.ReLU(True)
        )
        return x

    def __init__(self):

        super().__init__()
        
        self.s11=nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3, 1, 1)
        )
        self.s12=nn.Conv2d(32, 3, 3, 1, 1)
        self.s13=self.conv(32)
        
        self.s21=nn.Sequential(
            nn.Conv2d(32, 32, 3, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, 1, 1)
        )
        self.s22=nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 3, 3, 1, 1)
        )
        self.s23=self.conv(64)
        
        self.s31=nn.Sequential(
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1)
        )
        self.s32=nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 3, 3, 1, 1)
        )
        self.s33=self.conv(64)
        
        self.s41=nn.Sequential(
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1)
        )
        self.s42=nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 3, 3, 1, 1)
        )
        self.s43=self.conv(64)
        
        self.s51=nn.Sequential(
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1)
        )
        self.s52=nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 3, 3, 1, 1)
        )
        self.s53=self.conv(64)
        
    def forward(self, x):
        x1=self.s11(x)
        x2=self.s21(x1)
        x3=self.s31(x2)
        x4=self.s41(x3)
        x5=self.s51(x4)
        
        x1=self.s12(self.s13(x1))
        x2=self.s22(self.s23(x2))
        x3=self.s32(self.s33(x3))
        x4=self.s42(self.s43(x4))
        x5=self.s52(self.s53(x5))

        x=x1+x2+x3+x4+x5
        
        return torch.clamp(x, 0, 1)


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if use_gpu:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 5000 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data))


class MoirePic(data.Dataset):
    def __init__(self, rootX, rootY, train=True):
        self.train=train
        self.picX=[rootX+i for i in os.listdir(rootX)]
        self.picY=[rootY+i for i in os.listdir(rootY)]
        self.picX.sort()
        self.picY.sort()
        self.picX=self.picX
        self.picY=self.picY
        
    def __getitem__(self, index):
        tf=transforms.ToTensor()

        def rand_crop(data,label):
            img_w, img_h = 256, 256

            width1 = randint(0, data.shape[1] - img_w )
            height1 = randint(0, data.shape[2] - img_h)
            width2 = width1 + img_w
            height2 = height1 + img_h 

            return (data[:,width1:width2,height1:height2],
            label[:,width1:width2,height1:height2])

        if self.train:
            pathX, pathY=self.picX[index], self.picY[index]
            imgX, imgY=Image.open(pathX), Image.open(pathY)
            return rand_crop(tf(imgX), tf(imgY))
    
    def __len__(self):
        return len(self.picX)


dataset = MoirePic("/data_new/moire/trainData/source/",
                   "/data_new/moire/trainData/target/")
# dataset=MoirePic("/home/zhengjun/moire/moire-data/ValidationMoire/",
#                 "/home/zhengjun/moire/moire-data/ValidationClear/")
use_gpu = torch.cuda.is_available()
batch_size = 32
kwargs = {'num_workers': 14, 'pin_memory': True}
train_loader = DataLoader(dataset=dataset, shuffle=True,
                          batch_size=batch_size, **kwargs)

model = MoireCNN()

if use_gpu:
    model = model.cuda()
    # model = nn.DataParallel(model, device_ids=[3, 4, 5, 6])
    print('USE GPU')
else:
    print('USE CPU')

criterion = nn.MSELoss()
lr = 0.0001
optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=0.00001)

for epoch in range(101):
    train(epoch)
    if epoch % 10 == 0:
        torch.save(model, "moire-1.pth")
