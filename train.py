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
from random import randint, sample
from PIL import Image
import numpy as np
import struct
import gzip
import os

# from tensorboardX import SummaryWriter


torch.cuda.set_device(5)


class MoireCNN(nn.Module):

    def conv(self, channels):
        x=nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 3, 1, 1),
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
        
        return x


def train(epoch, lr):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        if use_gpu:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
        data, target = Variable(data), Variable(target)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.00001)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 5000 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data))


def test(epoch):
    model.eval()

    idx = 0
    loss_sum = 0.0
    for (data, target) in test_loader:
        if use_gpu:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
        data, target = Variable(data), Variable(target)

        with torch.no_grad():
            output = model(data)
            loss = criterion(output, target)

        loss_sum += loss.data
        idx += 1
    loss_sum /= idx

    print('Test Epoch: {} \tLoss: {:.6f}'.format(
                epoch, loss_sum))

    global pre_loss, lr, best_loss
    if loss_sum > pre_loss:
        lr *= 0.9
    
    if loss_sum < best_loss:
        best_loss = loss_sum
        torch.save(model, "moire-1.pth")

    pre_loss = loss_sum


class MoirePic(data.Dataset):
    def __init__(self, rootX, rootY, training=True):
        self.picX=[rootX+i for i in os.listdir(rootX)]
        self.picY=[rootY+i for i in os.listdir(rootY)]
        self.picX.sort()
        self.picY.sort()
        # self.picX=self.picX[:40]
        # self.picY=self.picY[:40]
        self.Len=len(self.picX)

        if not training:
            L = sample(range(self.Len), self.Len//10)
            tempX = [self.picX[i] for i in L]
            tempY = [self.picY[i] for i in L]
            self.picX=tempX
            self.picY=tempY
            self.Len=len(L)
        
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

        pathX, pathY=self.picX[index], self.picY[index]
        imgX, imgY=Image.open(pathX), Image.open(pathY)
        return rand_crop(tf(imgX), tf(imgY))
    
    def __len__(self):
        return self.Len

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(mean=0.0, std=0.01)
        m.bias.data.fill_(0)


dataset = MoirePic("/data_new/moire/trainData/source/",
                   "/data_new/moire/trainData/target/")
testdataset = MoirePic("/data_new/moire/trainData/source/",
                   "/data_new/moire/trainData/target/", False)
use_gpu = torch.cuda.is_available()
batch_size = 8
kwargs = {'num_workers': 14, 'pin_memory': True}
train_loader = DataLoader(dataset=dataset, shuffle=True,
                          batch_size=batch_size, **kwargs)
test_loader = DataLoader(dataset=testdataset, shuffle=True,
                          batch_size=batch_size, **kwargs)  

# model = MoireCNN()
model = torch.load("moire-1.pth")
# model.apply(weights_init)

# with SummaryWriter(comment='MoireCNN') as w:
#     w.add_graph(model, (x, ))

if use_gpu:
    model = model.cuda()
    # model = nn.DataParallel(model)
    print('USE GPU')
else:
    print('USE CPU')

criterion = nn.MSELoss()
lr = 0.00004
pre_loss = 100.0
best_loss = 100.0

for epoch in range(100):
    train(epoch, lr)
    test(epoch)