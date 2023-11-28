# Imports
import torch.nn as nn
from pytorchvideo.layers.accelerator.mobile_cpu.activation_functions import (
    supported_act_functions,
)
from pytorchvideo.layers.accelerator.mobile_cpu.convolutions import (
    Conv3d5x1x1BnAct,
)
from pytorchvideo.models.accelerator.mobile_cpu.residual_blocks import (
    X3dBottleneckBlock,
)
from pytorchvideo.layers.accelerator.mobile_cpu.pool import AdaptiveAvgPool3dOutSize1
from pytorchvideo.layers.accelerator.mobile_cpu.fully_connected import FullyConnected

import os

import torch
import cv2
import numpy as np
from torch.utils.data import ConcatDataset, Dataset
import torch
import torchvision
import timeit
from datetime import datetime
import socket
import os
import glob
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
# Use GPU if available else revert to CPU

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device being used:", device)

import torch
import torch.nn as nn


class HitDataset(Dataset):
    def __init__(self,data_folder=""):
        self.resize_height = 1080//4
        self.resize_width = 1920//4
        self.dataset=[]
        for dir in os.listdir(data_folder):
            sub_dir=os.path.join(data_folder, dir)
            
            if os.path.isdir(sub_dir):
                dir_name = os.path.basename(dir)
                if (dir_name)=="none":
                    label=[1,0,0]
                elif dir_name=="top":
                    label=[0,1,0]
                elif dir_name=="bottom":
                    label=[0,0,1]
                
                buffer = []
                for img_folder_name in os.listdir(sub_dir):
                    img_folder=os.path.join(sub_dir,img_folder_name)
                    # print(img_folder)
                    i=0
                    for img_name in os.listdir(img_folder):
                        img_path=os.path.join(img_folder,img_name)
                        # print(img_path)
                        img=cv2.imread(img_path)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img_torch = torchvision.transforms.ToTensor()(img) # already [0, 1]
                        img_torch = torchvision.transforms.functional.resize(img_torch, [self.resize_height,self.resize_weight], antialias=True)
                        # img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        # img=cv2.resize(img, (self.resize_width, self.resize_height))
                        
                        buffer.append(img_torch)
                        i+=1
                    # buffer=self.normalize(buffer)
                    
                    # print(buffer.shape)
                    self.dataset.append((buffer,label))

    # N * C * T * H * W

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample=self.dataset[index]
        data=torch.FloatTensor(sample[0])
        label=torch.FloatTensor(sample[1])
        
        return data,label
    

class MyNet(nn.Module):
    def __init__(
        self,
        in_channel=3,  # input channel of first 5x1x1 layer
        residual_block_channel=24,  # input channel of residual block
        expansion_ratio=3, # expansion ratio of residual block
        num_classes=3, # final output classes
    ):
        super().__init__()
        # s1 - 5x1x1 conv3d layer
        self.s1 = Conv3d5x1x1BnAct(
            in_channel,
            residual_block_channel,
            bias=False,
            groups=1,
            use_bn=False,
        )
        # s2 - residual block
        mid_channel = int(residual_block_channel * expansion_ratio)
        self.s2 = X3dBottleneckBlock(
                in_channels=residual_block_channel,
                mid_channels=mid_channel,
                out_channels=residual_block_channel,
                use_residual=True,
                spatial_stride=1,
                se_ratio=0.0625,
                act_functions=("relu", "swish", "relu"),
                use_bn=(True, True, True),
            )
        # Average pool and fully connected layer
        self.avg_pool = AdaptiveAvgPool3dOutSize1()
        self.projection = FullyConnected(residual_block_channel, num_classes, bias=True)
        self.act = supported_act_functions['relu']()

    def forward(self, x):
        x = self.s1(x)
        x = self.s2(x)
        x = self.avg_pool(x)
        # (N, C, T, H, W) -> (N, T, H, W, C).
        x = x.permute((0, 2, 3, 4, 1))
        x = self.projection(x)
        # Performs fully convolutional inference.
        if not self.training:
            x = self.act(x)
            x = x.mean([1, 2, 3])
        x = x.view(x.shape[0], -1)
        x = F.softmax(x, dim=1)  
        return x
    


train_dataset=HitDataset(data_folder="draft\pytorchvideodemo\ValDataset")
val_dataset=HitDataset(data_folder="draft\pytorchvideodemo\ValDataset")
test_dataset=HitDataset(data_folder="draft\pytorchvideodemo\ValDataset")


train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_dataloader   = DataLoader(val_dataset, batch_size=1)
test_dataloader  = DataLoader(test_dataset)


model = MyNet()
model.to(device)
print(model)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 100
train_loss_list = []


test_loss_list=[]
for epoch in range(num_epochs):
    train_loss_sum=0
    model.train()
    for batch_data in train_dataloader:
        inputs, labels = batch_data
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        train_loss = criterion(outputs, labels)
        train_loss_sum+=train_loss.detach() 
        # 反向传播和优化
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        # 执行你的训练或测试操作
    train_loss_list.append(train_loss_sum)
        
    # 打印训练信息
    if (epoch + 1) % 1== 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs,
                                                    train_loss_sum))


    model.eval()
    test_loss_sum=0
    total=0
    correct=0

    with torch.no_grad():
        cnt=0
        for batch_data in test_dataloader:
            inputs, labels = batch_data
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            total+=len(labels)

            outputs = model(inputs)

            y_true=torch.argmax(labels,dim=1)
            y_pred=torch.argmax(outputs,dim=1)

            print(labels,outputs)

            print(y_true,y_pred)
            correct+=sum(y_true==y_pred)
            test_loss = criterion(outputs, labels)
            test_loss_sum+=test_loss.detach()
    test_loss_list.append(test_loss_sum)
    print(correct/total)

# 保存整个模型
torch.save(model, 'model.pth')