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
from HitDataset import HitDataset
# Use GPU if available else revert to CPU

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device being used:", device)

import torch
import torch.nn as nn

import torch
import torch.nn as nn

class X3D(nn.Module):
    def __init__(self, num_classes=3):
        super(X3D, self).__init__()
        
        # 定义X3D的网络结构
        self.conv1 = nn.Conv3d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(256, num_classes)
        
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.conv3(x)
        x = self.relu(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        x = self.fc(x)
        x = F.softmax(x, dim=1)  # 应用softmax函数
        return x

    


train_dataset=HitDataset(data_folder="ValDataset")
val_dataset=HitDataset(data_folder="ValDataset")
test_dataset=HitDataset(data_folder="ValDataset")


train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_dataloader   = DataLoader(val_dataset, batch_size=2)
test_dataloader  = DataLoader(test_dataset)


model = X3D(num_classes=3)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10
train_loss_list = []


test_loss_list=[]
for epoch in range(num_epochs):
    train_loss_sum=0
    model.train()
    for batch_data in train_dataloader:
        inputs, labels = batch_data
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