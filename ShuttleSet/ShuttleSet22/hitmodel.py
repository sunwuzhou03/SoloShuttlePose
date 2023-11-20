
import torch
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader
import os

'''
human keypoints: 17*2
court keypoints: 6*2
net keypoints:   4*2
ball keypoints:  1*2
total keypoints: 28*2

'''

class HitModel(nn.Module):
    def __init__(self, feature_dim, num_consecutive_frames):
        super(HitModel, self).__init__()
        
        self.num_consecutive_frames = num_consecutive_frames
        
        self.gru1 = nn.GRU(input_size=feature_dim // num_consecutive_frames, hidden_size=64, bidirectional=True, batch_first=True)
        self.gru2 = nn.GRU(input_size=128, hidden_size=64, bidirectional=True, batch_first=True)
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(128, 3)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = x.view(-1, self.num_consecutive_frames, x.size(1) // self.num_consecutive_frames)
        
        x, _ = self.gru1(x)
        x, _ = self.gru2(x)
        
        x = x.transpose(1, 2).contiguous()
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.softmax(x)
        
        return x
    



class HitDataset(Dataset):
    def __init__(self, dataset_folder,num_consecutive_frames,normalization=True):
        self.dataset=[]
        # 遍历文件夹及其子文件夹，找到所有的CSV文件路径
        for root, dirs, files in os.walk(dataset_folder):
            for file in files:
                if file.endswith(".csv"):
                    # 定义处理函数
                    data_path=os.path.join(root, file)
                    print(data_path)
                    df= pd.read_csv(data_path, converters={"ball": eval,"top":eval,"bottom":eval,"court":eval,"net":eval})
                    
                    rows = len(df)
                    remainder = rows % 12
                    if remainder > 0:
                        num_to_pad = 12 - remainder
                    else:
                        num_to_pad = 0

                    if num_to_pad > 0:
                        last_row = df.iloc[-1]
                        padding_data = np.tile(last_row.values, (num_to_pad, 1))
                        padded_df = pd.DataFrame(padding_data, columns=df.columns)
                        df = pd.concat([df, padded_df], axis=0)
                        df = df.reset_index(drop=True)

                    small_dataset =df
                    
                    for i in range(len(small_dataset)):
                        
                        if i>=len(small_dataset)-num_consecutive_frames:
                            break
                        
                        pos=np.array(small_dataset.loc[i,'pos'])
                        
                        if str(pos)=='nan':
                            target=[1,0,0]
                        elif str(pos)=='top':
                            target=[0,1,0]
                        elif str(pos)=='bottom':
                            target=[0,0,1]
                        oridata=small_dataset.loc[i:i+num_consecutive_frames,:].copy()
                        oridata.reset_index(drop=True)
                        data=[]
                        feature_dim=2*17*2+6*2+4*2+1*2
                        for index,row in oridata.iterrows():
                            top=np.array(row['top']).reshape(-1,2)
                            bottom=np.array(row['bottom']).reshape(-1,2)
                            court=np.array(row['court']).reshape(-1,2)
                            net=np.array(row['net']).reshape(-1,2)
                            ball=np.array(row['ball']).reshape(-1,2)
                            
                            frame_data = np.concatenate((top, bottom, court, net, ball), axis=0)
                            if normalization:
                                frame_data[:,0]/=1920
                                frame_data[:,1]/=1080
                            data.append(frame_data.reshape(1,-1))
                        data=np.array(data)
                        self.dataset.append((data.reshape(-1,feature_dim),np.array(target)))
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # 假设每个样本是一个元组 (input, target)
        sample = self.dataset[index]
        input_data = sample[0]
        target = sample[1]
        # 转换为Tensor对象
        input_tensor = torch.tensor(input_data).reshape(1,-1)
        target_tensor = torch.tensor(target)

        return input_tensor, target_tensor
    
feature_dim=17*2+17*2+6*2+4*2+1*2
num_consecutive_frames=12
batch_size=30
shuffle=True
num_epochs = 10


TrainDataset=HitDataset("ShuttleSet\ShuttleSet22\dataset/train",num_consecutive_frames)
TestDataset=HitDataset("ShuttleSet\ShuttleSet22\dataset/test",num_consecutive_frames)

train_data_loader = DataLoader(TrainDataset, batch_size=batch_size, shuffle=shuffle)
test_data_loader = DataLoader(TestDataset, batch_size=batch_size, shuffle=shuffle)

model=HitModel(feature_dim,num_consecutive_frames)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


train_loss_list = []
test_loss_list=[]
for epoch in range(num_epochs):
    train_loss_sum=0
    model.train()
    for batch_data in train_data_loader:
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
                                                    train_loss.item()))