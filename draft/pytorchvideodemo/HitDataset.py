import os
from sklearn.model_selection import train_test_split

import torch
import cv2
import numpy as np
from torch.utils.data import ConcatDataset, Dataset

class HitDataset(Dataset):
    def __init__(self,data_folder=""):
        self.resize_height = 1920
        self.resize_width = 1080
        self.dataset=[]
        for dir in os.listdir(data_folder):
            sub_dir=os.path.join(data_folder, dir)
            
            if os.path.isdir(sub_dir):
                dir_name = os.path.basename(dir)
                if (dir_name)=="none":
                    label=torch.FloatTensor([1,0,0])
                elif dir_name=="top":
                    label=torch.FloatTensor([0,1,0])
                elif dir_name=="bottom":
                    label=torch.FloatTensor([0,0,1])
                
                buffer = np.empty((5, self.resize_height, self.resize_width, 3), np.dtype('float32'))
                for img_folder_name in os.listdir(sub_dir):
                    img_folder=os.path.join(sub_dir,img_folder_name)
                    # print(img_folder)
                    i=0
                    for img_name in os.listdir(img_folder):
                        img_path=os.path.join(img_folder,img_name)
                        # print(img_path)
                        img=cv2.imread(img_path)
                        # img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img=cv2.resize(img, (self.resize_width, self.resize_height))
                        img=np.array(img).astype(np.float64)/255.0
                        img=torch.FloatTensor(img)
                        buffer[i] = img
                        i+=1
                    # buffer=self.normalize(buffer)
                    
                    # print(buffer.shape)
                    self.dataset.append((buffer,label))

    def to_tensor(self, buffer):
        return buffer.transpose((3, 0, 1, 2))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample=self.dataset[index]
        data=sample[0]
        label=sample[1]
        data=self.to_tensor(data)
        return data,label
    


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import time

    start_time=time.time()

    train_data =HitDataset(data_folder="TrainDataset")

    end_time=time.time()

    print("spend {} s load data".format(end_time-start_time))
    train_loader = DataLoader(train_data, batch_size=100, shuffle=True, num_workers=4)


    for inputs,labels in train_loader:
        print(inputs.shape)
        print(labels.shape)
        break