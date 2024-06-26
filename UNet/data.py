import os
from utils import *
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from utils import keep_image_size_open

transform=transforms.Compose([
    transforms.ToTensor()
])

class MyDataset(Dataset):
    def __init__(self,path):
        self.path = path
        self.name = os.listdir(os.path.join(path, "SegmentationClass"))

    def __len__(self):
        return len(self.name)

    def __getitem__(self, index):
        segment_name = self.name[index]
        segment_path = os.path.join(self.path, "SegmentationClass",segment_name)
        image_path = os.path.join(self.path, "JPEGImages",segment_name.replace('png','jpg'))
        segment_image = keep_image_size_open(segment_path)
        image = keep_image_size_open(image_path)
        return transform(image),transform(segment_image)
    

if __name__ == "__main__":
    data = MyDataset("/data/wangweicheng/pytest/image")
    print(data[0][0].shape)
    print(data[0][1].shape)

    
    



