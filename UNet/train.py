from torch import nn,optim
from torch.utils.data import DataLoader
import torch
from torchvision.utils import save_image
from data import *
from net import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weight_path = 'params/unet.pth'
data_path = "/data/wangweicheng/pytest/image"
save_path = "train_image"

if __name__ == '__main__':
    data_loader = DataLoader(MyDataset(data_path),batch_size=4, shuffle=True)
    net=Unet().to(device)
    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path))
        print("successful load weight!")
    else:
        print("not successful load weight")
    
    opt = optim.Adam(net.parameters())
    loss_fun = nn.BCELoss()

    epoch = 1
    while True:
        for i, (image, segment_image) in enumerate(data_loader):
        
            image, segment_image = image.to(device), segment_image.to(device)

            out_image = net(image)
            # 裁剪目标张量为256的尺寸
            # out_image = F.interpolate(out_image, size=(256, 256), mode='nearest')
            train_loss = loss_fun(out_image, segment_image)

            opt.zero_grad()
            train_loss.backward()
            opt.step()

            if i%5 == 0:
                print(f'{epoch}-{i}-train_loss====>>{train_loss.item()}')
            
            if i%50 == 0:
                torch.save(net.state_dict(), weight_path)

            _image = image[0]
            _segment_image = segment_image[0]
            _out_image = out_image[0]

            img = torch.stack([_image, _segment_image, _out_image], dim=0)
            save_image(img,f'{save_path}/{i}.png')
        epoch += 1
