import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import os

# 自定义数据集类
class SDRHDRDataset(Dataset):
    def __init__(self, sdr_dir, hdr_dir, patch_size=512, stride=512):
        self.sdr_dir = sdr_dir
        self.hdr_dir = hdr_dir
        self.patch_size = patch_size
        self.stride = stride
        self.image_pairs = []
        self.patches = []  # 存储所有块的索引 (img_idx, y, x)

        sdr_files = sorted(list(set(os.listdir(sdr_dir))))
        hdr_files = sorted(list(set(os.listdir(hdr_dir))))
        common_files = sorted(list(set(sdr_files) & set(hdr_files)))

        for fname in common_files:
            sdr_path = os.path.join(sdr_dir, fname)
            hdr_path = os.path.join(hdr_dir, fname)
            self.image_pairs.append((sdr_path, hdr_path))

        # 预处理：统计每张图片能切多少块
        for img_idx, (sdr_path, hdr_path) in enumerate(self.image_pairs):
            sdr_img = Image.open(sdr_path).convert('L')
            width, height = sdr_img.size
            for y in range(0, height - patch_size + 1, stride):
                for x in range(0, width - patch_size + 1, stride):
                    self.patches.append((img_idx, y, x))

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        img_idx, y, x = self.patches[idx]
        sdr_path, hdr_path = self.image_pairs[img_idx]

        sdr_img = Image.open(sdr_path).convert('L')
        hdr_img = Image.open(hdr_path).convert('L')

        # 裁剪patch
        sdr_patch = sdr_img.crop((x, y, x + self.patch_size, y + self.patch_size))
        hdr_patch = hdr_img.crop((x, y, x + self.patch_size, y + self.patch_size))

        sdr_array = np.array(sdr_patch, dtype=np.float32) / 255.0
        hdr_array = np.array(hdr_patch, dtype=np.float32) / 65535.0

        sdr_tensor = torch.from_numpy(sdr_array).unsqueeze(0)
        hdr_tensor = torch.from_numpy(hdr_array).unsqueeze(0)
        return sdr_tensor, hdr_tensor

# U-Net模型定义
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        # 编码器 (下采样)
        self.enc1 = self._block(1, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = self._block(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = self._block(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = self._block(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        
        # 瓶颈层
        self.bottleneck = self._block(512, 1024)
        
        # 解码器 (上采样)
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = self._block(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self._block(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self._block(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self._block(128, 64)
        
        # 输出层
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # 编码路径
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))
        
        # 瓶颈
        bottleneck = self.bottleneck(self.pool4(enc4))
        
        # 解码路径 (带跳跃连接)
        dec4 = self.up4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.up3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.up2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.up1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)
        
        return self.final_conv(dec1)

    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

# 训练函数
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    criterion = nn.L1Loss()  # 使用L1损失保留HDR细节
    
    for batch_idx, (sdr, hdr) in enumerate(train_loader):
        sdr, hdr = sdr.to(device), hdr.to(device)
        
        optimizer.zero_grad()
        output = model(sdr)
        loss = criterion(output, hdr)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(sdr)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

# 主函数
def main():
    # 参数设置
    sdr_dir = 'img/sdr_gray'
    hdr_dir = 'img/hdr_gray'
    batch_size = 1
    epochs = 50
    lr = 0.001
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建数据集和数据加载器
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    dataset = SDRHDRDataset(sdr_dir, hdr_dir, patch_size=512, stride=512)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 初始化模型
    model = UNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 训练循环
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
    
    # 保存模型
    torch.save(model.state_dict(), 'sdr_to_hdr.pth')
    print("训练完成，模型已保存")

if __name__ == '__main__':
    main()