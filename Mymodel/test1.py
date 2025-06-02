import torch
import numpy as np
from PIL import Image
from model1 import UNet


def convert_sdr_to_hdr(model_path, sdr_image_path, output_path):
    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # 加载并预处理SDR图像
    sdr_img = Image.open(sdr_image_path).convert('L')
    sdr_array = np.array(sdr_img, dtype=np.float32) / 255.0
    sdr_tensor = torch.tensor(sdr_array).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, H, W)
    
    # 转换
    with torch.no_grad():
        hdr_output = model(sdr_tensor)
    
    # 后处理
    hdr_array = hdr_output.squeeze().cpu().numpy()  # (H, W)
    hdr_array = np.clip(hdr_array, 0, 1)  # 确保值在[0,1]范围内
    hdr_16bit = (hdr_array * 65535).astype(np.uint16)
    
    # 保存为16位PNG
    Image.fromarray(hdr_16bit).save(output_path)
    print(f"HDR图像已保存至 {output_path}")

# 示例用法
convert_sdr_to_hdr('sdr_to_hdr.pth', 'img/sdr_gray/002.png', 'output_hdr.png')