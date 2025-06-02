import cv2
import os

input_dir = 'img/sdr'
output_dir = 'img/sdr_gray'
os.makedirs(output_dir, exist_ok=True)

for fname in os.listdir(input_dir):
    if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')):
        img_path = os.path.join(input_dir, fname)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"无法读取: {img_path}")
            continue
        # 转为灰度
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 保存
        out_path = os.path.join(output_dir, fname)
        cv2.imwrite(out_path, gray)
        print(f"已保存: {out_path}")