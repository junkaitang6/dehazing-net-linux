import os
from PIL import Image
# 设置文件夹路径（替换为你的文件夹路径）
folder_path = "/root/autodl-tmp/Light-DehazeNet-main/calculate"

# 遍历文件夹中的所有文件
import os

# 修改为你的图片文件夹路径
folder = "/root/autodl-tmp/Light-DehazeNet-main/calculate"

# 获取文件夹中的所有文件，并按照原始顺序排序
files = sorted(os.listdir(folder))

# 过滤出常见的图片文件
image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif"]
images = [f for f in files if os.path.splitext(f)[1].lower() in image_extensions]

# 遍历并重命名
for idx, file_name in enumerate(images, start=1):
    old_path = os.path.join(folder, file_name)
    new_name = f"{idx}.jpg"   # 统一改成.jpg
    new_path = os.path.join(folder, new_name)
    os.rename(old_path, new_path)

print(f"✅ 已完成重命名，共处理 {len(images)} 张图片。")
