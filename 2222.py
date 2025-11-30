import os

# === 1. 修改为你的图片文件夹路径 ===
folder_path = r"/root/autodl-tmp/Light-DehazeNet-main/data/original_images/GT"   # 改成你的图片路径

# === 2. 获取所有图片文件 ===
# 支持常见格式
valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
files = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_exts)]

# 按名称排序，防止乱序
files.sort()

# === 3. 重命名 ===
for i, filename in enumerate(files, start=1):
    old_path = os.path.join(folder_path, filename)
    new_name = f"{i}.jpg"
    new_path = os.path.join(folder_path, new_name)
    os.rename(old_path, new_path)
    print(f"✅ {filename} → {new_name}")

print(f"\n全部完成，共重命名 {len(files)} 张图片！")