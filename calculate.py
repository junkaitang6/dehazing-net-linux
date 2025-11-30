import numpy as np
from PIL import Image
import cv2
import os
import glob
import pandas as pd
from skimage.metrics import structural_similarity as ssim
from typing import List, Tuple, Optional


def calculate_psnr(img1, img2):
    """计算两幅图像的PSNR值"""
    # 确保数据类型为float32
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    # 计算MSE
    mse = np.mean((img1 - img2) ** 2)

    if mse == 0:
        return float('inf')  # 图像完全相同

    # 计算PSNR
    max_pixel_value = 255.0
    psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
    return psnr


def calculate_ssim(img1, img2):
    """计算两幅图像的SSIM值"""
    # 转换为灰度图像进行SSIM计算
    if len(img1.shape) == 3:
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    else:
        img1_gray = img1

    if len(img2.shape) == 3:
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    else:
        img2_gray = img2

    # 计算SSIM
    ssim_value = ssim(img1_gray, img2_gray, data_range=255)
    return ssim_value


def resize_images(img1, img2, target_size='smaller'):
    """调整图像尺寸使其匹配"""
    if img1.shape == img2.shape:
        return img1, img2

    if target_size == 'smaller':
        target_h = min(img1.shape[0], img2.shape[0])
        target_w = min(img1.shape[1], img2.shape[1])
    elif target_size == 'larger':
        target_h = max(img1.shape[0], img2.shape[0])
        target_w = max(img1.shape[1], img2.shape[1])
    else:  # 默认使用第一张图的尺寸
        target_h, target_w = img1.shape[:2]

    if img1.shape[:2] != (target_h, target_w):
        img1 = cv2.resize(img1, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    if img2.shape[:2] != (target_h, target_w):
        img2 = cv2.resize(img2, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    return img1, img2


def get_image_files(folder_path):
    """获取文件夹中的所有图像文件"""
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
    image_files = []

    for ext in extensions:
        pattern = os.path.join(folder_path, ext)
        image_files.extend(glob.glob(pattern))
        # 也检查大写扩展名
        pattern = os.path.join(folder_path, ext.upper())
        image_files.extend(glob.glob(pattern))

    return sorted(image_files)


def match_images(folder1, folder2):
    """根据文件名匹配两个文件夹中的图像"""
    files1 = get_image_files(folder1)
    files2 = get_image_files(folder2)

    # 创建文件名映射
    name_to_path1 = {os.path.splitext(os.path.basename(f))[0]: f for f in files1}
    name_to_path2 = {os.path.splitext(os.path.basename(f))[0]: f for f in files2}

    # 找到匹配的文件
    matched_pairs = []
    for name in name_to_path1:
        if name in name_to_path2:
            matched_pairs.append((name_to_path1[name], name_to_path2[name]))

    return matched_pairs


def calculate_batch_metrics(folder1, folder2, output_csv=None, target_size='smaller'):
    """
    批量计算PSNR和SSIM值

    Args:
        folder1: 第一个图像文件夹
        folder2: 第二个图像文件夹
        output_csv: 输出CSV文件路径
        target_size: 尺寸调整策略 ('smaller', 'larger', 'first')
    """
    # 获取匹配的图像对
    matched_pairs = match_images(folder1, folder2)

    if not matched_pairs:
        print(f"在文件夹中没有找到匹配的图像文件")
        return

    print(f"找到 {len(matched_pairs)} 对匹配的图像")

    results = []
    psnr_values = []
    ssim_values = []

    for i, (img1_path, img2_path) in enumerate(matched_pairs, 1):
        try:
            # 加载图像
            img1 = np.array(Image.open(img1_path))
            img2 = np.array(Image.open(img2_path))

            # 调整图像尺寸
            img1, img2 = resize_images(img1, img2, target_size)

            # 计算PSNR和SSIM
            psnr = calculate_psnr(img1, img2)
            ssim_val = calculate_ssim(img1, img2)

            psnr_values.append(psnr if psnr != float('inf') else 100)  # 处理无限值
            ssim_values.append(ssim_val)

            # 保存结果
            results.append({
                'image_name': os.path.basename(img1_path),
                'psnr': psnr,
                'ssim': ssim_val
            })

            print(f"[{i}/{len(matched_pairs)}] {os.path.basename(img1_path)}: "
                  f"PSNR={psnr:.4f} dB, SSIM={ssim_val:.4f}")

        except Exception as e:
            print(f"处理 {os.path.basename(img1_path)} 时出错: {e}")
            continue

    # 计算统计信息
    if results:
        print("\n=== 统计结果 ===")
        print(f"处理图像数: {len(results)}")
        print(f"平均 PSNR: {np.mean(psnr_values):.4f} dB")
        print(f"平均 SSIM: {np.mean(ssim_values):.4f}")
        print(f"PSNR 标准差: {np.std(psnr_values):.4f}")
        print(f"SSIM 标准差: {np.std(ssim_values):.4f}")
        print(f"最小 PSNR: {np.min(psnr_values):.4f} dB")
        print(f"最大 PSNR: {np.max(psnr_values):.4f} dB")
        print(f"最小 SSIM: {np.min(ssim_values):.4f}")
        print(f"最大 SSIM: {np.max(ssim_values):.4f}")

        # 保存到CSV
        if output_csv:
            df = pd.DataFrame(results)
            df.to_csv(output_csv, index=False, encoding='utf-8')
            print(f"\n结果已保存到: {output_csv}")

    return results


def compare_multiple_folders(reference_folder, test_folders, output_csv=None):
    """
    比较一个参考文件夹与多个测试文件夹
    """
    all_results = []

    for test_folder in test_folders:
        folder_name = os.path.basename(test_folder)
        print(f"\n正在处理: {folder_name}")
        print("-" * 40)

        # 获取匹配的图像对
        matched_pairs = match_images(reference_folder, test_folder)

        if not matched_pairs:
            print(f"在 {folder_name} 中没有找到匹配的图像")
            continue

        psnr_values = []
        ssim_values = []

        for img1_path, img2_path in matched_pairs:
            try:
                img1 = np.array(Image.open(img1_path))
                img2 = np.array(Image.open(img2_path))

                img1, img2 = resize_images(img1, img2)

                psnr = calculate_psnr(img1, img2)
                ssim_val = calculate_ssim(img1, img2)

                psnr_values.append(psnr if psnr != float('inf') else 100)
                ssim_values.append(ssim_val)

            except Exception as e:
                print(f"处理 {os.path.basename(img1_path)} 时出错: {e}")
                continue

        if psnr_values:
            result = {
                'folder': folder_name,
                'image_count': len(psnr_values),
                'avg_psnr': np.mean(psnr_values),
                'avg_ssim': np.mean(ssim_values),
                'std_psnr': np.std(psnr_values),
                'std_ssim': np.std(ssim_values)
            }
            all_results.append(result)

            print(f"图像数: {len(psnr_values)}")
            print(f"平均 PSNR: {np.mean(psnr_values):.4f} dB")
            print(f"平均 SSIM: {np.mean(ssim_values):.4f}")

    # 保存汇总结果
    if output_csv and all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(output_csv, index=False, encoding='utf-8')
        print(f"\n汇总结果已保存到: {output_csv}")

    return all_results


# 使用示例
if __name__ == "__main__":
    # 示例1: 计算两个文件夹的PSNR和SSIM
    folder1 = "/root/autodl-tmp/Light-DehazeNet-main/calculate"#"原图"
    folder2 = "/root/autodl-tmp/Light-DehazeNet-main/vis_results"#"处理后的"

    results = calculate_batch_metrics(
        folder1=folder1,
        folder2=folder2,
        output_csv="metrics_results.csv",
        target_size='smaller'
    )

    # 示例2: 多文件夹对比
    reference_folder = "/path/to/reference/images"
    test_folders = [
        "/path/to/test1/images",
        "/path/to/test2/images",
        "/path/to/test3/images"
    ]

    multi_results = compare_multiple_folders(
        reference_folder=reference_folder,
        test_folders=test_folders,
        output_csv="multi_folder_comparison.csv"
    )