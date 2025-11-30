# @author: hayat (updated for enhanced model)
import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import image_data_loader
import lightdehazeNet
import numpy
import numpy as np
from torchvision import transforms
from PIL import Image
import glob
from matplotlib import pyplot as plt
import cv2


def image_haze_removel(input_image, model_type='enhanced', weights_path=None):
	"""
    去雾函数
    Args:
        input_image: 输入的有雾图像
        model_type: 模型类型，'original' 或 'enhanced'
        weights_path: 权重文件路径，如果为None则自动选择
    """

	# 图像预处理
	hazy_image = (np.asarray(input_image) / 255.0)
	hazy_image = torch.from_numpy(hazy_image).float()
	hazy_image = hazy_image.permute(2, 0, 1)
	hazy_image = hazy_image.cuda().unsqueeze(0)

	# 根据模型类型选择网络
	if model_type == 'enhanced':
		print("使用增强版LightDehaze网络")
		ld_net = lightdehazeNet.LightDehaze_Net_GlobalEnhanced().cuda()
		default_weights = 'trained_weights/best_Global_Enhanced_LDNet.pth'
	else:
		print("使用原始LightDehaze网络")
		ld_net = lightdehazeNet.LightDehaze_Net_GlobalEnhanced().cuda()
		default_weights = 'trained_weights/best_Global_Enhanced_LDNet.pth'

	# 确定权重文件路径
	if weights_path is None:
		weights_path = default_weights

	# 检查权重文件是否存在
	if not os.path.exists(weights_path):
		print(f"错误：权重文件不存在 - {weights_path}")
		print("可用的权重文件:")
		weight_dir = os.path.dirname(weights_path) if os.path.dirname(weights_path) else 'trained_weights'
		if os.path.exists(weight_dir):
			for f in os.listdir(weight_dir):
				if f.endswith('.pth'):
					print(f"  - {os.path.join(weight_dir, f)}")
		return None

	# 加载权重
	try:
		print(f"加载权重文件: {weights_path}")
		checkpoint = torch.load(weights_path)
		ld_net.load_state_dict(checkpoint)
		print("权重加载成功")
	except Exception as e:
		print(f"权重加载失败: {e}")
		print("尝试兼容性加载...")

		# 尝试兼容性加载（跳过不匹配的键）
		try:
			model_dict = ld_net.state_dict()
			pretrained_dict = {k: v for k, v in checkpoint.items()
							   if k in model_dict and model_dict[k].shape == v.shape}
			model_dict.update(pretrained_dict)
			ld_net.load_state_dict(model_dict)
			print(f"兼容性加载成功，加载了 {len(pretrained_dict)}/{len(checkpoint)} 层")
		except Exception as e2:
			print(f"兼容性加载也失败: {e2}")
			return None

	# 设置为评估模式
	ld_net.eval()

	# 前向推理
	with torch.no_grad():
		dehaze_image = ld_net(hazy_image)

	return dehaze_image


def image_haze_removel_batch(input_images, model_type='enhanced', weights_path=None):
	"""
    批量去雾函数
    Args:
        input_images: 输入图像列表
        model_type: 模型类型
        weights_path: 权重文件路径
    """
	# 只加载一次模型
	if model_type == 'enhanced':
		ld_net = lightdehazeNet.LightDehaze_Net_GlobalEnhanced(use_attention=True).cuda()
		default_weights = 'trained_weights/best_Global_Enhanced_LDNet.pth'
	else:
		ld_net = lightdehazeNet.LightDehaze_Net_GlobalEnhanced().cuda()
		default_weights = 'trained_weights/best_Global_Enhanced_LDNet.pth'

	if weights_path is None:
		weights_path = default_weights

	# 加载权重
	if os.path.exists(weights_path):
		checkpoint = torch.load(weights_path)
		try:
			ld_net.load_state_dict(checkpoint)
		except:
			# 兼容性加载
			model_dict = ld_net.state_dict()
			pretrained_dict = {k: v for k, v in checkpoint.items()
							   if k in model_dict and model_dict[k].shape == v.shape}
			model_dict.update(pretrained_dict)
			ld_net.load_state_dict(model_dict)
	else:
		print(f"权重文件不存在: {weights_path}")
		return []

	ld_net.eval()

	results = []
	with torch.no_grad():
		for input_image in input_images:
			# 图像预处理
			hazy_image = (np.asarray(input_image) / 255.0)
			hazy_image = torch.from_numpy(hazy_image).float()
			hazy_image = hazy_image.permute(2, 0, 1)
			hazy_image = hazy_image.cuda().unsqueeze(0)

			# 推理
			dehaze_image = ld_net(hazy_image)
			results.append(dehaze_image)

	return results


def auto_detect_model_type(weights_path):
	"""
    自动检测权重文件对应的模型类型
    """
	if not os.path.exists(weights_path):
		return None

	try:
		checkpoint = torch.load(weights_path, map_location='cpu')

		# 检查是否包含注意力模块的键
		attention_keys = ['attention1.temperature', 'attention2.temperature', 'gate1', 'gate2']
		has_attention = any(key in checkpoint for key in attention_keys)

		if has_attention:
			return 'enhanced'
		else:
			return 'original'
	except:
		return None


def list_available_weights():
	"""
    列出可用的权重文件
    """
	weight_dir = 'trained_weights'
	if not os.path.exists(weight_dir):
		print("权重目录不存在")
		return []

	weight_files = []
	for f in os.listdir(weight_dir):
		if f.endswith('.pth'):
			full_path = os.path.join(weight_dir, f)
			model_type = auto_detect_model_type(full_path)
			weight_files.append({
				'file': f,
				'path': full_path,
				'type': model_type
			})

	return weight_files


# 兼容性函数（保持原有接口）
def image_haze_removel_legacy(input_image):
	"""
    保持向后兼容的函数
    """
	return image_haze_removel(input_image, model_type='enhanced')


if __name__ == "__main__":
	# 测试函数
	print("可用的权重文件:")
	weights = list_available_weights()
	for w in weights:
		print(f"  - {w['file']} (类型: {w['type']})")

	# 测试加载
	if weights:
		test_weight = weights[0]
		print(f"\n测试加载权重: {test_weight['file']}")

		# 创建测试图像
		test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

		try:
			result = image_haze_removel(test_image,
										model_type=test_weight['type'],
										weights_path=test_weight['path'])
			if result is not None:
				print("推理成功！")
				print(f"输出形状: {result.shape}")
			else:
				print("推理失败")
		except Exception as e:
			print(f"推理出错: {e}")
	
		
