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
import numpy as np
from torchvision import transforms
import torch.nn.functional as F

# 导入增强模型（假设保存在enhanced_lightdehazeNet.py中）
try:
    from lightdehazeNet import (
        LightDehaze_Net_GlobalEnhanced,
        LightDehaze_Net_Hybrid,
        count_parameters
    )

    ENHANCED_MODELS_AVAILABLE = True
except ImportError:
    print("增强模型文件未找到，仅使用原始模型")
    ENHANCED_MODELS_AVAILABLE = False


class SSIMLoss(nn.Module):
    """SSIM损失函数，用于结构相似性优化"""

    def __init__(self, window_size=11, channel=3):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.channel = channel

        # 创建高斯窗口
        window = self.create_window(window_size, channel)
        self.register_buffer('window', window)

    def create_window(self, window_size, channel):
        def gaussian(window_size, sigma):
            gauss = torch.Tensor([np.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
                                  for x in range(window_size)])
            return gauss / gauss.sum()

        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def ssim(self, img1, img2):
        mu1 = F.conv2d(img1, self.window, padding=self.window_size // 2, groups=self.channel)
        mu2 = F.conv2d(img2, self.window, padding=self.window_size // 2, groups=self.channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, self.window, padding=self.window_size // 2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, self.window, padding=self.window_size // 2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, self.window, padding=self.window_size // 2, groups=self.channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()

    def forward(self, img1, img2):
        return 1 - self.ssim(img1, img2)


class PerceptualLoss(nn.Module):
    """基于VGG的感知损失"""

    def __init__(self):
        super(PerceptualLoss, self).__init__()
        # 使用VGG16的前几层
        vgg = torchvision.models.vgg16(pretrained=True).features
        self.feature_extractor = nn.Sequential()

        # 选择多个层来提取特征
        self.layers = [3, 8, 15, 22]  # relu1_2, relu2_2, relu3_3, relu4_3

        for i, layer in enumerate(vgg):
            self.feature_extractor.add_module(str(i), layer)
            if i == max(self.layers):
                break

        # 冻结参数
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.mse_loss = nn.MSELoss()

    def forward(self, pred, target):
        pred_features = []
        target_features = []

        x_pred = pred
        x_target = target

        for i, layer in enumerate(self.feature_extractor):
            x_pred = layer(x_pred)
            x_target = layer(x_target)

            if i in self.layers:
                pred_features.append(x_pred)
                target_features.append(x_target)

        # 计算多层特征损失
        loss = 0
        weights = [1.0, 0.8, 0.6, 0.4]  # 不同层的权重

        for pred_feat, target_feat, weight in zip(pred_features, target_features, weights):
            loss += weight * self.mse_loss(pred_feat, target_feat)

        return loss


class EdgePreservingLoss(nn.Module):
    """边缘保持损失"""

    def __init__(self):
        super(EdgePreservingLoss, self).__init__()

        # Sobel算子
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)

        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3).repeat(3, 1, 1, 1))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3).repeat(3, 1, 1, 1))

    def forward(self, pred, target):
        # 计算梯度
        pred_grad_x = F.conv2d(pred, self.sobel_x, padding=1, groups=3)
        pred_grad_y = F.conv2d(pred, self.sobel_y, padding=1, groups=3)
        pred_grad = torch.sqrt(pred_grad_x ** 2 + pred_grad_y ** 2 + 1e-8)

        target_grad_x = F.conv2d(target, self.sobel_x, padding=1, groups=3)
        target_grad_y = F.conv2d(target, self.sobel_y, padding=1, groups=3)
        target_grad = torch.sqrt(target_grad_x ** 2 + target_grad_y ** 2 + 1e-8)

        return F.mse_loss(pred_grad, target_grad)


class PSNROptimizedLoss(nn.Module):
    """专门为PSNR优化设计的复合损失函数"""

    def __init__(self, use_perceptual=True):
        super(PSNROptimizedLoss, self).__init__()

        self.mse_loss = nn.MSELoss()
        self.ssim_loss = SSIMLoss()
        self.edge_loss = EdgePreservingLoss()

        self.use_perceptual = use_perceptual
        if use_perceptual:
            try:
                self.perceptual_loss = PerceptualLoss()
            except:
                print("警告: 无法加载VGG模型，跳过感知损失")
                self.use_perceptual = False

        # 损失权重
        self.w_mse = 1.0  # MSE权重最高，直接影响PSNR
        self.w_ssim = 0.3  # SSIM权重，保持结构
        self.w_edge = 0.2  # 边缘权重，保持细节
        self.w_perceptual = 0.1  # 感知权重，提升视觉质量

    def forward(self, pred, target, hazy_input=None):
        # 主要损失：MSE（直接优化PSNR）
        mse_loss = self.mse_loss(pred, target)

        # SSIM损失（结构相似性）
        ssim_loss = self.ssim_loss(pred, target)

        # 边缘保持损失
        edge_loss = self.edge_loss(pred, target)

        # 组合损失
        total_loss = (self.w_mse * mse_loss +
                      self.w_ssim * ssim_loss +
                      self.w_edge * edge_loss)

        # 感知损失（如果可用）
        if self.use_perceptual:
            perceptual_loss = self.perceptual_loss(pred, target)
            total_loss += self.w_perceptual * perceptual_loss

        # 额外的正则化项：确保去雾结果不过饱和
        if hazy_input is not None:
            # 防止过度增强
            enhancement_reg = F.mse_loss(torch.clamp(pred, 0, 1), pred) * 0.01
            total_loss += enhancement_reg

        return total_loss


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_model(args):
    """根据参数选择合适的模型"""
    model_type = args.get("model_type")

    if model_type == "enhanced" and args.get("use_attention", False):
        print("使用增强版LightDehaze网络（带注意力机制）")
        model = lightdehazeNet.LightDehaze_Net_Enhanced(use_attention=True).cuda()
        model_name = "Enhanced_LDNet"

    elif model_type == "global" and ENHANCED_MODELS_AVAILABLE:
        print("使用全局特征增强版LightDehaze网络")
        model = LightDehaze_Net_GlobalEnhanced(
            use_global_attention=args.get("use_global_attention", True),
            use_multiscale=args.get("use_multiscale", True),
            use_context=args.get("use_context", True)
        ).cuda()
        model_name = "Global_Enhanced_LDNet"

    elif model_type == "hybrid" and ENHANCED_MODELS_AVAILABLE:
        print("使用混合架构LightDehaze网络（CNN + Transformer）")
        model = LightDehaze_Net_Hybrid(
            use_transformer=args.get("use_transformer", True)
        ).cuda()
        model_name = "Hybrid_LDNet"
    elif model_type == "lsnet":
        print("使用 LS-Net 去雾模型 (Large Kernel + Spatial Attention)")
        model = lightdehazeNet.LSNet_Dehaze(base_ch=int(args.get("base_ch", 64))).cuda()
        model_name = "LSNet_Dehaze"
    return model, model_name


def setup_optimizer(model, args, model_type):
    """根据模型类型设置优化器"""
    learning_rate = float(args["learning_rate"])

    if model_type == "lsnet":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=1e-4)
        print(f"使用 LS-Net 专用优化器 AdamW，学习率: {learning_rate}, weight_decay=0.01")

    return optimizer


def calculate_loss(dehaze_image, hazefree_image, hazy_image, loss_type="combined", psnr_loss=None):
    """计算损失函数，支持多种损失类型"""
    if loss_type == "psnr_optimized":
        if psnr_loss is None:
            psnr_loss = PSNROptimizedLoss(use_perceptual=True)
        return psnr_loss(dehaze_image, hazefree_image, hazy_image)

    elif loss_type == "psnr_simple":
        # 简化版PSNR优化损失
        mse_loss = nn.MSELoss()(dehaze_image, hazefree_image)
        ssim_loss = SSIMLoss()(dehaze_image, hazefree_image)
        return 0.8 * mse_loss + 0.2 * ssim_loss

    elif loss_type == "mse":
        return nn.MSELoss()(dehaze_image, hazefree_image)

    elif loss_type == "l1":
        return nn.L1Loss()(dehaze_image, hazefree_image)

    elif loss_type == "combined":
        mse_loss = nn.MSELoss()(dehaze_image, hazefree_image)
        l1_loss = nn.L1Loss()(dehaze_image, hazefree_image)

        # 感知损失（简化版）
        perceptual_loss = nn.MSELoss()(
            torch.mean(dehaze_image, dim=[2, 3]),
            torch.mean(hazefree_image, dim=[2, 3])
        )

        return 0.7 * mse_loss + 0.2 * l1_loss + 0.1 * perceptual_loss

    else:
        return nn.MSELoss()(dehaze_image, hazefree_image)


def calculate_psnr(img1, img2):
    """计算PSNR值"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def train(args):
    # 选择模型
    model, model_name = get_model(args)
    model_type = args.get("model_type")

    # 打印模型信息
    if ENHANCED_MODELS_AVAILABLE:
        total_params = count_parameters(model)
    else:
        total_params = lightdehazeNet.count_parameters(model)
    print(f"模型总参数数量: {total_params:,}")

    # 如果指定了预训练权重，则加载
    if args.get("pretrained_weights", None):
        if os.path.exists(args["pretrained_weights"]):
            print(f"加载预训练权重: {args['pretrained_weights']}")
            try:
                checkpoint = torch.load(args["pretrained_weights"])
                model_dict = model.state_dict()

                # 过滤不匹配的键
                pretrained_dict = {k: v for k, v in checkpoint.items()
                                   if k in model_dict and model_dict[k].shape == v.shape}

                # 更新模型字典
                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict)

                print(f"成功加载 {len(pretrained_dict)}/{len(checkpoint)} 层的权重")
            except Exception as e:
                print(f"加载权重时出错: {e}")
                print("将使用随机初始化")
        else:
            print(f"预训练权重文件不存在: {args['pretrained_weights']}")

    # 应用权重初始化
    model.apply(weights_init)

    # 数据加载器
    batch_size = int(args.get("batch_size", 8))


    training_data = image_data_loader.hazy_data_loader(args["train_original"],
                                                       args["train_hazy"])
    validation_data = image_data_loader.hazy_data_loader(args["train_original"],
                                                         args["train_hazy"], mode="val")

    training_data_loader = torch.utils.data.DataLoader(training_data,
                                                       batch_size=batch_size,
                                                       shuffle=True,
                                                       num_workers=12,
                                                       pin_memory=True)
    validation_data_loader = torch.utils.data.DataLoader(validation_data,
                                                         batch_size=batch_size,
                                                         shuffle=True,
                                                         num_workers=12,
                                                         pin_memory=True)

    # 设置优化器
    optimizer = setup_optimizer(model, args, model_type)

    # 学习率调度器
    scheduler_type = args.get("scheduler", "step")
    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(args["epochs"]), eta_min=1e-7
        )

    # 损失函数类型
    loss_type = args.get("loss_type", "psnr_optimized")
    print(f"使用 {loss_type} 损失函数")

    # 初始化PSNR优化损失函数
    psnr_loss = None
    if loss_type in ["psnr_optimized"]:
        psnr_loss = PSNROptimizedLoss(use_perceptual=True).cuda()

    # 混合精度训练
    use_amp = args.get("use_amp", False)
    if use_amp:
        scaler = torch.cuda.amp.GradScaler()
        print("启用混合精度训练")

    model.train()

    # 创建保存目录
    os.makedirs("trained_weights", exist_ok=True)
    os.makedirs("training_data_captures", exist_ok=True)

    num_of_epochs = int(args["epochs"])
    best_loss = float('inf')
    best_psnr = 0.0
    patience_counter = 0
    max_patience = int(args.get("early_stopping_patience", 20))

    print(f"开始训练，共 {num_of_epochs} 个epoch")
    print(f"早停耐心值: {max_patience}")

    # 训练循环
    for epoch in range(num_of_epochs):
        if epoch < 2:
            warmup_factor = (epoch + 1) / 5
            for param_group in optimizer.param_groups:
                param_group['lr'] = float(args["learning_rate"]) * warmup_factor
            print(f"  [Warmup] 学习率: {optimizer.param_groups[0]['lr']:.6f}")
        elif epoch == 2:
            # 恢复正常学习率
            for param_group in optimizer.param_groups:
                param_group['lr'] = float(args["learning_rate"])
            print(f"  [Normal] 学习率: {optimizer.param_groups[0]['lr']:.6f}")
        epoch_loss = 0.0
        epoch_psnr = 0.0
        num_batches = 0

        print(f"\nEpoch [{epoch + 1}/{num_of_epochs}]")

        # 训练阶段
        model.train()
        epoch_start_time = time.time()

        for iteration, (hazefree_image, hazy_image) in enumerate(training_data_loader):
            hazefree_image = hazefree_image.cuda(non_blocking=True)
            hazy_image = hazy_image.cuda(non_blocking=True)

            optimizer.zero_grad()

            if use_amp:
                with torch.cuda.amp.autocast():
                    dehaze_image = model(hazy_image)
                    loss = calculate_loss(dehaze_image, hazefree_image, hazy_image, loss_type, psnr_loss)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                dehaze_image = model(hazy_image)
                loss = calculate_loss(dehaze_image, hazefree_image, hazy_image, loss_type, psnr_loss)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            # 计算PSNR
            with torch.no_grad():
                psnr = calculate_psnr(dehaze_image, hazefree_image)
                epoch_psnr += psnr.item()

            epoch_loss += loss.item()
            num_batches += 1

            if ((iteration + 1) % 10) == 0:
                print(f"  Iteration {iteration + 1:4d}, Loss: {loss.item():.6f}, PSNR: {psnr.item():.2f}dB")

            if ((iteration + 1) % 200) == 0:
                torch.save(model.state_dict(),
                           f"trained_weights/Epoch{epoch}_{model_name}_iter{iteration + 1}.pth")

        epoch_time = time.time() - epoch_start_time
        avg_train_loss = epoch_loss / num_batches
        avg_train_psnr = epoch_psnr / num_batches
        print(
            f"  训练时间: {epoch_time:.2f}s, 平均训练损失: {avg_train_loss:.6f}, 平均训练PSNR: {avg_train_psnr:.2f}dB")

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_psnr = 0.0
        val_batches = 0

        with torch.no_grad():
            for iter_val, (hazefree_image, hazy_image) in enumerate(validation_data_loader):
                hazefree_image = hazefree_image.cuda(non_blocking=True)
                hazy_image = hazy_image.cuda(non_blocking=True)

                if use_amp:
                    with torch.cuda.amp.autocast():
                        dehaze_image = model(hazy_image)
                        loss = calculate_loss(dehaze_image, hazefree_image, hazy_image, loss_type, psnr_loss)
                else:
                    dehaze_image = model(hazy_image)
                    loss = calculate_loss(dehaze_image, hazefree_image, hazy_image, loss_type, psnr_loss)

                psnr = calculate_psnr(dehaze_image, hazefree_image)

                val_loss += loss.item()
                val_psnr += psnr.item()
                val_batches += 1

                # 保存前几个验证样本的可视化结果
                if iter_val < 5:
                    torchvision.utils.save_image(torch.cat((hazy_image, dehaze_image, hazefree_image), 0),
                                                 f"training_data_captures/epoch{epoch}_val{iter_val + 1}.jpg")

        avg_val_loss = val_loss / val_batches if val_batches > 0 else float('inf')
        avg_val_psnr = val_psnr / val_batches if val_batches > 0 else 0.0
        print(f"  平均验证损失: {avg_val_loss:.6f}, 平均验证PSNR: {avg_val_psnr:.2f}dB")

        # 保存最好的模型（基于PSNR）
        if avg_val_psnr > best_psnr:
            best_psnr = avg_val_psnr
            best_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"trained_weights/best_{model_name}_psnr.pth")
            print(f"  保存最佳模型 (验证PSNR: {best_psnr:.2f}dB)")
        else:
            patience_counter += 1
            print(f"  验证PSNR未改善，耐心计数: {patience_counter}/{max_patience}")

        # 每个epoch结束后保存模型
        torch.save(model.state_dict(), f"trained_weights/epoch{epoch}_{model_name}.pth")

        # 更新学习率
        if scheduler_type == "plateau":
            scheduler.step(avg_val_loss)
        else:
            scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        print(f"  当前学习率: {current_lr:.8f}")

        # 早停检查
        if patience_counter >= max_patience:
            print(f"\n验证PSNR连续 {max_patience} 个epoch未改善，提前停止训练")
            break

        # 显存清理
        if (epoch + 1) % 10 == 0:
            torch.cuda.empty_cache()

    # 保存最终模型
    torch.save(model.state_dict(), f"trained_weights/final_{model_name}.pth")
    print(f"\n训练完成！最终模型已保存为: final_{model_name}.pth")
    print(f"最佳验证PSNR: {best_psnr:.2f}dB")
    print(f"对应验证损失: {best_loss:.6f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-th", "--train_hazy", required=True, help="path to hazy training images")
    ap.add_argument("-to", "--train_original", required=True, help="path to original training images")
    ap.add_argument("-e", "--epochs", required=True, help="number of epochs for training")
    ap.add_argument("-lr", "--learning_rate", required=True, help="learning rate for training")

    # 模型选择参数
    ap.add_argument("-mt", "--model_type", default="lsnet",
                    choices=["original", "enhanced", "global", "hybrid", "lsnet"],
                    help="choose model type: original / enhanced / global / hybrid / lsnet")
    ap.add_argument("-bc", "--base_ch", default="64",
                    help="base channel width for LS-Net (default: 64)")

    # 原有参数
    ap.add_argument("-att", "--use_attention", action="store_true",
                    help="use enhanced model with attention mechanism")
    ap.add_argument("-bs", "--batch_size", default="8",
                    help="batch size for training (default: 8)")
    ap.add_argument("-pw", "--pretrained_weights", default=None,
                    help="path to pretrained weights file")
    ap.add_argument("-ls", "--lr_step", default="30",
                    help="learning rate decay step (default: 30)")

    # 新增全局特征相关参数
    ap.add_argument("-ga", "--use_global_attention", action="store_true", default=True,
                    help="use global attention in global model")
    ap.add_argument("-ms", "--use_multiscale", action="store_true", default=True,
                    help="use multiscale feature aggregation")
    ap.add_argument("-ctx", "--use_context", action="store_true", default=True,
                    help="use global context modeling")
    ap.add_argument("-tf", "--use_transformer", action="store_true", default=True,
                    help="use transformer in hybrid model")

    # 训练优化参数
    ap.add_argument("-lt", "--loss_type", default="psnr_optimized",
                    choices=["mse", "l1", "combined", "psnr_optimized", "psnr_simple"],
                    help="loss function type")
    ap.add_argument("-sc", "--scheduler", default="step",
                    choices=["step", "cosine", "plateau"],
                    help="learning rate scheduler type")
    ap.add_argument("-amp", "--use_amp", action="store_true",
                    help="use automatic mixed precision training")
    ap.add_argument("-esp", "--early_stopping_patience", default="20",
                    help="early stopping patience (default: 20)")
    ap.add_argument("-aab", "--auto_adjust_batch", action="store_true", default=True,
                    help="automatically adjust batch size for complex models")

    args = vars(ap.parse_args())

    print("=" * 60)
    print("训练配置:")
    for key, value in args.items():
        print(f"  {key}: {value}")
    print("=" * 60)


    train(args)