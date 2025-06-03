import torch
import torch.nn as nn
from model.Modules.Basic_Module import *


# --- 4. 检测头 (Head) ---

class Detect(nn.Module):
    """
    YOLOv8m的检测头（解耦头）。
    为每个尺度生成分类和回归预测，通道数与YOLOv8m颈部网络输出匹配。
    """

    def __init__(self, nc=80, ch=(96, 192, 384)):  # nc: 类别数, ch: 颈部网络输出的通道数
        """
        初始化Detect模块。
        Args:
            nc (int): 类别数。
            ch (tuple): 颈部网络输出的各层通道数。
        """
        super().__init__()
        self.nc = nc  # 类别数
        self.nl = len(ch)  # 检测层数 (通常是3个尺度)
        self.no = nc + 4  # 输出维度 (类别数 + 4个边界框坐标)

        # 分类分支
        self.cls_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        # 回归分支
        self.reg_convs = nn.ModuleList()
        self.reg_preds = nn.ModuleList()

        for i in range(self.nl):  # 遍历每个尺度
            # 分类分支的卷积层和预测层
            self.cls_convs.append(nn.Sequential(
                Conv(ch[i], ch[i], 3, 1),
                Conv(ch[i], ch[i], 3, 1)
            ))
            self.cls_preds.append(nn.Conv2d(ch[i], self.nc, 1))

            # 回归分支的卷积层和预测层
            self.reg_convs.append(nn.Sequential(
                Conv(ch[i], ch[i], 3, 1),
                Conv(ch[i], ch[i], 3, 1)
            ))
            self.reg_preds.append(nn.Conv2d(ch[i], 4, 1))  # 4个边界框坐标 (x, y, w, h)

    def forward(self, x):
        """
        前向传播。
        Args:
            x (list): 颈部网络输出的多尺度特征图列表。
        Returns:
            list: 包含每个尺度预测结果的列表。
                  每个预测结果是一个张量，形状为 (batch_size, (4 + nc), H, W)
        """
        outputs = []
        for i in range(self.nl):
            # 获取当前尺度的特征图
            feature = x[i]

            # 分类分支
            cls_feature = self.cls_convs[i](feature)
            cls_output = self.cls_preds[i](cls_feature)

            # 回归分支
            reg_feature = self.reg_convs[i](feature)
            reg_output = self.reg_preds[i](reg_feature)

            # 将分类和回归结果拼接
            # 这里的输出形状是 (batch_size, (4 + nc), H, W)
            output = torch.cat([reg_output, cls_output], 1)
            outputs.append(output)

        # 在实际的YOLOv8中，这里会对输出进行reshape和后处理
        # 例如，将 (N, C, H, W) 转换为 (N, H*W, C) 或 (N, num_boxes, C)
        # 并进行解码以获取实际的边界框坐标、置信度和类别概率
        return outputs