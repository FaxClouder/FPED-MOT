import torch
import torch.nn as nn
import numpy as np

from Modules.Basic_Module import *
from backbone.v8_backbone import *
from neck.v8_neck import *
from head.v8_head import *

# --- 5. 组装YOLOv8m完整模型 ---

class YOLOv8mModel(nn.Module):
    """
    YOLOv8m模型的完整骨架。
    将骨干网络、颈部网络和检测头组合起来，并针对medium版本进行配置。
    """
    def __init__(self, nc=80):
        """
        初始化YOLOv8m模型。
        Args:
            nc (int): 类别数。
        """
        super().__init__()
        # 骨干网络
        self.backbone = V8_Backbone()

        # 颈部网络
        # 骨干网络输出的P3, P4, P5的通道数，对应YOLOv8m的典型配置
        neck_in_channels = [96, 192, 384]
        self.neck = Neck(in_channels=neck_in_channels)

        # 检测头
        # 检测头需要颈部网络输出的各层通道数，与neck_in_channels相同
        self.head = Detect(nc=nc, ch=neck_in_channels)

    def forward(self, x):
        """
        前向传播。
        Args:
            x (torch.Tensor): 输入图像张量。
        Returns:
            list: 模型的最终预测结果列表。
        """
        # 骨干网络前向传播，获取多尺度特征 P3, P4, P5_SPPF
        features_from_backbone = self.backbone(x) # [p3, p4, p5_sppf]

        # 颈部网络前向传播，融合特征
        fused_features = self.neck(features_from_backbone) # [p3_out, p4_out, p5_out]

        # 检测头前向传播，生成预测
        predictions = self.head(fused_features)

        return predictions