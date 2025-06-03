import torch
import torch.nn as nn
from model.Modules.Basic_Module import *


class V8_Backbone(nn.Module):
    """
    YOLOv8m骨干网络。
    负责从输入图像中提取多尺度特征，针对medium版本进行通道数和C2f重复次数的优化。
    """
    def __init__(self):
        super().__init__()
        # 初始干层
        self.stem = Conv(3, 48, 3, 2) # P1/2, (H/2, W/2, 48) - YOLOv8m起始通道数

        # P3 (8x下采样)
        self.stage1 = Conv(48, 96, 3, 2) # P2/4, (H/4, W/4, 96)
        self.c2f_1 = C2f(96, 96, n=2, shortcut=True) # P3, (H/8, W/8, 96) - YOLOv8m C2f n=2

        # P4 (16x下采样)
        self.stage2 = Conv(96, 192, 3, 2) # P3/8, (H/8, W/8, 192)
        self.c2f_2 = C2f(192, 192, n=4, shortcut=True) # P4, (H/16, W/16, 192) - YOLOv8m C2f n=4

        # P5 (32x下采样)
        self.stage3 = Conv(192, 384, 3, 2) # P4/16, (H/16, W/16, 384)
        self.c2f_3 = C2f(384, 384, n=4, shortcut=True) # P5, (H/32, W/32, 384) - YOLOv8m C2f n=4

        self.sppf = SPPF(384, 384, k=5) # 感受野增强

    def forward(self, x):
        """
        前向传播。
        Args:
            x (torch.Tensor): 输入图像张量。
        Returns:
            list: 骨干网络输出的多尺度特征图列表 [p3, p4, p5_sppf]。
        """
        x = self.stem(x)
        x = self.stage1(x)
        p3 = self.c2f_1(x) # P3特征图 (8x下采样)

        x = self.stage2(p3)
        p4 = self.c2f_2(x) # P4特征图 (16x下采样)

        x = self.stage3(p4)
        p5 = self.c2f_3(x) # P5特征图 (32x下采样)
        p5_sppf = self.sppf(p5) # 对P5应用SPPF

        return [p3, p4, p5_sppf] # 返回P3, P4, P5_SPPF给颈部网络
