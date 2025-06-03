import torch
import torch.nn as nn
from model.Modules.Basic_Module import *

class Neck(nn.Module):
    """
    YOLOv8m颈部网络（PAFPN结构）。
    实现自顶向下（FPN）和自底向上（PAN）的特征融合，针对medium版本进行通道数和C2f重复次数的优化。
    """
    def __init__(self, in_channels):
        """
        初始化Neck模块。
        Args:
            in_channels (list): 骨干网络输出的各层通道数列表 [c3, c4, c5]。
        """
        super().__init__()
        c3, c4, c5 = in_channels # 假设输入通道为 [96, 192, 384] for YOLOv8m

        # FPN 路径 (自顶向下)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")

        # P5 -> P4_up
        self.conv_p5_to_p4 = Conv(c5, c4, 1, 1) # 调整P5通道数以与P4拼接
        self.c2f_p4_fused_fpn = C2f(c4 * 2, c4, n=2, shortcut=False) # 融合P4和上采样P5, YOLOv8m C2f n=2

        # P4_fused_fpn -> P3_up
        self.conv_p4_to_p3 = Conv(c4, c3, 1, 1) # 调整P4_fused_fpn通道数以与P3拼接
        self.c2f_p3_fused_fpn = C2f(c3 * 2, c3, n=2, shortcut=False) # 融合P3和上采样P4, YOLOv8m C2f n=2

        # PAN 路径 (自底向上)
        # P3_fused_fpn -> P4_down
        self.conv_p3_to_p4_down = Conv(c3, c3, 3, 2) # 下采样P3_fused_fpn
        self.c2f_p4_fused_pan = C2f(c3 + c4, c4, n=2, shortcut=False) # 融合下采样P3和P4_fused_fpn, YOLOv8m C2f n=2

        # P4_fused_pan -> P5_down
        self.conv_p4_to_p5_down = Conv(c4, c4, 3, 2) # 下采样P4_fused_pan
        self.c2f_p5_fused_pan = C2f(c4 + c5, c5, n=2, shortcut=False) # 融合下采样P4和P5_sppf, YOLOv8m C2f n=2

    def forward(self, features):
        """
        前向传播。
        Args:
            features (list): 骨干网络输出的多尺度特征图列表 [p3, p4, p5_sppf]。
        Returns:
            list: 融合后的多尺度特征图列表 [p3_out, p4_out, p5_out]。
        """
        p3_in, p4_in, p5_in = features # 对应骨干网络的P3, P4, P5_SPPF输出

        # FPN 路径 (自顶向下)
        # P5 -> P4
        p5_up = self.up(self.conv_p5_to_p4(p5_in))
        p4_fused_fpn = self.c2f_p4_fused_fpn(torch.cat([p4_in, p5_up], 1))

        # P4 -> P3
        p4_up = self.up(self.conv_p4_to_p3(p4_fused_fpn))
        p3_fused_fpn = self.c2f_p3_fused_fpn(torch.cat([p3_in, p4_up], 1))

        # PAN 路径 (自底向上)
        # P3 -> P4
        p3_to_p4_down = self.conv_p3_to_p4_down(p3_fused_fpn)
        p4_fused_pan = self.c2f_p4_fused_pan(torch.cat([p3_to_p4_down, p4_fused_fpn], 1))

        # P4 -> P5
        p4_to_p5_down = self.conv_p4_to_p5_down(p4_fused_pan)
        p5_fused_pan = self.c2f_p5_fused_pan(torch.cat([p4_to_p5_down, p5_in], 1)) # 注意这里与原始P5_in拼接

        # 颈部网络最终输出的三个尺度特征图
        return [p3_fused_fpn, p4_fused_pan, p5_fused_pan]