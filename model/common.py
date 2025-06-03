import torch
import torch.nn as nn

"""
    YOLO Module:
        --Conv模块:Conv + BN + SiLu
        --C2F模块:
"""


class Conv(nn.Module):
    """
    标准卷积模块：Conv2d + BatchNorm2d + SiLU激活函数。
    这是YOLOv8中许多模块的基础构建单元。
    """
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """
        初始化Conv模块。
        Args:
            c1 (int): 输入通道数。
            c2 (int): 输出通道数。
            k (int): 卷积核大小。
            s (int): 步长。
            p (int, optional): 填充大小。如果为None，则自动计算以保持输出尺寸。
            g (int): 分组数。
            act (bool): 是否使用SiLU激活函数。
        """
        super().__init__()
        # 计算填充大小，以确保输出尺寸与输入尺寸匹配（当步长为1时）
        if p is None:
            p = k // 2 if k % 2 == 1 else 0 # 自动计算填充
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity() # SiLU激活函数

    def forward(self, x):
        """
        前向传播。
        """
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """
    YOLOv8中C2f模块内部使用的瓶颈块。
    包含两个Conv层，可选残差连接。
    """
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        """
        初始化Bottleneck模块。
        Args:
            c1 (int): 输入通道数。
            c2 (int): 输出通道数。
            shortcut (bool): 是否使用残差连接。
            g (int): 分组数。
            e (float): 扩展率。
        """
        super().__init__()
        c_ = int(c2 * e)  # 隐藏通道数
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2 # 只有当输入输出通道数相同时才添加残差连接

    def forward(self, x):
        """
        前向传播。
        """
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


# C2f模块: YOLOv8骨干网络和颈部网络中的核心模块
class C2f(nn.Module):
    """
    YOLOv8的C2f模块。
    它将输入通道分成两部分，一部分直接通过，另一部分通过多个瓶颈块。
    所有瓶颈块的输出与直接通过的部分连接起来。
    """

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """
        初始化C2f模块。
        Args:
            c1 (int): 输入通道数。
            c2 (int): 输出通道数。
            n (int): 瓶颈块的数量。
            shortcut (bool): 是否使用残差连接。
            g (int): 分组数。
            e (float): 扩展率，用于中间通道数的计算。
        """
        super().__init__()
        self.c = int(c2 * e)  # 隐藏通道数
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)  # 初始卷积，将输入通道扩展为2*c
        self.cv2 = Conv(2 * self.c, c2, 1)  # 最终卷积，将拼接后的通道数调整为c2
        # 瓶颈块列表
        self.m = nn.ModuleList(
            [Bottleneck(self.c, self.c, shortcut, g, e=1.0) for _ in range(n)])  # YOLOv8的C2f中Bottleneck的e通常为1.0

    def forward(self, x):
        """
        前向传播。
        """
        # 将输入通过cv1，然后沿通道维度拆分为两部分
        x1, x2 = self.cv1(x).chunk(2, 1)
        # 遍历瓶颈块，并将每个瓶颈块的输出与x1拼接
        y = [x1]  # 初始化拼接列表，包含直接通过的部分
        for m in self.m:
            x2 = m(x2)  # x2通过瓶颈块
            y.append(x2)  # 将每个瓶颈块的输出添加到列表中

        # 将所有分支的输出在通道维度上拼接，然后通过cv2进行最终调整
        return self.cv2(torch.cat(y, 1))

# SPPF模块: 空间金字塔池化 - 快速版
class SPPF(nn.Module):
    """
    空间金字塔池化 - 快速版。
    通过并行执行多个不同核大小的最大池化操作来增加感受野。
    """
    def __init__(self, c1, c2, k=5):
        """
        初始化SPPF模块。
        Args:
            c1 (int): 输入通道数。
            c2 (int): 输出通道数。
            k (int): 池化核大小。
        """
        super().__init__()
        c_ = c1 // 2  # 隐藏通道数
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1) # 4是因为有原始输入和三次池化输出
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2) # 最大池化层

    def forward(self, x):
        """
        前向传播。
        """
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        # 将原始输入和三次池化后的结果在通道维度上拼接
        return self.cv2(torch.cat((x, y1, y2, y3), 1))