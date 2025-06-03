import torch
from torch.utils.data import Dataset
import os
import cv2
import numpy as np


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """
    保持纵横比的resize，加padding至指定大小。
    """
    shape = img.shape[:2]  # 原始 h, w
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # padding宽高
    dw /= 2
    dh /= 2

    # resize
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    # padding
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return img, r, (dw, dh)


class YOLOv8Dataset(Dataset):
    """
    一个用于加载YOLOv8格式标注数据集的PyTorch Dataset。
    标注文件格式: 每行 'class_id x_center y_center width height' (归一化坐标)。
    """

    def __init__(self, img_dir, label_dir, img_size=(640, 640), transform=None):
        """
        初始化数据集。
        Args:
            img_dir (str): 包含图像文件的目录路径。
            label_dir (str): 包含YOLOv8格式标注文件的目录路径。
            img_size (tuple): 输出图像的尺寸 (width, height)。
            transform (callable, optional): 应用于图像和标注的额外转换。
        """
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.transform = transform

        # 获取所有图像文件的列表
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
        self.img_files.sort()  # 确保顺序一致

        print(f"Found {len(self.img_files)} images in {img_dir}")

    def __len__(self):
        """
        返回数据集中样本的总数。
        """
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(self.label_dir, label_name)

        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not load image {img_path}. Skipping.")
            return torch.zeros(3, *self.img_size), torch.empty(0, 5)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_h, original_w = img.shape[:2]

        # 加载标签
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = list(map(float, line.strip().split()))
                    if len(parts) == 5:
                        labels.append(parts)
        labels = np.array(labels, dtype=np.float32)

        # 图像Letterbox缩放
        img_resized, r, (dw, dh) = letterbox(img, self.img_size)

        # 同步调整标签坐标
        if labels.shape[0] > 0:
            labels[:, 1] = labels[:, 1] * original_w * r + dw
            labels[:, 2] = labels[:, 2] * original_h * r + dh
            labels[:, 3] = labels[:, 3] * original_w * r
            labels[:, 4] = labels[:, 4] * original_h * r

            # 再归一化
            labels[:, 1:] /= torch.tensor([self.img_size[1], self.img_size[0],
                                           self.img_size[1], self.img_size[0]], dtype=np.float32)

        # 图像转Tensor
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0

        # 应用增强
        if self.transform:
            img_tensor, labels = self.transform(img_tensor, labels)

        target_tensor = torch.from_numpy(labels).float() if labels.shape[0] > 0 else torch.empty(0, 5)
        return img_tensor, target_tensor
