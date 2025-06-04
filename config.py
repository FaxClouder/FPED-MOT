# config.py

import os
import torch # 导入 torch 以便在 DEVICE 中使用 cuda.is_available()

# --- 路径配置 ---
# 获取当前脚本所在目录的父目录，作为项目根目录
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 根据您的图片结构修改数据目录
# 假设 'data' 目录与 'config' 目录在同一层级
DATA_ROOT = os.path.join(os.path.dirname(_PROJECT_ROOT), "data", "PED")

# 训练图像和标签目录
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "images", "train")
TRAIN_LABEL_DIR = os.path.join(DATA_ROOT, "labels", "train")

# 验证图像和标签目录
VAL_IMG_DIR = os.path.join(DATA_ROOT, "images", "val")
VAL_LABEL_DIR = os.path.join(DATA_ROOT, "labels", "val")

# 测试图像和标签目录 (如果需要进行单独的测试阶段)
TEST_IMG_DIR = os.path.join(DATA_ROOT, "images", "test")
TEST_LABEL_DIR = os.path.join(DATA_ROOT, "labels", "test")


# 模型保存路径
SAVE_DIR = os.path.join(_PROJECT_ROOT, "runs")
os.makedirs(SAVE_DIR, exist_ok=True)
BEST_MODEL_PATH = os.path.join(SAVE_DIR, "best_yolov8m_model.pth")

# --- 模型超参数 ---
IMG_SIZE = (640, 640) # 输入图像尺寸 (宽度, 高度)

# 定义类别名称列表，并根据其长度设置 NUM_CLASSES
CLASS_NAMES = [
    "bike", "people"
]
NUM_CLASSES = len(CLASS_NAMES) # 根据 CLASS_NAMES 自动设置类别数

MODEL_NAME = "yolov8m" # 模型名称 (例如 yolov8n, yolov8s, yolov8m 等)

# --- 训练超参数 ---
BATCH_SIZE = 16      # 训练批次大小
EPOCHS = 500          # 训练的总 epoch 数 (示例用，实际训练需要更多)
LEARNING_RATE = 0.001 # 初始学习率
WEIGHT_DECAY = 0.0005 # 优化器的权重衰减
MOMENTUM = 0.937      # SGD优化器的动量 (如果使用AdamW，此参数可能不适用)

# --- 损失函数权重 ---
BOX_LOSS_WEIGHT = 7.5  # 边界框损失权重
CLS_LOSS_WEIGHT = 0.5  # 分类损失权重
OBJ_LOSS_WEIGHT = 1.5  # 目标置信度损失权重 (在YOLOv8中通常与DFL或分类结合)

# --- 推理/评估超参数 ---
CONF_THRESHOLD = 0.25 # 推理时的目标置信度阈值
IOU_THRESHOLD = 0.45  # 推理时的NMS IoU阈值

# --- 其他配置 ---
NUM_WORKERS = os.cpu_count() // 2 # DataLoader 的工作进程数
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # 训练设备
