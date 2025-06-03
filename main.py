# main.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import cv2
import numpy as np
import time
from tqdm import tqdm  # 用于显示进度条
import shutil  # 用于清理虚拟数据

# 从自定义模块导入
from config import (
    IMG_SIZE, NUM_CLASSES, BATCH_SIZE, EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    BOX_LOSS_WEIGHT, CLS_LOSS_WEIGHT, OBJ_LOSS_WEIGHT,
    CONF_THRESHOLD, IOU_THRESHOLD, NUM_WORKERS, DEVICE,
    TRAIN_IMG_DIR, TRAIN_LABEL_DIR, VAL_IMG_DIR, VAL_LABEL_DIR,
    BEST_MODEL_PATH, DATA_ROOT  # 导入 DATA_ROOT
)
from model.FYOLOv8m import YOLOv8mModel
from data.YOLO_Dataset import YOLOv8Dataset, letterbox
from data.collate_fn import custom_collate_fn
from utils.loss import YOLOv8Loss
from utils.general import decode_predictions, non_max_suppression
from utils.metrics import calculate_map  # 导入mAP计算骨架


# --- 辅助函数：创建虚拟数据 ---
def create_dummy_data(num_train_images=10, num_val_images=3):
    """
    创建用于测试的虚拟图像和标注文件。
    这些文件将直接创建在 config.py 中定义的 TRAIN/VAL 路径下。
    """
    print("Creating dummy data for demonstration...")
    # 确保训练和验证图像/标签目录存在
    os.makedirs(TRAIN_IMG_DIR, exist_ok=True)
    os.makedirs(TRAIN_LABEL_DIR, exist_ok=True)
    os.makedirs(VAL_IMG_DIR, exist_ok=True)
    os.makedirs(VAL_LABEL_DIR, exist_ok=True)

    # 训练数据
    for i in range(num_train_images):
        dummy_img = np.zeros((480, 640, 3), dtype=np.uint8) + (i * 20 % 255)
        cv2.imwrite(os.path.join(TRAIN_IMG_DIR, f"train_img_{i}.jpg"), dummy_img)
        with open(os.path.join(TRAIN_LABEL_DIR, f"train_img_{i}.txt"), "w") as f:
            f.write(f"0 0.5 0.5 0.2 0.3\n")  # 类别0，中心 (0.5, 0.5)，宽0.2，高0.3
            f.write(f"1 0.2 0.8 0.1 0.15\n")  # 类别1，中心 (0.2, 0.8)，宽0.1，高0.15

    # 验证数据
    for i in range(num_val_images):
        dummy_img = np.zeros((480, 640, 3), dtype=np.uint8) + (i * 50 % 255)
        cv2.imwrite(os.path.join(VAL_IMG_DIR, f"val_img_{i}.jpg"), dummy_img)
        with open(os.path.join(VAL_LABEL_DIR, f"val_img_{i}.txt"), "w") as f:
            f.write(f"0 0.6 0.4 0.3 0.2\n")  # 类别0，中心 (0.6, 0.4)，宽0.3，高0.2
    print("Dummy data created.")


# --- 训练函数 ---
def train(model, train_dataloader, val_dataloader, optimizer, loss_fn, scheduler, epochs, device):
    """
    执行模型的训练过程。
    """
    print("\n--- Starting Training ---")
    best_val_loss = float('inf')  # 用于保存最佳模型

    for epoch in range(epochs):
        start_time = time.time()

        # 训练一个epoch
        model.train()  # 设置模型为训练模式
        total_train_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs} Training")

        for batch_idx, (images, targets) in enumerate(progress_bar):
            images = images.to(device)
            targets = targets.to(device)  # targets: [batch_idx, class_id, x_c, y_c, w, h]

            # 前向传播
            predictions = model(images)

            # 计算损失
            loss, loss_dict = loss_fn(predictions, targets)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item(),
                                     box_loss=loss_dict['box_loss'],
                                     cls_loss=loss_dict['cls_loss'],
                                     obj_loss=loss_dict['obj_loss'])

        avg_train_loss = total_train_loss / len(train_dataloader)
        scheduler.step()  # 学习率调度

        print(f"\nEpoch {epoch + 1}/{epochs} finished.")
        print(f"  Average Training Loss: {avg_train_loss:.4f}")
        print(f"  Current Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        # 验证
        print("Running validation...")
        val_results, val_ground_truths = validate(model, val_dataloader, device)

        # 评估mAP (这里使用骨架函数)
        metrics = calculate_map(val_results, val_ground_truths, num_classes=NUM_CLASSES)
        print(f"  Validation mAP@0.5: {metrics['mAP@0.5']:.4f}")
        print(f"  Validation mAP@0.5:0.95: {metrics['mAP@0.5:0.95']:.4f}")

        # 保存最佳模型 (这里简化为根据训练损失保存，实际应根据mAP)
        if avg_train_loss < best_val_loss:  # 实际应根据验证mAP
            best_val_loss = avg_train_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"  Saved best model to {BEST_MODEL_PATH} (based on training loss).")

        end_time = time.time()
        print(f"  Epoch took {(end_time - start_time):.2f} seconds.")

    print("\n--- Training Complete ---")
    print(f"Best model saved to {BEST_MODEL_PATH}")


# --- 验证函数 ---
def validate(model, dataloader, device):
    """
    在验证集上评估模型性能。
    """
    model.eval()  # 设置模型为评估模式
    all_decoded_preds = []  # 存储所有图像的解码预测结果
    all_ground_truths = []  # 存储所有图像的真实标注

    with torch.no_grad():  # 评估时不需要计算梯度
        progress_bar = tqdm(dataloader, desc="Validating")
        for images, targets in progress_bar:
            images = images.to(device)

            predictions = model(images)

            # 解码预测 (不应用NMS，NMS在mAP计算内部或之后应用)
            # decode_predictions 返回的是一个列表，每个元素是 (num_detections, 6) 的张量
            decoded_preds_batch = decode_predictions(predictions, img_size=IMG_SIZE, conf_threshold=CONF_THRESHOLD)

            # 将每个图像的预测和真实标注添加到列表中
            # targets 是 (total_num_objects_in_batch, 6)
            # 需要将其按批次索引拆分回列表
            for b_idx in range(images.shape[0]):
                all_decoded_preds.append(decoded_preds_batch[b_idx])
                # 提取当前图像的真实标注
                current_gt = targets[targets[:, 0] == b_idx][:, 1:]  # 移除batch_idx列
                all_ground_truths.append(current_gt)

    return all_decoded_preds, all_ground_truths


# --- 推理函数 (示例) ---
def inference(model_path, image_path, img_size, conf_threshold, iou_threshold, device):
    """
    使用训练好的模型进行单张图像的推理。
    """
    print(f"\n--- Running Inference on {image_path} ---")

    # 1. 加载模型
    model = YOLOv8mModel(nc=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Model loaded from {model_path}")

    # 2. 加载和预处理图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image {image_path} for inference.")
        return None

    original_img = img.copy()  # 保存原始图像用于可视化
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_resized, _, _ = letterbox(img, img_size)
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)  # 添加批次维度

    # 3. 模型前向传播
    with torch.no_grad():
        predictions = model(img_tensor)

    # 4. 解码预测并应用NMS
    # decode_predictions 返回一个列表，因为是单张图像，所以取第一个元素
    decoded_preds_batch = decode_predictions(predictions, img_size=img_size, conf_threshold=conf_threshold)
    final_detections = non_max_suppression(decoded_preds_batch[0], iou_threshold=iou_threshold)

    print(f"Found {final_detections.shape[0]} detections.")

    # 5. 可视化结果 (简化)
    if final_detections.shape[0] > 0:
        # 将归一化坐标转换回原始图像像素坐标
        # 注意：这里的转换需要考虑 letterbox 的缩放和填充
        # 为了简化，我们直接在 letterbox 后的图像上绘制
        display_img = img_resized.copy()
        display_img = cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR)  # 转换回BGR用于cv2绘制

        for det in final_detections:
            x1, y1, x2, y2, conf, cls_id = det.cpu().numpy()
            x1, y1, x2, y2 = int(x1 * img_size[1]), int(y1 * img_size[0]), int(x2 * img_size[1]), int(y2 * img_size[0])

            color = (0, 255, 0)  # 绿色框
            cv2.rectangle(display_img, (x1, y1), (x2, y2), color, 2)
            label = f"Class {int(cls_id)}: {conf:.2f}"
            cv2.putText(display_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        output_path = "inference_result.jpg"
        cv2.imwrite(output_path, display_img)
        print(f"Inference result saved to {output_path}")
        # cv2.imshow("Inference Result", display_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    return final_detections


# --- 主函数 ---
def main():
    # 检查并创建虚拟数据
    # 现在直接使用 config 中定义的 TRAIN/VAL 路径
    if not os.path.exists(TRAIN_IMG_DIR) or not os.listdir(TRAIN_IMG_DIR):
        create_dummy_data()

    # --- 设备设置 ---
    device = torch.device(DEVICE)
    print(f"Using device: {device}")

    # --- 数据集和数据加载器 ---
    train_dataset = YOLOv8Dataset(img_dir=TRAIN_IMG_DIR, label_dir=TRAIN_LABEL_DIR, img_size=IMG_SIZE)
    val_dataset = YOLOv8Dataset(img_dir=VAL_IMG_DIR, label_dir=VAL_LABEL_DIR, img_size=IMG_SIZE)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=NUM_WORKERS, collate_fn=custom_collate_fn, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=NUM_WORKERS, collate_fn=custom_collate_fn, pin_memory=True)

    print(f"Train Dataloader batches: {len(train_dataloader)}")
    print(f"Val Dataloader batches: {len(val_dataloader)}")

    # --- 模型实例化 ---
    model = YOLOv8mModel(nc=NUM_CLASSES).to(device)
    # print("\n--- Model Architecture ---")
    # print(model) # 打印模型结构可能非常长

    # --- 损失函数、优化器和学习率调度器 ---
    loss_fn = YOLOv8Loss(nc=NUM_CLASSES, img_size=IMG_SIZE,
                         box_loss_weight=BOX_LOSS_WEIGHT,
                         cls_loss_weight=CLS_LOSS_WEIGHT,
                         obj_loss_weight=OBJ_LOSS_WEIGHT).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # --- 训练 ---
    train(model, train_dataloader, val_dataloader, optimizer, loss_fn, scheduler, EPOCHS, device)

    # --- 推理示例 ---
    # 假设我们想对验证集中的第一张图片进行推理
    # 确保 val_img_0.jpg 存在于 VAL_IMG_DIR 中
    if os.path.exists(BEST_MODEL_PATH) and os.path.exists(os.path.join(VAL_IMG_DIR, "val_img_0.jpg")):
        inference_image_path = os.path.join(VAL_IMG_DIR, "val_img_0.jpg")
        inference(BEST_MODEL_PATH, inference_image_path, IMG_SIZE, CONF_THRESHOLD, IOU_THRESHOLD, device)
    else:
        print("Skipping inference: best model or example image not found.")

    # 清理虚拟数据 (可选，如果您想保留数据，请注释掉)
    # 注意：如果您的 TRAIN_IMG_DIR 等指向了真实数据，请勿取消注释此行，否则会删除您的数据！
    # print("\nCleaning up dummy data...")
    # if os.path.exists(DATA_ROOT): # 清理 DATA_ROOT 下的虚拟数据
    #     shutil.rmtree(DATA_ROOT)
    # print("Dummy data cleaned.")


if __name__ == "__main__":
    main()

