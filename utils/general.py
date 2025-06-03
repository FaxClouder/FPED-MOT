# utils/general.py

import torch
import math
from utils.loss import bbox_iou  # 导入 IoU 计算函数


def decode_predictions(predictions, img_size=(640, 640), conf_threshold=0.25):
    """
    解码YOLOv8模型的原始输出。
    Args:
        predictions (list): 模型在每个尺度上的原始输出。
                            每个元素形状为 (batch_size, (4 + nc), H, W)。
        img_size (tuple): 输入图像的尺寸 (width, height)。
        conf_threshold (float): 目标置信度阈值。
    Returns:
        list: 每个图像的解码结果列表。
              每个结果是一个张量，形状为 (num_detections, 6)，
              每行 [x1, y1, x2, y2, confidence, class_id]。
              坐标是归一化后的。
    """
    final_detections_batch = [[] for _ in range(predictions[0].shape[0])]

    for b_idx in range(predictions[0].shape[0]):  # 遍历批次中的每张图像
        current_image_detections = []
        for i, pred_per_scale in enumerate(predictions):  # 遍历每个尺度
            # 提取当前图像在当前尺度的预测
            # pred_per_scale_img: (4+nc, H, W)
            pred_per_scale_img = pred_per_scale[b_idx]

            # 展平为 (H*W, 4+nc)
            pred_per_scale_flat = pred_per_scale_img.permute(1, 2, 0).reshape(-1, pred_per_scale_img.shape[0])

            # 边界框坐标 (x_c, y_c, w, h) - 假设已归一化
            box_coords = pred_per_scale_flat[..., :4]
            # 类别置信度 (logits -> probabilities)
            cls_probs = torch.sigmoid(pred_per_scale_flat[..., 4:])

            # 获取每个框的最高分类概率和对应的类别ID
            max_cls_prob, class_id = cls_probs.max(dim=-1)
            # 最终的置信度分数 (这里简化为最高类别概率)
            confidences = max_cls_prob

            # 过滤低置信度的预测
            keep_mask = confidences > conf_threshold

            filtered_boxes = box_coords[keep_mask]
            filtered_confidences = confidences[keep_mask]
            filtered_class_ids = class_id[keep_mask]

            if filtered_boxes.shape[0] == 0:
                continue

            # 将 (x_c, y_c, w, h) 转换为 (x1, y1, x2, y2)
            # 归一化坐标
            x1 = filtered_boxes[:, 0] - filtered_boxes[:, 2] / 2
            y1 = filtered_boxes[:, 1] - filtered_boxes[:, 3] / 2
            x2 = filtered_boxes[:, 0] + filtered_boxes[:, 2] / 2
            y2 = filtered_boxes[:, 1] + filtered_boxes[:, 3] / 2

            detections_per_scale_img = torch.cat([x1.unsqueeze(1), y1.unsqueeze(1), x2.unsqueeze(1), y2.unsqueeze(1),
                                                  filtered_confidences.unsqueeze(1),
                                                  filtered_class_ids.unsqueeze(1).float()], dim=1)
            current_image_detections.append(detections_per_scale_img)

        if len(current_image_detections) > 0:
            final_detections_batch[b_idx] = torch.cat(current_image_detections, dim=0)
        else:
            final_detections_batch[b_idx] = torch.empty(0, 6).to(predictions[0].device)  # 确保设备一致

    return final_detections_batch


def non_max_suppression(detections, iou_threshold=0.45):
    """
    对检测结果应用非极大值抑制 (NMS)。
    Args:
        detections (torch.Tensor): 形状为 (num_detections, 6)，
                                  每行 [x1, y1, x2, y2, confidence, class_id]。
                                  坐标是归一化后的。
        iou_threshold (float): NMS的IoU阈值。
    Returns:
        torch.Tensor: 经过NMS后的检测结果，形状为 (final_num_detections, 6)。
    """
    if detections.shape[0] == 0:
        return torch.empty(0, 6).to(detections.device)

    # 按照置信度降序排序
    scores = detections[:, 4]
    # 确保 order 是一个包含原始索引的张量
    _, order = scores.sort(0, descending=True)

    # 获取排序后的检测框
    sorted_detections = detections[order]

    keep = []
    # 遍历每个类别进行NMS (类别感知NMS)
    unique_classes = sorted_detections[:, 5].unique()
    for cls in unique_classes:
        cls_mask = (sorted_detections[:, 5] == cls)
        cls_detections = sorted_detections[cls_mask]
        cls_order_indices = order[cls_mask]  # 对应原始索引

        if cls_detections.shape[0] == 0:
            continue

        current_keep_for_cls = []
        while cls_detections.shape[0] > 0:
            # 保留当前置信度最高的框
            idx = 0
            current_keep_for_cls.append(cls_order_indices[idx])

            # 获取当前框的IoU
            current_box = cls_detections[idx, :4]
            # 剩余的框
            other_boxes = cls_detections[1:, :4]

            if other_boxes.shape[0] == 0:
                break  # 没有其他框了

            # 计算当前框与所有其他框的IoU
            ious = bbox_iou(current_box.unsqueeze(0), other_boxes, x1y1x2y2=True).squeeze(0)

            # 找到IoU小于阈值的框（即不重叠的框）
            non_overlap_indices = torch.where(ious < iou_threshold)[0] + 1  # +1 因为移除了第一个元素

            cls_detections = cls_detections[non_overlap_indices]
            cls_order_indices = cls_order_indices[non_overlap_indices]

        keep.extend(current_keep_for_cls)

    # 最终的NMS结果，使用原始索引从原始 detections 中选择
    # 确保 keep 列表不为空，否则会尝试索引空张量
    if len(keep) > 0:
        return detections[torch.tensor(keep, device=detections.device)].contiguous()
    else:
        return torch.empty(0, 6).to(detections.device)

