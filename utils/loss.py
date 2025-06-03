# utils/loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# IoU 计算辅助函数
def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    """
    计算两个边界框集合的IoU。
    Args:
        box1 (torch.Tensor): 边界框集合1。
        box2 (torch.Tensor): 边界框集合2。
        x1y1x2y2 (bool): 如果为True，框的格式是 (x1, y1, x2, y2)；否则是 (x_c, y_c, w, h)。
        GIoU, DIoU, CIoU (bool): 是否计算GIoU, DIoU, CIoU。
        eps (float): 避免除以零的小常数。
    Returns:
        torch.Tensor: IoU值或GIoU/DIoU/CIoU损失。
    """
    # 将 (x_c, y_c, w, h) 转换为 (x1, y1, x2, y2)
    if not x1y1x2y2:
        b1_x1, b1_x2 = box1[..., 0] - box1[..., 2] / 2, box1[..., 0] + box1[..., 2] / 2
        b1_y1, b1_y2 = box1[..., 1] - box1[..., 3] / 2, box1[..., 1] + box1[..., 3] / 2
        b2_x1, b2_x2 = box2[..., 0] - box2[..., 2] / 2, box2[..., 0] + box2[..., 2] / 2
        b2_y1, b2_y2 = box2[..., 1] - box2[..., 3] / 2, box2[..., 1] + box2[..., 3] / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[..., 0], box1[..., 1], box1[..., 2], box1[..., 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[..., 0], box2[..., 1], box2[..., 2], box2[..., 3]

    # 交集面积
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # 并集面积
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    if GIoU or DIoU or CIoU:
        # 最小外接矩形
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex ( enclosing ) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if GIoU:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area
        if DIoU or CIoU:  # DIoU https://arxiv.org/pdf/1911.08287.pdf
            # 中心点距离
            c2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance
            rho2 = c2 / (cw ** 2 + ch ** 2 + eps)  # distance normalized by diagonal of minimum bounding box
            if DIoU:
                return iou - rho2
            if CIoU:  # CIoU https://arxiv.org/pdf/2004.10934.pdf
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + rho2))
                return iou - (rho2 + v * alpha)
    return iou


# 边界框回归损失 (CIoU Loss)
class CIoULoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred_boxes, target_boxes):
        """
        计算CIoU损失。
        Args:
            pred_boxes (torch.Tensor): 预测边界框，形状为 (N, 4)，格式 (x_c, y_c, w, h)。
            target_boxes (torch.Tensor): 真实边界框，形状为 (N, 4)，格式 (x_c, y_c, w, h)。
        Returns:
            torch.Tensor: CIoU损失。
        """
        iou = bbox_iou(pred_boxes, target_boxes, x1y1x2y2=False, CIoU=True)
        loss = 1.0 - iou
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


# YOLOv8总损失函数
class YOLOv8Loss(nn.Module):
    def __init__(self, nc=80, img_size=(640, 640), strides=(8, 16, 32),
                 box_loss_weight=7.5, cls_loss_weight=0.5, obj_loss_weight=1.5):
        super().__init__()
        self.nc = nc  # 类别数
        self.img_size = img_size  # 输入图像尺寸
        self.strides = strides  # 每个检测头的下采样步长

        # 损失权重
        self.box_loss_weight = box_loss_weight
        self.cls_loss_weight = cls_loss_weight
        self.obj_loss_weight = obj_loss_weight  # 在YOLOv8中通常与分类损失合并或以不同方式处理

        # 损失函数实例
        self.bce_cls = nn.BCEWithLogitsLoss(reduction='none')  # 分类损失 (二元交叉熵)
        self.ciou_loss = CIoULoss(reduction='none')  # CIoU损失

    def forward(self, predictions, targets):
        """
        计算YOLOv8的总损失。
        Args:
            predictions (list): 模型在每个尺度上的原始输出。
                                每个元素形状为 (batch_size, (4 + nc), H, W)。
            targets (torch.Tensor): 批次标注，形状为 (total_num_objects_in_batch, 6)，
                                    每行 [batch_idx, class_id, x_c, y_c, w, h] (归一化)。
        Returns:
            torch.Tensor: 总损失。
            dict: 包含各项损失值的字典。
        """
        total_box_loss = 0.0
        total_cls_loss = 0.0
        total_obj_loss = 0.0  # 简化处理，作为目标置信度损失

        # 将所有预测展平并拼接
        # (batch_size, num_preds_per_scale, 4+nc)
        # 这里的 num_preds_per_scale 是 H*W
        all_preds_flat = []
        for pred_per_scale in predictions:
            bs, _, H, W = pred_per_scale.shape
            all_preds_flat.append(pred_per_scale.permute(0, 2, 3, 1).reshape(bs, H * W, -1))

        # all_preds_flat 现在是 (batch_size, total_num_cells, 4+nc)
        all_preds_flat = torch.cat(all_preds_flat, dim=1)

        # 分离边界框预测和分类预测
        pred_boxes_raw = all_preds_flat[..., :4]  # 原始边界框预测 (x_c, y_c, w, h)
        pred_cls_logits = all_preds_flat[..., 4:]  # 原始分类 logits

        # === 简化匹配和损失计算 (仅用于演示，不代表YOLOv8的完整训练逻辑) ===
        # 警告：此处的匹配逻辑是高度简化的，不包含TaskAlignedAssigner和DFL的复杂性。
        # 实际YOLOv8的训练需要更复杂的匹配策略来处理正负样本分配。

        if targets.shape[0] > 0:
            # 遍历批次中的每个图像
            for b in range(bs):
                # 获取当前图像的真实框和预测
                current_gt_boxes = targets[targets[:, 0] == b][:, 2:]  # (num_gts, 4)
                current_gt_cls = targets[targets[:, 0] == b][:, 1].long()  # (num_gts,)

                # 获取当前图像的所有预测
                current_pred_boxes_raw = pred_boxes_raw[b]  # (num_cells_total, 4)
                current_pred_cls_logits = pred_cls_logits[b]  # (num_cells_total, nc)

                if current_gt_boxes.shape[0] == 0:  # 如果当前图像没有真实框
                    # 仅计算分类和目标置信度损失（作为负样本）
                    total_obj_loss += self.bce_cls(current_pred_cls_logits.sum(dim=-1),
                                                   torch.zeros_like(current_pred_cls_logits.sum(dim=-1))).sum()
                    total_cls_loss += self.bce_cls(current_pred_cls_logits,
                                                   torch.zeros_like(current_pred_cls_logits)).sum()
                    continue

                # 计算当前图像所有预测框与所有真实框之间的 IoU
                # (num_preds, 4) vs (num_gts, 4) -> (num_preds, num_gts)
                ious = bbox_iou(current_pred_boxes_raw.unsqueeze(1), current_gt_boxes.unsqueeze(0), x1y1x2y2=False,
                                CIoU=True)

                # 找到每个真实框对应的最佳预测框 (IoU最高)
                # (num_gts,)
                best_pred_for_gt_iou, best_pred_for_gt_idx = ious.max(dim=0)

                # 找到每个预测框对应的最佳真实框 (IoU最高)
                # (num_preds,)
                best_gt_for_pred_iou, best_gt_for_pred_idx = ious.max(dim=1)

                # 确定正样本：
                # 1. 每个真实框的最佳预测
                # 2. IoU > 0.5 的预测 (简化阈值)

                # 真实框对应的最佳预测作为正样本
                matched_pred_indices = best_pred_for_gt_idx
                matched_gt_indices = torch.arange(current_gt_boxes.shape[0], device=current_gt_boxes.device)

                # 匹配的预测框和真实框
                matched_pred_boxes = current_pred_boxes_raw[matched_pred_indices]
                matched_gt_boxes = current_gt_boxes[matched_gt_indices]
                matched_pred_cls_logits = current_pred_cls_logits[matched_pred_indices]
                matched_gt_cls = current_gt_cls[matched_gt_indices]

                # 边界框损失 (CIoU)
                box_loss = self.ciou_loss(matched_pred_boxes, matched_gt_boxes).sum()  # sum over matched boxes
                total_box_loss += box_loss

                # 分类损失
                # 为匹配的真实类别创建one-hot编码
                target_one_hot = F.one_hot(matched_gt_cls, num_classes=self.nc).float()
                cls_loss = self.bce_cls(matched_pred_cls_logits, target_one_hot).sum()  # sum over matched boxes
                total_cls_loss += cls_loss

                # 目标置信度损失 (简化处理)
                # 假设匹配到的预测是正样本 (目标置信度为1)
                # 其他所有预测是负样本 (目标置信度为0)
                obj_target = torch.zeros_like(current_pred_cls_logits.sum(dim=-1))
                obj_target[matched_pred_indices] = 1.0  # 匹配到的预测置信度为1

                obj_loss = self.bce_cls(current_pred_cls_logits.sum(dim=-1),
                                        obj_target).sum()  # sum over all predictions
                total_obj_loss += obj_loss
        else:  # 如果批次中没有真实框
            # 所有预测都应该是负样本
            total_obj_loss = self.bce_cls(pred_cls_logits.sum(dim=-1),
                                          torch.zeros_like(pred_cls_logits.sum(dim=-1))).sum()
            total_cls_loss = self.bce_cls(pred_cls_logits, torch.zeros_like(pred_cls_logits)).sum()

        # 总损失
        total_loss = self.box_loss_weight * total_box_loss + \
                     self.cls_loss_weight * total_cls_loss + \
                     self.obj_loss_weight * total_obj_loss

        return total_loss, {
            'box_loss': total_box_loss.item(),
            'cls_loss': total_cls_loss.item(),
            'obj_loss': total_obj_loss.item(),
            'total_loss': total_loss.item()
        }

