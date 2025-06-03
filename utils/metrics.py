# utils/metrics.py

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import os
import json
import tempfile
import numpy as np

# 从 config.py 导入 IMG_SIZE
from config import IMG_SIZE


def calculate_map(predictions, ground_truths, num_classes=80):
    """
    使用 pycocotools 计算 COCO mAP。
    Args:
        predictions (list): 每张图像预测，列表中的每个元素是 shape=(N,6) 的tensor。
                            每行格式：[x1, y1, x2, y2, conf, cls_id]（归一化坐标）。
        ground_truths (list): 每张图像GT，列表中的每个元素是 shape=(M,5) 的tensor。
                               每行格式：[cls_id, x_c, y_c, w, h]（归一化坐标）。
        num_classes (int): 数据集中的类别总数。
    Returns:
        dict: 包含 mAP、precision、recall 等指标。
    """
    # COCO格式的真实标注字典
    coco_gt_dict = {"images": [], "annotations": [], "categories": []}
    # COCO格式的检测结果列表
    coco_dt = []

    # 从 config 中获取图像的宽度和高度
    # 注意：config.IMG_SIZE 是 (width, height)
    img_width, img_height = IMG_SIZE[0], IMG_SIZE[1]

    # 填充类别信息
    for i in range(num_classes):
        coco_gt_dict["categories"].append({
            "id": i,
            "name": str(i)  # 类别名称，这里简化为字符串形式的ID
        })

    ann_id = 1  # 标注ID，递增
    for i, (gts_per_image, preds_per_image) in enumerate(zip(ground_truths, predictions)):
        # 添加图像信息
        # 假设图像ID就是其在列表中的索引 i
        # 使用从 config 中获取的图像尺寸
        coco_gt_dict["images"].append({
            "id": i,
            "file_name": f"{i}.jpg",  # 虚拟文件名
            "width": img_width,  # 从 config 获取
            "height": img_height  # 从 config 获取
        })

        # 处理真实标注 (Ground Truths)
        for gt in gts_per_image:
            # 真实标注格式: [cls_id, x_c, y_c, w, h] (归一化坐标)
            cls_id, x_c, y_c, w, h = gt.tolist()

            # 将归一化 (x_c, y_c, w, h) 转换为 COCO 的 [x, y, width, height] 像素坐标
            # COCO bbox: [x_top_left, y_top_left, width, height]
            x1 = (x_c - w / 2) * img_width
            y1 = (y_c - h / 2) * img_height
            bbox_w = w * img_width
            bbox_h = h * img_height

            coco_gt_dict["annotations"].append({
                "id": ann_id,
                "image_id": i,
                "category_id": int(cls_id),
                "bbox": [x1, y1, bbox_w, bbox_h],
                "area": bbox_w * bbox_h,  # 面积
                "iscrowd": 0  # 非人群标注
            })
            ann_id += 1

        # 处理预测结果 (Detections)
        for pred in preds_per_image:
            # 预测格式: [x1, y1, x2, y2, conf, cls_id] (归一化坐标)
            x1_norm, y1_norm, x2_norm, y2_norm, score, cls_id = pred.tolist()

            # 将归一化 (x1, y1, x2, y2) 转换为 COCO 的 [x, y, width, height] 像素坐标
            x1_px = x1_norm * img_width
            y1_px = y1_norm * img_height
            bbox_w_px = (x2_norm - x1_norm) * img_width
            bbox_h_px = (y2_norm - y1_norm) * img_height

            coco_dt.append({
                "image_id": i,
                "category_id": int(cls_id),
                "bbox": [x1_px, y1_px, bbox_w_px, bbox_h_px],
                "score": float(score)
            })

    # 使用临时文件保存JSON，然后用pycocotools加载
    with tempfile.TemporaryDirectory() as tmp_dir:
        gt_path = os.path.join(tmp_dir, "gt.json")
        dt_path = os.path.join(tmp_dir, "dt.json")

        with open(gt_path, "w") as f:
            json.dump(coco_gt_dict, f)
        with open(dt_path, "w") as f:
            json.dump(coco_dt, f)

        # 初始化COCO API
        coco_gt = COCO(gt_path)
        coco_dt = coco_gt.loadRes(dt_path)

        # 创建COCOeval对象并评估
        coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        # 返回mAP统计数据
        return {
            "mAP@0.5:0.95": coco_eval.stats[0],  # mAP (IoU=0.50:0.05:0.95)
            "mAP@0.5": coco_eval.stats[1],  # mAP (IoU=0.50)
            "mAP@0.75": coco_eval.stats[2],  # mAP (IoU=0.75)
            "mAP_small": coco_eval.stats[3],  # mAP for small objects
            "mAP_medium": coco_eval.stats[4],  # mAP for medium objects
            "mAP_large": coco_eval.stats[5],  # mAP for large objects
            "AR@1": coco_eval.stats[6],  # Average Recall (max Dets=1)
            "AR@10": coco_eval.stats[7],  # Average Recall (max Dets=10)
            "AR@100": coco_eval.stats[8]  # Average Recall (max Dets=100)
        }

