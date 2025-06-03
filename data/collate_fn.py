# data/collate_fn.py

import torch


def custom_collate_fn(batch):
    """
    自定义 collate_fn，用于将批次中的图像和变长标注进行堆叠。
    Args:
        batch (list): DataLoader 返回的批次数据，每个元素是 (image_tensor, target_tensor)。
                      target_tensor 形状为 (num_objects, 5)，每行 [class_id, x_c, y_c, w, h]。
    Returns:
        tuple: (images, targets)
            images (torch.Tensor): 堆叠后的图像张量，形状为 (batch_size, C, H, W)。
            targets (torch.Tensor): 拼接后的标注张量，形状为 (total_num_objects_in_batch, 6)，
                                    每行 [batch_idx, class_id, x_c, y_c, w, h]。
    """
    images = []
    targets = []
    for i, (img, target) in enumerate(batch):
        images.append(img)
        if target.shape[0] > 0:  # 检查是否有标注
            # 为每个标注添加批次索引
            batch_idx = torch.full((target.shape[0], 1), i, dtype=torch.float32, device=target.device)  # 确保device一致
            targets.append(torch.cat((batch_idx, target), dim=1))

    images = torch.stack(images, 0)  # 将图像堆叠成一个批次张量

    if len(targets) > 0:
        targets = torch.cat(targets, 0)  # 将所有标注拼接成一个大张量
    else:
        targets = torch.empty(0, 6)  # 如果批次中都没有标注，返回 (0, 6)

    return images, targets

