import torch
import torch.nn as nn
import torch.nn.functional as F


def select_candidates_in_gts(xy_centers, gt_bboxes, eps=1e-9):
    """
    Select the positive anchor center in gt.
    Args:
        xy_centers (Tensor): shape(bs, n_anchors, 2)
        gt_bboxes (Tensor): shape(bs, n_boxes, 4)
    Return:
        (Tensor): shape(bs, n_anchors, n_boxes)
    """
    n_anchors = xy_centers.shape[1]
    bs, n_boxes, _ = gt_bboxes.shape
    lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)  # left-top, right-bottom
    bbox_deltas = torch.cat((xy_centers.view(bs, n_anchors, 1, 2) - lt, rb - xy_centers.view(bs, n_anchors, 1, 2)),
                            dim=3)
    return bbox_deltas.amin(3).gt_(eps)


def select_highest_overlaps(mask_pos, overlaps, n_max_boxes):
    """
    If an anchor box is assigned to multiple gts,
        select the one with the highest IoU.
    Args:
        mask_pos (Tensor): shape(bs, n_anchors, n_boxes)
        overlaps (Tensor): shape(bs, n_anchors, n_boxes)
    Return:
        target_gt_idx (Tensor): shape(bs, n_anchors)
        fg_mask (Tensor): shape(bs, n_anchors)
        mask_pos (Tensor): shape(bs, n_anchors, n_boxes)
    """
    # (bs, n_anchors, n_boxes) -> (bs, n_anchors)
    fg_mask = mask_pos.sum(-1)
    if fg_mask.max() > 1:  # one anchor is assigned to multiple gt_bboxes
        mask_multi_gts = (fg_mask.unsqueeze(2) > 1).repeat([1, 1, n_max_boxes])
        max_overlaps_idx = overlaps.argmax(-1)
        is_max_overlaps = F.one_hot(max_overlaps_idx, n_max_boxes)
        is_max_overlaps = is_max_overlaps.to(overlaps.dtype)
        mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos)
        fg_mask = mask_pos.sum(-1)
    # find each anchor_idx to which gt_box_idx it belongs
    target_gt_idx = mask_pos.argmax(-1)  # (bs, n_anchors)
    return target_gt_idx, fg_mask, mask_pos


# Simple IoU calculation (you already have a more comprehensive one in utils.loss.bbox_iou)
# For simplicity here, we'll assume pred_bboxes and gt_bboxes are in (x_center, y_center, w, h) format
# and need conversion to (x1, y1, x2, y2) for IoU.
# Re-use your existing bbox_iou function if possible.
def bbox_iou_for_assigner(box1_cxcywh, box2_cxcywh, x1y1x2y2=False, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    """
    Helper to use your existing bbox_iou. Assumes input is cxcywh if x1y1x2y2 is False.
    """
    if not x1y1x2y2:  # Convert cxcywh to x1y1x2y2
        b1_x1, b1_x2 = box1_cxcywh[..., 0] - box1_cxcywh[..., 2] / 2, box1_cxcywh[..., 0] + box1_cxcywh[..., 2] / 2
        b1_y1, b1_y2 = box1_cxcywh[..., 1] - box1_cxcywh[..., 3] / 2, box1_cxcywh[..., 1] + box1_cxcywh[..., 3] / 2
        box1 = torch.stack((b1_x1, b1_y1, b1_x2, b1_y2), dim=-1)

        b2_x1, b2_x2 = box2_cxcywh[..., 0] - box2_cxcywh[..., 2] / 2, box2_cxcywh[..., 0] + box2_cxcywh[..., 2] / 2
        b2_y1, b2_y2 = box2_cxcywh[..., 1] - box2_cxcywh[..., 3] / 2, box2_cxcywh[..., 1] + box2_cxcywh[..., 3] / 2
        box2 = torch.stack((b2_x1, b2_y1, b2_x2, b2_y2), dim=-1)
    else:
        box1, box2 = box1_cxcywh, box2_cxcywh

    # Intersection
    inter_x1 = torch.max(box1[..., 0], box2[..., 0])
    inter_y1 = torch.max(box1[..., 1], box2[..., 1])
    inter_x2 = torch.min(box1[..., 2], box2[..., 2])
    inter_y2 = torch.min(box1[..., 3], box2[..., 3])
    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter = inter_w * inter_h

    # Union
    w1, h1 = box1[..., 2] - box1[..., 0], box1[..., 3] - box1[..., 1]
    w2, h2 = box2[..., 2] - box2[..., 0], box2[..., 3] - box2[..., 1]
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    if GIoU or DIoU or CIoU:
        # This part is complex and depends on your bbox_iou implementation
        # For now, we will just return plain IoU for the assigner.
        # You should integrate your more complete bbox_iou from utils.loss.py here.
        # For simplicity, this assigner will use plain IoU.
        # If you need GIoU/DIoU/CIoU for the assigner's overlap metric, you'd compute it here.
        pass
    return iou


class TaskAlignedAssigner(nn.Module):
    def __init__(self, topk=13, num_classes=80, alpha=1.0, beta=6.0, eps=1e-9):
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.bg_idx = num_classes  # background index
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """
        Args:
            pd_scores (Tensor): shape(bs, n_anchors, num_classes)
            pd_bboxes (Tensor): shape(bs, n_anchors, 4) (format: cxcywh)
            anc_points (Tensor): shape(n_anchors, 2) (format: xy)
            gt_labels (Tensor): shape(bs, n_max_boxes, 1)
            gt_bboxes (Tensor): shape(bs, n_max_boxes, 4) (format: cxcywh)
            mask_gt (Tensor): shape(bs, n_max_boxes, 1)
        Returns:
            target_labels (Tensor): shape(bs, n_anchors, num_classes)
            target_bboxes (Tensor): shape(bs, n_anchors, 4)
            target_scores (Tensor): shape(bs, n_anchors, num_classes)
            fg_mask (Tensor): shape(bs, n_anchors)
        """
        self.bs = pd_scores.size(0)
        self.n_anchors = pd_scores.size(1)

        # mask_gt filters out padded ground truths, (bs, n_max_boxes, 1) -> (bs, n_max_boxes)
        mask_gt = mask_gt.squeeze(-1)

        # Initialize targets with background
        target_labels = torch.full((self.bs, self.n_anchors, self.num_classes), 0.0, device=pd_scores.device)
        target_bboxes = torch.zeros((self.bs, self.n_anchors, 4), device=pd_scores.device)
        target_scores = torch.full((self.bs, self.n_anchors, self.num_classes), 0.0, device=pd_scores.device)
        fg_mask = torch.zeros(self.bs, self.n_anchors, dtype=torch.bool, device=pd_scores.device)

        for b in range(self.bs):
            # Filter out padded GTs for this batch item
            b_gt_labels = gt_labels[b][mask_gt[b]]  # (n_gt_for_item, 1)
            b_gt_bboxes = gt_bboxes[b][mask_gt[b]]  # (n_gt_for_item, 4)
            b_pd_scores = pd_scores[b]  # (n_anchors, num_classes)
            b_pd_bboxes = pd_bboxes[b]  # (n_anchors, 4)

            if b_gt_labels.numel() == 0:  # No GTs for this image
                continue

            # Alignment Metric calculation
            # Expand pd_bboxes and gt_bboxes for IoU calculation:
            # pd_bboxes_expanded: (n_anchors, 1, 4)
            # gt_bboxes_expanded: (1, n_gt_for_item, 4)
            # overlaps: (n_anchors, n_gt_for_item)
            overlaps = bbox_iou_for_assigner(b_pd_bboxes.unsqueeze(1), b_gt_bboxes.unsqueeze(0), x1y1x2y2=False)

            # Get classification scores for GT classes: (n_anchors, n_gt_for_item)
            # b_gt_labels is (n_gt_for_item, 1), need to convert to long for gather
            # pd_scores_for_gt_classes will be (n_anchors, n_gt_for_item)
            # Each column j corresponds to gt_labels[j], and contains scores from all anchors for that gt class
            # Example: gt_labels might be [[0], [1], [0]]. pd_scores is (num_anchors, num_classes)
            # We want to pick scores corresponding to these GT labels.
            # Unsqueeze b_gt_labels to (1, n_gt_for_item, 1) and expand for gather
            # expanded_gt_labels = b_gt_labels.squeeze(-1).long().unsqueeze(0).expand(self.n_anchors, -1) # (n_anchors, n_gt_for_item)
            # This might be tricky with gather, let's do it a bit more manually or use a different approach
            # For each anchor, and for each GT, get the score of the anchor for THAT GT's class.
            # This is equivalent to: b_pd_scores[:, b_gt_labels.squeeze(-1).long()]
            # which if b_gt_labels.squeeze(-1).long() is [c1, c2, c3], it means
            # [b_pd_scores[:, c1], b_pd_scores[:, c2], b_pd_scores[:, c3]]
            # This creates a list of tensors, we need to stack them.
            # Or, use advanced indexing if versions support it well.
            # A common way is to use one_hot encoding for gt_labels and multiply.

            # Let's try a more direct indexing if possible:
            # gt_cls_ids = b_gt_labels.squeeze(-1).long() # (n_gt_for_item)
            # pd_scores_for_gt_classes = b_pd_scores[:, gt_cls_ids] # (n_anchors, n_gt_for_item)

            # A more robust way for classification scores part:
            cls_preds_for_gt = b_pd_scores.unsqueeze(1).repeat(1, b_gt_labels.shape[0],
                                                               1)  # (n_anchors, n_gt, num_classes)
            # Create a mask for gt classes
            gt_cls_mask = F.one_hot(b_gt_labels.squeeze(-1).long(), self.num_classes).float()  # (n_gt, num_classes)
            gt_cls_mask = gt_cls_mask.unsqueeze(0).repeat(self.n_anchors, 1, 1)  # (n_anchors, n_gt, num_classes)

            pd_scores_for_gt_classes = (cls_preds_for_gt * gt_cls_mask).sum(-1)  # (n_anchors, n_gt_for_item)

            # alignment_metrics = (pd_scores_for_gt_classes.sigmoid_() ** self.alpha) * (overlaps ** self.beta)
            # Use .pow() for clarity and to avoid in-place modification issues with sigmoid_()
            alignment_metrics = (pd_scores_for_gt_classes.sigmoid().pow(self.alpha)) * (overlaps.pow(self.beta))

            # Select top-k candidates for each GT
            # (n_anchors, n_gt_for_item) -> (n_gt_for_item, n_anchors) for topk selection
            topk_metrics, topk_indices = torch.topk(alignment_metrics.t(), self.topk, dim=1, largest=True)

            # Create a mask for selected candidates
            # This mask indicates which (anchor, gt) pairs are selected by topk
            candidate_mask = torch.zeros_like(alignment_metrics, dtype=torch.bool)  # (n_anchors, n_gt_for_item)
            # topk_indices are relative to anchors for each GT.
            # We need to mark these anchor indices for each GT.
            # gt_indices_for_topk = torch.arange(b_gt_labels.shape[0], device=pd_scores.device).unsqueeze(1).repeat(1, self.topk)
            # candidate_mask[topk_indices.view(-1), gt_indices_for_topk.view(-1)] = True # This might be wrong indexing

            # Corrected way to populate candidate_mask:
            # For each gt, topk_indices gives the anchor indices.
            for gt_idx in range(b_gt_labels.shape[0]):
                candidate_mask[topk_indices[gt_idx], gt_idx] = True

            # Filter by whether anchor center is in GT bbox (optional, but common in some assigners)
            # For simplicity, we might skip this or assume anc_points are already pre-filtered if they represent grid cell centers.
            # If anc_points are true anchor box centers from pd_bboxes, this check makes more sense.
            # Let's assume for now anc_points are just grid cell centers for simplicity of TAA.
            # TAA typically works directly on predictions from all locations.

            # Assign GTs to anchors based on candidates
            # An anchor can be assigned to at most one GT. If multiple GTs are assigned to one anchor,
            # pick the one with the highest alignment metric.

            # (n_anchors, n_gt_for_item)
            anchor_max_alignment_metric, anchor_assigned_gt_idx = alignment_metrics.max(dim=1)

            # Filter out assignments where the anchor was not a top-k candidate for that GT
            # is_candidate_for_assigned_gt = candidate_mask.gather(1, anchor_assigned_gt_idx.unsqueeze(1)).squeeze(1)

            # Let's re-think assignment:
            # An anchor is a positive if it's among the top-k for *any* GT.
            # And if multiple GTs select the same anchor, the GT with highest alignment_metric wins for that anchor.

            # (n_anchors, n_gt_for_item)
            is_assigned_to_gt = torch.zeros_like(alignment_metrics, dtype=torch.bool)

            # For each anchor, find the GT that has the highest alignment metric with it
            # This is already given by anchor_max_alignment_metric and anchor_assigned_gt_idx

            # Now, we need to ensure this anchor was selected as a top-k candidate *for that specific GT*
            # Check if (anchor_idx, anchor_assigned_gt_idx[anchor_idx]) is in candidate_mask

            # fg_mask_b indicates which anchors are foreground for this batch item
            fg_mask_b = torch.zeros(self.n_anchors, dtype=torch.bool, device=pd_scores.device)

            # Iterate over anchors to assign
            temp_target_labels_b = torch.full((self.n_anchors, self.num_classes), 0.0, device=pd_scores.device)
            temp_target_bboxes_b = torch.zeros((self.n_anchors, 4), device=pd_scores.device)
            temp_target_scores_b = torch.full((self.n_anchors, self.num_classes), 0.0,
                                              device=pd_scores.device)  # For IoU-aware cls score

            # For each GT, consider its top-k anchors
            for gt_idx in range(b_gt_labels.shape[0]):
                current_gt_topk_anchor_indices = topk_indices[gt_idx]  # (topk,)
                current_gt_label = b_gt_labels[gt_idx].long()  # scalar
                current_gt_bbox = b_gt_bboxes[gt_idx]  # (4,)

                # Assign this GT to these top-k anchors if this GT provides them with the max alignment score
                for anchor_idx in current_gt_topk_anchor_indices:
                    # If this anchor is already assigned to another GT with higher alignment, skip
                    if fg_mask_b[anchor_idx] and anchor_max_alignment_metric[anchor_idx] > alignment_metrics[
                        anchor_idx, gt_idx]:
                        continue

                    fg_mask_b[anchor_idx] = True
                    # Assign the GT label (one-hot) and bbox
                    temp_target_labels_b[anchor_idx, :] = 0.0  # Reset if previously assigned by a lower score GT
                    temp_target_labels_b[anchor_idx, current_gt_label] = 1.0
                    temp_target_bboxes_b[anchor_idx, :] = current_gt_bbox

                    # target_scores often uses IoU with the assigned GT as the target for all classes
                    # or only for the GT class. Let's use it for the GT class for IoU-aware objective.
                    temp_target_scores_b[anchor_idx, :] = 0.0
                    temp_target_scores_b[anchor_idx, current_gt_label] = overlaps[anchor_idx, gt_idx]

            target_labels[b] = temp_target_labels_b
            target_bboxes[b] = temp_target_bboxes_b
            target_scores[b] = temp_target_scores_b  # This is the IoU-aware part
            fg_mask[b] = fg_mask_b

        return target_labels, target_bboxes, target_scores, fg_mask