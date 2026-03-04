# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0 
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""NMS (PyTorch Version)"""
import numpy as np
import torch


def apply_nms(all_boxes, all_scores, thres, max_boxes):
    """Apply NMS to bboxes using NumPy (CPU-based)."""
    # Convert torch tensor to numpy if needed
    if isinstance(all_boxes, torch.Tensor):
        all_boxes = all_boxes.cpu().numpy()
    if isinstance(all_scores, torch.Tensor):
        all_scores = all_scores.cpu().numpy()
        
    y1 = all_boxes[:, 0]
    x1 = all_boxes[:, 1]
    y2 = all_boxes[:, 2]
    x2 = all_boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    order = all_scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        if len(keep) >= max_boxes:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thres)[0]

        order = order[inds + 1]
    return np.array(keep)


def nms(bboxes,
        scores,
        pre_max_size=None,
        post_max_size=None,
        iou_threshold=0.5):
    """NMS (Non-Maximum Suppression)
    
    Args:
        bboxes: torch.Tensor or np.ndarray, shape [N, 4]
        scores: torch.Tensor or np.ndarray, shape [N]
        pre_max_size: int, maximum number of boxes to consider before NMS
        post_max_size: int, maximum number of boxes to keep after NMS
        iou_threshold: float, IoU threshold for suppression
        
    Returns:
        torch.Tensor or None: indices of kept boxes
    """
    # Determine device from input if it's a torch tensor
    device = None
    if isinstance(scores, torch.Tensor):
        device = scores.device
        
    if pre_max_size is not None:
        num_keeped_scores = scores.shape[0]
        pre_max_size = min(num_keeped_scores, pre_max_size)
        
        if isinstance(scores, torch.Tensor):
            # PyTorch topk returns (values, indices)
            scores, indices = torch.topk(scores, pre_max_size)
        else:
            # NumPy version
            indices = np.argsort(scores)[::-1][:pre_max_size]
            scores = scores[indices]
            
        if isinstance(bboxes, torch.Tensor):
            bboxes = bboxes[indices]
        else:
            bboxes = bboxes[indices]

    keep = apply_nms(bboxes, scores, iou_threshold, post_max_size)
    
    if keep.shape[0] == 0:
        return None
        
    if pre_max_size is not None:
        # Map back to original indices
        keep = torch.tensor(keep, device=device, dtype=torch.long)
        return indices[keep]
    
    return torch.tensor(keep, device=device, dtype=torch.long)