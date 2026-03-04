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
"""losses (PyTorch Version)"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def indices_to_dense_vector(indices,
                            size,
                            indices_value=1.,
                            default_value=0,
                            dtype=torch.float32,
                            device=None):
    """Creates dense vector with indices set to specific value and rest to zeros.

    Args:
      indices: 1d Tensor with integer indices which are to be set to
        indices_values.
      size: scalar with size (integer) of output Tensor.
      indices_value: values of elements specified by indices in the output vector
      default_value: values of other elements in the output vector.
      dtype: data type of the output tensor.
      device: device of the output tensor.

    Returns:
      dense 1D Tensor of shape [size] with indices set to indices_values and the
      rest set to default_value.
    """
    dense = torch.full((size,), default_value, dtype=dtype, device=device)
    if indices.numel() > 0:
        dense.scatter_(0, indices.long(), indices_value)
    return dense


def _sigmoid_cross_entropy_with_logits(logits, labels):
    """sigmoid cross entropy with logits"""
    # Numerically stable implementation: max(x, 0) - x*z + log(1 + exp(-abs(x)))
    loss = torch.clamp(logits, min=0) - logits * labels.to(logits.dtype)
    loss += torch.log1p(torch.exp(-torch.abs(logits)))
    return loss


def _softmax_cross_entropy_with_logits(logits, labels):
    """softmax cross entropy with logits"""
    # labels are one-hot encoded, convert to class indices for F.cross_entropy
    target_indices = torch.argmax(labels, dim=-1)
    loss = F.cross_entropy(logits, target_indices, reduction='none')
    return loss


class SigmoidFocalClassificationLoss(nn.Module):
    """Sigmoid focal cross entropy loss.

    Focal loss down-weights well classified examples and focuses on the hard
    examples. See https://arxiv.org/pdf/1708.02002.pdf for the loss definition.
    """

    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self._alpha = alpha
        self._gamma = gamma

    def forward(self,
                prediction_tensor,
                target_tensor,
                weights,
                class_indices=None):
        """Compute loss function."""
        weights = weights.unsqueeze(2)
        if class_indices is not None:
            class_weights = indices_to_dense_vector(
                class_indices,
                prediction_tensor.shape[2],
                indices_value=1.,
                default_value=0,
                dtype=prediction_tensor.dtype,
                device=prediction_tensor.device
            ).view(1, 1, -1)
            weights = weights * class_weights
        
        per_entry_cross_ent = _sigmoid_cross_entropy_with_logits(
            logits=prediction_tensor, 
            labels=target_tensor
        )
        
        prediction_probabilities = torch.sigmoid(prediction_tensor)
        p_t = ((target_tensor * prediction_probabilities) +
               ((1 - target_tensor) * (1 - prediction_probabilities)))
        
        modulating_factor = 1.0
        if self._gamma:
            modulating_factor = torch.pow(1.0 - p_t, self._gamma)
        
        alpha_weight_factor = 1.0
        if self._alpha is not None:
            alpha_weight_factor = (target_tensor * self._alpha +
                                   (1 - target_tensor) * (1 - self._alpha))

        focal_cross_entropy_loss = modulating_factor * alpha_weight_factor * per_entry_cross_ent
        return focal_cross_entropy_loss * weights


class WeightedSmoothL1LocalizationLoss(nn.Module):
    """Smooth L1 localization loss function.

    The smooth L1_loss is defined elementwise as .5 x^2 if |x|<1 and |x|-.5
    otherwise, where x is the difference between predictions and target.

    See also Equation (3) in the Fast R-CNN paper by Ross Girshick (ICCV 2015)
    """

    def __init__(self, sigma=3.0, code_weights=None, codewise=True):
        super().__init__()
        self._sigma = sigma
        if code_weights is not None:
            self._code_weights = torch.tensor(code_weights, dtype=torch.float32)
        else:
            self._code_weights = None
        self._codewise = codewise

    def forward(self, prediction_tensor, target_tensor, weights=None):
        """Compute loss function.

        Args:
            prediction_tensor: A float tensor of shape [batch_size, num_anchors,
              code_size] representing the (encoded) predicted locations of objects.
            target_tensor: A float tensor of shape [batch_size, num_anchors,
              code_size] representing the regression targets
            weights: a float tensor of shape [batch_size, num_anchors]

        Returns:
            loss: a float tensor of shape [batch_size, num_anchors, code_size] or
                [batch_size, num_anchors] depending on codewise parameter.
        """
        diff = prediction_tensor - target_tensor
        
        if self._code_weights is not None:
            code_weights = self._code_weights.to(prediction_tensor.dtype).to(prediction_tensor.device)
            diff = code_weights.view(1, 1, -1) * diff
            
        abs_diff = torch.abs(diff)
        sigma2 = self._sigma ** 2
        abs_diff_lt_1 = (abs_diff <= 1 / sigma2).to(abs_diff.dtype)
        
        loss = (abs_diff_lt_1 * 0.5 * torch.pow(abs_diff * self._sigma, 2) +
                (abs_diff - 1 / (2 * sigma2)) * (1. - abs_diff_lt_1))
        
        if self._codewise:
            anchorwise_smooth_l1norm = loss
            if weights is not None:
                anchorwise_smooth_l1norm = anchorwise_smooth_l1norm * weights.unsqueeze(-1)
        else:
            anchorwise_smooth_l1norm = loss.sum(dim=2)
            if weights is not None:
                anchorwise_smooth_l1norm = anchorwise_smooth_l1norm * weights
                
        return anchorwise_smooth_l1norm


class WeightedSoftmaxClassificationLoss(nn.Module):
    """Softmax loss function."""

    def __init__(self, logit_scale=1.0):
        """Constructor.

        Args:
          logit_scale: When this value is high, the prediction is "diffused" and
                       when this value is low, the prediction is made peakier.
                       (default 1.0)
        """
        super().__init__()
        self._logit_scale = logit_scale

    def forward(self, prediction_tensor, target_tensor, weights):
        """Compute loss function.

        Args:
          prediction_tensor: A float tensor of shape [batch_size, num_anchors,
            num_classes] representing the predicted logits for each class
          target_tensor: A float tensor of shape [batch_size, num_anchors,
            num_classes] representing one-hot encoded classification targets
          weights: a float tensor of shape [batch_size, num_anchors]

        Returns:
          loss: a float tensor of shape [batch_size, num_anchors]
            representing the value of the loss function.
        """
        num_classes = prediction_tensor.shape[-1]
        prediction_tensor = prediction_tensor / self._logit_scale
        
        # Reshape for cross_entropy: [batch*anchors, classes]
        logits_flat = prediction_tensor.view(-1, num_classes)
        labels_flat = target_tensor.view(-1, num_classes)
        
        per_row_cross_ent = _softmax_cross_entropy_with_logits(
            labels=labels_flat,
            logits=logits_flat
        )
        
        return per_row_cross_ent.view(weights.shape) * weights