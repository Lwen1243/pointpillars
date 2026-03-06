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
"""PointPillarsNet (PyTorch Version)"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SigmoidFocalClassificationLoss(nn.Module):
    """Sigmoid Focal Classification Loss"""
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, predictions, targets, weights=None):
        """
        predictions: [N, ...] raw logits
        targets: [N, ...] one-hot encoded
        """
        p = torch.sigmoid(predictions)
        ce_loss = F.binary_cross_entropy_with_logits(predictions, targets, reduction='none')
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        loss = alpha_t * (1. - p_t) ** self.gamma * ce_loss
        
        if weights is not None:
            if weights.dim() == 2 and loss.dim() == 3:
                weights = weights.unsqueeze(-1)
            loss = loss * weights
        return loss


class WeightedSmoothL1LocalizationLoss(nn.Module):
    """Weighted Smooth L1 Localization Loss"""
    def __init__(self, sigma=3.0, code_weights=None):
        super().__init__()
        self.sigma = sigma
        self.code_weights = code_weights
        
    def forward(self, predictions, targets, weights=None):
        """
        predictions: [N, M, code_size]
        targets: [N, M, code_size]
        weights: [N, M]
        """
        diff = predictions - targets
        abs_diff = torch.abs(diff)
        
        # Apply code weights if provided
        if self.code_weights is not None:
            code_weights = torch.tensor(self.code_weights, device=predictions.device, dtype=predictions.dtype)
            diff = diff * code_weights.view(1, 1, -1)
            abs_diff = abs_diff * code_weights.view(1, 1, -1)
        
        sigma2 = self.sigma ** 2
        mask = (abs_diff < 1.0 / sigma2).float()
        smooth_l1 = mask * (sigma2 * 0.5 * abs_diff ** 2) + (1 - mask) * (abs_diff - 0.5 / sigma2)
        
        if weights is not None:
            smooth_l1 = smooth_l1 * weights.unsqueeze(-1)
            
        return smooth_l1


class WeightedSoftmaxClassificationLoss(nn.Module):
    """Weighted Softmax Classification Loss"""
    def __init__(self):
        super().__init__()
        
    def forward(self, predictions, targets, weights=None):
        """
        predictions: [N, M, num_classes] logits
        targets: [N, M, num_classes] one-hot
        weights: [N, M]
        """
        # Cross entropy
        loss = -targets * F.log_softmax(predictions, dim=-1)
        loss = loss.sum(dim=-1)
        
        if weights is not None:
            loss = loss * weights
            
        return loss


def prepare_loss_weights(labels, pos_cls_weight=1.0, neg_cls_weight=1.0, dtype=torch.float32):
    """get cls_weights and reg_weights from labels."""
    cared = labels >= 0
    positives = labels > 0
    negatives = labels == 0
    
    negative_cls_weights = negatives.to(dtype) * neg_cls_weight
    cls_weights = negative_cls_weights + pos_cls_weight * positives.to(dtype)
    reg_weights = positives.to(dtype)
    
    pos_normalizer = positives.sum(1, keepdim=True).to(dtype)
    reg_weights = reg_weights / torch.clamp(pos_normalizer, min=1.0, max=pos_normalizer.max())
    cls_weights = cls_weights / torch.clamp(pos_normalizer, min=1.0, max=pos_normalizer.max())
    
    return cls_weights, reg_weights, cared


def create_loss(loc_loss_ftor, cls_loss_ftor, box_preds, cls_preds, cls_targets, 
                cls_weights, reg_targets, reg_weights, num_class, 
                encode_background_as_zeros=True, encode_rad_error_by_sin=True, 
                box_code_size=7):
    """create loss"""
    batch_size = box_preds.shape[0]
    box_preds = box_preds.view(batch_size, -1, box_code_size)
    
    if encode_background_as_zeros:
        cls_preds = cls_preds.view(batch_size, -1, num_class)
    else:
        cls_preds = cls_preds.view(batch_size, -1, num_class + 1)
    
    cls_targets = cls_targets.squeeze(-1)
    one_hot_targets = F.one_hot(cls_targets.long(), num_classes=num_class + 1).to(box_preds.dtype)
    
    if encode_background_as_zeros:
        one_hot_targets = one_hot_targets[..., 1:]
        
    if encode_rad_error_by_sin:
        box_preds, reg_targets = add_sin_difference(box_preds, reg_targets)
        
    loc_losses = loc_loss_ftor(box_preds, reg_targets, weights=reg_weights)
    cls_losses = cls_loss_ftor(cls_preds, one_hot_targets, weights=cls_weights)
    
    return loc_losses, cls_losses


def add_sin_difference(boxes1, boxes2):
    """add sin difference"""
    rad_pred_encoding = torch.sin(boxes1[..., -1:]) * torch.cos(boxes2[..., -1:])
    rad_tg_encoding = torch.cos(boxes1[..., -1:]) * torch.sin(boxes2[..., -1:])
    boxes1 = torch.cat([boxes1[..., :-1], rad_pred_encoding], dim=-1)
    boxes2 = torch.cat([boxes2[..., :-1], rad_tg_encoding], dim=-1)
    return boxes1, boxes2


def _get_pos_neg_loss(cls_loss, labels):
    """get pos neg loss"""
    batch_size = cls_loss.shape[0]
    if cls_loss.shape[-1] == 1 or len(cls_loss.shape) == 2:
        cls_pos_loss = (labels > 0).to(cls_loss.dtype) * cls_loss.view(batch_size, -1)
        cls_neg_loss = (labels == 0).to(cls_loss.dtype) * cls_loss.view(batch_size, -1)
        cls_pos_loss = cls_pos_loss.sum() / batch_size
        cls_neg_loss = cls_neg_loss.sum() / batch_size
    else:
        cls_pos_loss = cls_loss[..., 1:].sum() / batch_size
        cls_neg_loss = cls_loss[..., 0].sum() / batch_size
    return cls_pos_loss, cls_neg_loss


def get_direction_target(anchors, reg_targets, one_hot=True):
    """get direction target"""
    batch_size = reg_targets.shape[0]
    anchors = anchors.view(batch_size, -1, 7)
    rot_gt = reg_targets[..., -1] + anchors[..., -1]
    dir_cls_targets = (rot_gt > 0).long()
    
    if one_hot:
        dir_cls_targets = F.one_hot(dir_cls_targets, num_classes=2).to(reg_targets.dtype)
    return dir_cls_targets


def get_paddings_indicator(actual_num, max_num, axis=0):
    """Create boolean mask by actually number of a padded tensor"""
    actual_num = actual_num.unsqueeze(axis + 1)
    max_num_range = torch.arange(max_num, device=actual_num.device).view(1, -1)
    paddings_indicator = actual_num > max_num_range
    return paddings_indicator


class PFNLayer(nn.Module):
    """PFN layer"""
    def __init__(self, in_channels, out_channels, use_norm, last_layer):
        super().__init__()
        self.last_vfe = last_layer
        if not self.last_vfe:
            out_channels = out_channels // 2

        self.units = out_channels
        self.use_norm = use_norm
        
        self.linear = nn.Linear(in_channels, self.units, bias=not use_norm)
        
        if use_norm:
            # Note: MindSpore momentum=0.99 corresponds to PyTorch momentum=0.01
            self.norm = nn.BatchNorm1d(self.units, eps=1e-3, momentum=0.01)
        else:
            self.norm = nn.Identity()

    def forward(self, inputs):
        """forward"""
        # inputs: [bs, V, P, C]
        x = self.linear(inputs)
        
        # Reshape for BatchNorm1d: [bs*V, C, P]
        bs, v, p, c = x.shape
        x = x.permute(0, 1, 3, 2).contiguous().view(bs * v, c, p)
        x = self.norm(x)
        x = x.view(bs, v, c, p).permute(0, 1, 3, 2).contiguous()
        
        x = F.relu(x)
        x_max = torch.max(x, dim=2, keepdim=True)[0]  # [bs, V, 1, C]
        
        if self.last_vfe:
            return x_max.squeeze(2)  # [bs, V, C]
        
        x_repeat = x_max.repeat(1, 1, inputs.shape[2], 1)  # [bs, V, P, C]
        x_concatenated = torch.cat([x, x_repeat], dim=-1)
        return x_concatenated


class PillarFeatureNet(nn.Module):
    """Pillar feature net"""
    def __init__(
            self,
            num_input_features=4,
            use_norm=True,
            num_filters=(64,),
            with_distance=False,
            voxel_size=(0.2, 0.2, 4),
            pc_range=(0, -40, -3, 70.4, 40, 1)
    ):
        super().__init__()
        num_input_features += 5

        if with_distance:
            num_input_features += 1
        self._with_distance = with_distance

        # Create PillarFeatureNet layers
        num_filters = [num_input_features] + list(num_filters)
        pfn_layers = []

        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            if i < len(num_filters) - 2:
                last_layer = False
            else:
                last_layer = True

            pfn_layers.append(
                PFNLayer(in_filters, out_filters, use_norm, last_layer=last_layer)
            )
        self.pfn_layers = nn.Sequential(*pfn_layers)

        # Need pillar (voxel) size and x/y offset in order to calculate pillar offset
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.x_offset = self.vx / 2 + pc_range[0]
        self.y_offset = self.vy / 2 + pc_range[1]

    def forward(self, features, num_points, coors):
        """forward"""
        bs, v, p, _ = features.shape
        
        points_mean = features[:, :, :, :3].sum(dim=2, keepdim=True) / torch.clamp(num_points.view(bs, v, 1, 1), min=1.0)
        f_cluster = features[:, :, :, :3] - points_mean

        # Find distance of x, y from pillar center
        f_center = torch.zeros_like(features[:, :, :, :2])
        
        coors_float = coors.float()
        f_center[:, :, :, 0] = features[:, :, :, 0] - (coors_float[:, :, 2].unsqueeze(2) * self.vx + self.x_offset)
        f_center[:, :, :, 1] = features[:, :, :, 1] - (coors_float[:, :, 1].unsqueeze(2) * self.vy + self.y_offset)

        # Combine feature decorations
        features_ls = [features, f_cluster, f_center]
        if self._with_distance:
            points_dist = torch.norm(features[:, :, :, :3], p=2, dim=3, keepdim=True)
            features_ls.append(points_dist)
        features = torch.cat(features_ls, dim=-1)

        # The feature decorations were calculated without regard to whether pillar was empty. Need to ensure that
        # empty pillars remain set to zero.
        voxel_count = features.shape[2]
        mask = get_paddings_indicator(num_points, voxel_count, axis=1)
        mask = mask.unsqueeze(-1).to(features.dtype)
        features = features * mask
        
        # Forward pass through PFNLayers
        features = self.pfn_layers(features)
        return features


class PointPillarsScatter(nn.Module):
    """PointPillars scatter"""
    def __init__(self, output_shape, num_input_features):
        super().__init__()
        self.output_shape = output_shape
        self.ny = output_shape[2]
        self.nx = output_shape[3]
        self.n_channels = num_input_features

    def forward(self, voxel_features, coords):
        """forward"""
        batch_size = voxel_features.shape[0]
        
        # Create canvas
        canvas = torch.zeros(
            batch_size, self.n_channels, self.ny, self.nx,
            dtype=voxel_features.dtype, device=voxel_features.device
        )
        
        # coords: [batch_size, num_voxels, 4] -> [batch_id, z, y, x]
        # Set batch id
        for i in range(batch_size):
            coords[i, :, 0] = i
            
        # Scatter to canvas
        # coords indices: batch, y, x (ignore z)
        batch_idx = coords[:, :, 0].long()
        y_idx = coords[:, :, 2].long()  # 注意：通常 coords 是 [b, z, y, x]，所以 y 是索引2，x 是索引3
        x_idx = coords[:, :, 3].long()  # 如果 coords 确实是 [b, z, y, x]
        
        # 删除这行：voxel_features_t = voxel_features.permute(0, 2, 1)
        # 直接使用 voxel_features，其形状应该是 [batch, num_voxels, channels]
        
        # Advanced indexing
        canvas[batch_idx, :, y_idx, x_idx] = voxel_features
        
        return canvas


class RPN(nn.Module):
    """RPN"""
    def __init__(
            self,
            use_norm=True,
            num_class=2,
            layer_nums=(3, 5, 5),
            layer_strides=(2, 2, 2),
            num_filters=(128, 128, 256),
            upsample_strides=(1, 2, 4),
            num_upsample_filters=(256, 256, 256),
            num_input_filters=128,
            num_anchor_per_loc=2,
            encode_background_as_zeros=True,
            use_direction_classifier=True,
            use_bev=False,
            box_code_size=7,
    ):
        super().__init__()
        self._num_anchor_per_loc = num_anchor_per_loc
        self.use_direction_classifier = use_direction_classifier
        self.use_bev = use_bev
        self._use_norm = use_norm

        if len(layer_nums) != 3:
            raise ValueError(f'Layer nums must be 3, got {layer_nums}')
        if len(layer_nums) != len(layer_strides):
            raise ValueError(f'Layer nums and layer strides must have same length')
        if len(layer_nums) != len(num_filters):
            raise ValueError(f'Layer nums and num filters must have same length')
        if len(layer_nums) != len(upsample_strides):
            raise ValueError(f'Layer nums and upsample strides must have same length')
        if len(layer_nums) != len(num_upsample_filters):
            raise ValueError(f'Layer nums and num upsample strides must have same length')

        if use_norm:
            batch_norm2d_class = nn.BatchNorm2d
        else:
            batch_norm2d_class = nn.Identity

        block2_input_filters = num_filters[0]

        if use_bev:
            self.bev_extractor = nn.Sequential(
                nn.Conv2d(6, 32, 3, padding=1, bias=not use_norm),
                batch_norm2d_class(32, eps=1e-3, momentum=0.01),
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, padding=1, bias=not use_norm),
                batch_norm2d_class(64, eps=1e-3, momentum=0.01),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
            )
            block2_input_filters += 64

        self.block1 = self._make_block(
            num_input_filters, num_filters[0], layer_nums[0], layer_strides[0], use_norm
        )
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(
                num_filters[0],
                num_upsample_filters[0],
                upsample_strides[0],
                stride=upsample_strides[0],
                bias=not use_norm
            ),
            batch_norm2d_class(num_upsample_filters[0], eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )
        
        self.block2 = self._make_block(
            block2_input_filters, num_filters[1], layer_nums[1], layer_strides[1], use_norm
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(
                num_filters[1],
                num_upsample_filters[1],
                upsample_strides[1],
                stride=upsample_strides[1],
                bias=not use_norm
            ),
            batch_norm2d_class(num_upsample_filters[1], eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )
        
        self.block3 = self._make_block(
            num_filters[1], num_filters[2], layer_nums[2], layer_strides[2], use_norm
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(
                num_filters[2],
                num_upsample_filters[2],
                upsample_strides[2],
                stride=upsample_strides[2],
                bias=not use_norm
            ),
            batch_norm2d_class(num_upsample_filters[2], eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )
        
        if encode_background_as_zeros:
            num_cls = num_anchor_per_loc * num_class
        else:
            num_cls = num_anchor_per_loc * (num_class + 1)
            
        self.conv_cls = nn.Conv2d(sum(num_upsample_filters), num_cls, 1)
        self.conv_box = nn.Conv2d(sum(num_upsample_filters), num_anchor_per_loc * box_code_size, 1)
        
        if use_direction_classifier:
            self.conv_dir_cls = nn.Conv2d(sum(num_upsample_filters), num_anchor_per_loc * 2, 1)

    def _make_block(self, in_channels, out_channels, num_layers, stride, use_norm):
        """Make a conv block"""
        layers = []
        layers.append(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=not use_norm)
        )
        if use_norm:
            layers.append(nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01))
        layers.append(nn.ReLU())
        
        for _ in range(num_layers):
            layers.append(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=not use_norm)
            )
            if use_norm:
                layers.append(nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01))
            layers.append(nn.ReLU())
            
        return nn.Sequential(*layers)

    def forward(self, x, bev=None):
        """forward"""
        x = self.block1(x)
        up1 = self.deconv1(x)
        
        if self.use_bev:
            bev[:, -1] = torch.log(1 + bev[:, -1]) / np.log(16)
            bev[:, -1] = torch.clamp(bev[:, -1], min=bev[:, -1].min(), max=1.0)
            x = torch.cat([x, self.bev_extractor(bev)], dim=1)
            
        x = self.block2(x)
        up2 = self.deconv2(x)
        
        x = self.block3(x)
        up3 = self.deconv3(x)
        
        x = torch.cat([up1, up2, up3], dim=1)
        
        box_preds = self.conv_box(x)
        cls_preds = self.conv_cls(x)
        
        # [N, C, y(H), x(W)] -> [N, y(H), x(W), C]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()

        if self.use_direction_classifier:
            dir_cls_preds = self.conv_dir_cls(x)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            return box_preds, cls_preds, dir_cls_preds
        return box_preds, cls_preds


class PointPillarsNet(nn.Module):
    """PointPillars net"""
    def __init__(
            self,
            output_shape,
            num_class=2,
            num_input_features=4,
            vfe_num_filters=(32, 128),
            with_distance=False,
            rpn_layer_nums=(3, 5, 5),
            rpn_layer_strides=(2, 2, 2),
            rpn_num_filters=(128, 128, 256),
            rpn_upsample_strides=(1, 2, 4),
            rpn_num_upsample_filters=(256, 256, 256),
            use_norm=True,
            use_direction_classifier=True,
            encode_background_as_zeros=True,
            num_anchor_per_loc=2,
            code_size=7,
            use_bev=False,
            voxel_size=(0.2, 0.2, 4),
            pc_range=(0, -40, -3, 70.4, 40, 1)
    ):
        super().__init__()

        self.num_class = num_class
        self.encode_background_as_zeros = encode_background_as_zeros
        self.use_direction_classifier = use_direction_classifier
        self.use_bev = use_bev
        self.code_size = code_size
        self.num_anchor_per_loc = num_anchor_per_loc

        self.voxel_feature_extractor = PillarFeatureNet(
            num_input_features,
            use_norm,
            num_filters=vfe_num_filters,
            with_distance=with_distance,
            voxel_size=voxel_size,
            pc_range=pc_range
        )
        self.middle_feature_extractor = PointPillarsScatter(
            output_shape=output_shape,
            num_input_features=vfe_num_filters[-1]
        )
        num_rpn_input_filters = self.middle_feature_extractor.n_channels

        self.rpn = RPN(
            use_norm=True,
            num_class=num_class,
            layer_nums=rpn_layer_nums,
            layer_strides=rpn_layer_strides,
            num_filters=rpn_num_filters,
            upsample_strides=rpn_upsample_strides,
            num_upsample_filters=rpn_num_upsample_filters,
            num_input_filters=num_rpn_input_filters,
            num_anchor_per_loc=num_anchor_per_loc,
            encode_background_as_zeros=encode_background_as_zeros,
            use_direction_classifier=use_direction_classifier,
            use_bev=use_bev,
            box_code_size=code_size
        )

    def forward(self, voxels, num_points, coors, bev_map=None):
        """forward"""
        voxel_features = self.voxel_feature_extractor(voxels, num_points, coors)
        spatial_features = self.middle_feature_extractor(voxel_features, coors)
        if self.use_bev:
            preds = self.rpn(spatial_features, bev_map)
        else:
            preds = self.rpn(spatial_features)
        return preds


class PointPillarsWithLossCell(nn.Module):
    """PointPillars with loss cell"""
    def __init__(self, network, cfg):
        super().__init__()
        self.network = network
        self.cfg = cfg
        loss_cfg = cfg['loss']
        self.loss_cls = SigmoidFocalClassificationLoss(
            gamma=loss_cfg['classification_loss']['gamma'],
            alpha=loss_cfg['classification_loss']['alpha']
        )
        self.loss_loc = WeightedSmoothL1LocalizationLoss(
            sigma=loss_cfg['localization_loss']['sigma'],
            code_weights=loss_cfg['localization_loss']['code_weight']
        )
        self.loss_dir = WeightedSoftmaxClassificationLoss()
        self.w_cls_loss = loss_cfg['classification_weight']
        self.w_loc_loss = loss_cfg['localization_weight']
        self.w_dir_loss = cfg.get('direction_loss_weight', 0.2)
        self._pos_cls_weight = cfg['pos_class_weight']
        self._neg_cls_weight = cfg['neg_class_weight']
        self.code_size = network.code_size

    def forward(self, voxels, num_points, coors, bev_map, labels, reg_targets, anchors):
        """forward"""
        batch_size_dev = labels.shape[0]
        preds = self.network(voxels, num_points, coors, bev_map)
        
        if self.cfg['use_direction_classifier']:
            box_preds, cls_preds, dir_cls_preds = preds
            dir_targets = get_direction_target(anchors, reg_targets)
            dir_logits = dir_cls_preds.view(batch_size_dev, -1, 2)
            weights = (labels > 0).to(dir_logits.dtype)
            weights = weights / torch.clamp(weights.sum(-1, keepdim=True), min=1.0, max=weights.sum(-1, keepdim=True).max())
            dir_loss = self.loss_dir(dir_logits, dir_targets, weights=weights)
            dir_loss = dir_loss.sum() / batch_size_dev
            loss = dir_loss * self.w_dir_loss
        else:
            loss = 0
            box_preds, cls_preds = preds

        cls_weights, reg_weights, cared = prepare_loss_weights(
            labels,
            pos_cls_weight=self._pos_cls_weight,
            neg_cls_weight=self._neg_cls_weight,
            dtype=voxels.dtype
        )
        cls_targets = labels * cared.to(labels.dtype)
        cls_targets = cls_targets.unsqueeze(-1)

        loc_loss, cls_loss = create_loss(
            self.loss_loc,
            self.loss_cls,
            box_preds=box_preds,
            cls_preds=cls_preds,
            cls_targets=cls_targets,
            cls_weights=cls_weights,
            reg_targets=reg_targets,
            reg_weights=reg_weights,
            num_class=self.cfg['num_class'],
            encode_rad_error_by_sin=self.cfg.get('encode_rad_error_by_sin', True),
            encode_background_as_zeros=self.cfg['encode_background_as_zeros'],
            box_code_size=self.code_size,
        )
        loc_loss_reduced = loc_loss.sum() / batch_size_dev
        loc_loss_reduced *= self.w_loc_loss
        cls_pos_loss, cls_neg_loss = _get_pos_neg_loss(cls_loss, labels)
        cls_pos_loss /= self._pos_cls_weight
        cls_neg_loss /= self._neg_cls_weight
        cls_loss_reduced = cls_loss.sum() / batch_size_dev
        cls_loss_reduced *= self.w_cls_loss
        loss += loc_loss_reduced + cls_loss_reduced
        return loss


# Note: TrainingWrapper is not needed in PyTorch as gradients are handled by autograd
# The training loop should be implemented in the training script using standard PyTorch patterns