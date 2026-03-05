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
"""run export (PyTorch Version)"""
import argparse
import os

import numpy as np
import torch

from src.utils import get_config
from src.utils import get_model_dataset


def run_export(args):
    """run export"""
    cfg_path = args.cfg_path
    ckpt_path = args.ckpt_path
    file_name = args.file_name
    file_format = args.file_format

    cfg = get_config(cfg_path)

    device_target = cfg['train_config'].get('device_target', 'GPU')
    device_id = int(os.getenv('DEVICE_ID', '0'))
    
    # Set device
    if device_target == 'GPU' and torch.cuda.is_available():
        device = torch.device(f'cuda:{device_id}')
    else:
        device = torch.device('cpu')
    
    pointpillarsnet, _ = get_model_dataset(cfg)
    pointpillarsnet.to(device)
    pointpillarsnet.eval()

    # Load checkpoint
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        # Remove 'module.' prefix if present (from DDP training)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
                
        pointpillarsnet.load_state_dict(new_state_dict)
        print(f"Loaded checkpoint from {ckpt_path}")
    else:
        print(f"Warning: Checkpoint {ckpt_path} not found, exporting random initialized model")

    # Create dummy inputs
    v = cfg['eval_input_reader']['max_number_of_voxels']
    p = cfg['model']['voxel_generator']['max_number_of_points_per_voxel']
    n = cfg['model']['num_point_features']
    
    voxels = torch.zeros((1, v, p, n), dtype=torch.float32, device=device)
    num_points = torch.zeros((1, v), dtype=torch.int32, device=device)
    coors = torch.zeros((1, v, 4), dtype=torch.int32, device=device)
    
    dummy_input = (voxels, num_points, coors)
    
    if cfg['model']['use_bev']:
        pc_range = np.array(cfg['model']['voxel_generator']['point_cloud_range'])
        voxel_size = np.array(cfg['model']['voxel_generator']['voxel_size'])
        x, y, z = ((pc_range[3:] - pc_range[:3]) / voxel_size).astype('int32')
        bev_map = torch.zeros((1, z, x * 2, y * 2), dtype=torch.float32, device=device)
        dummy_input = (voxels, num_points, coors, bev_map)

    # Export model
    if file_format.upper() == 'ONNX':
        export_onnx(pointpillarsnet, dummy_input, file_name, device)
    elif file_format.upper() == 'TORCHSCRIPT':
        export_torchscript(pointpillarsnet, dummy_input, file_name)
    else:
        print(f"Unsupported format {file_format}, defaulting to ONNX")
        export_onnx(pointpillarsnet, dummy_input, file_name, device)


def export_onnx(model, dummy_input, file_name, device):
    """Export model to ONNX format"""
    onnx_file = f"{file_name}.onnx"
    
    # Set model to eval mode
    model.eval()
    
    # Export
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            onnx_file,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['voxels', 'num_points', 'coordinates', 'bev_map'] if len(dummy_input) == 4 else ['voxels', 'num_points', 'coordinates'],
            output_names=['box_preds', 'cls_preds', 'dir_cls_preds'] if model.use_direction_classifier else ['box_preds', 'cls_preds'],
            dynamic_axes={
                'voxels': {0: 'batch_size'},
                'num_points': {0: 'batch_size'},
                'coordinates': {0: 'batch_size'},
                'box_preds': {0: 'batch_size'},
                'cls_preds': {0: 'batch_size'},
            }
        )
    print(f'{onnx_file} exported successfully!')


def export_torchscript(model, dummy_input, file_name):
    """Export model to TorchScript format"""
    pt_file = f"{file_name}.pt"
    
    # Trace the model
    model.eval()
    with torch.no_grad():
        traced_model = torch.jit.trace(model, dummy_input)
        traced_model.save(pt_file)
    print(f'{pt_file} exported successfully!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_path', required=True, help='Path to config file')
    parser.add_argument('--ckpt_path', required=True, help='Path to checkpoint file')
    parser.add_argument('--file_name', default='model', help='Output file name')
    parser.add_argument('--file_format', default='ONNX', choices=['ONNX', 'TORCHSCRIPT'], help='Export format')

    parse_args = parser.parse_args()
    run_export(parse_args)