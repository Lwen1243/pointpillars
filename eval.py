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
"""Evaluation script (PyTorch Version)"""

import argparse
import os
import warnings
from time import time

import torch
from torch.utils.data import DataLoader

from src.core.eval_utils import get_official_eval_result
from src.predict import predict
from src.predict import predict_kitti_to_anno
from src.utils import get_config
from src.utils import get_model_dataset

warnings.filterwarnings('ignore')


def run_evaluate(args):
    """run evaluate"""
    cfg_path = args.cfg_path
    ckpt_path = args.ckpt_path

    cfg = get_config(cfg_path)

    device_id = int(os.getenv('DEVICE_ID', '0'))
    device_target = args.device_target

    # Set device
    if device_target == 'GPU' and torch.cuda.is_available():
        device = torch.device(f'cuda:{device_id}')
        torch.cuda.set_device(device_id)
    else:
        device = torch.device('cpu')
    
    print(f"Using device: {device}")

    model_cfg = cfg['model']
    center_limit_range = model_cfg['post_center_limit_range']

    pointpillarsnet, eval_dataset, box_coder = get_model_dataset(cfg, False)
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
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
            
        pointpillarsnet.load_state_dict(new_state_dict)
        print(f"Loaded checkpoint from {ckpt_path}")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    eval_input_cfg = cfg['eval_input_reader']
    batch_size = eval_input_cfg['batch_size']

    # Create DataLoader
    ds = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=6,
        pin_memory=True if device.type == 'cuda' else False,
        drop_last=False,
        collate_fn=eval_dataset.collate_fn if hasattr(eval_dataset, 'collate_fn') else None
    )

    class_names = list(eval_input_cfg['class_names'])

    dt_annos = []
    gt_annos = [info["annos"] for info in eval_dataset.kitti_infos]

    log_freq = 100
    len_dataset = len(eval_dataset)
    start = time()
    
    with torch.no_grad():
        for i, data in enumerate(ds):
            # Move data to device
            voxels = data["voxels"].to(device)
            num_points = data["num_points"].to(device)
            coors = data["coordinates"].to(device)
            
            bev_map = data.get('bev_map', None)
            if bev_map is not None and not isinstance(bev_map, bool):
                bev_map = bev_map.to(device)

            # Forward pass
            preds = pointpillarsnet(voxels, num_points, coors, bev_map)
            
            # Handle tuple output
            if isinstance(preds, tuple):
                if len(preds) == 2:
                    preds_dict = {
                        'box_preds': preds[0],
                        'cls_preds': preds[1],
                    }
                else:
                    preds_dict = {
                        'box_preds': preds[0],
                        'cls_preds': preds[1],
                        'dir_cls_preds': preds[2]
                    }
            else:
                preds_dict = preds

            # Predict and convert to annotations
            preds_list = predict(data, preds_dict, model_cfg, box_coder)
            dt_annos += predict_kitti_to_anno(
                preds_list,
                data,
                class_names,
                center_limit_range
            )

            if i % log_freq == 0 and i > 0:
                time_used = time() - start
                processed = min(i * batch_size, len_dataset)
                print(f'processed: {processed}/{len_dataset} imgs, time elapsed: {time_used:.2f} s',
                      flush=True)

    # Evaluate results
    result = get_official_eval_result(gt_annos, dt_annos, class_names)
    print(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_path', required=True, help='Path to config file.')
    parser.add_argument('--ckpt_path', required=True, help='Path to checkpoint.')
    parser.add_argument('--device_target', default='GPU', help='device target (GPU/CPU)')

    parse_args = parser.parse_args()
    run_evaluate(parse_args)