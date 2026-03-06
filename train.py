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
"""Train script"""
import argparse
import datetime
import os
import warnings
from pathlib import Path
from time import time
import numpy as np

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from src.pointpillars import PointPillarsWithLossCell
from src.utils import get_config
from src.utils import get_model_dataset

warnings.filterwarnings('ignore')


def set_default(args):
    """set default"""
    torch.manual_seed(0)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    cfg_path = Path(args.cfg_path)
    save_path = Path(args.save_path)
    save_path.mkdir(exist_ok=True, parents=True)

    cfg = get_config(cfg_path)

    is_distributed = int(args.is_distributed)
    device_target = args.device_target

    if device_target == 'GPU' and torch.cuda.is_available():
        device = torch.device('cuda')
    elif device_target == 'CPU':
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if is_distributed:
        # init distributed
        dist.init_process_group(backend='nccl' if device.type == 'cuda' else 'gloo')
        rank = dist.get_rank()
        device_num = dist.get_world_size()
        torch.cuda.set_device(rank % torch.cuda.device_count())
    else:
        rank = 0
        device_num = 1

    return cfg, rank, device_num, device


def run_train(args):
    """run train"""
    cfg, rank, device_num, device = set_default(args)
    save_ckpt_log_flag = rank == 0

    train_cfg = cfg['train_config']

    pointpillarsnet, dataset = get_model_dataset(cfg, True)
    pointpillarsnet = pointpillarsnet.to(device)
    
    if save_ckpt_log_flag:
        print('PointPillarsNet created', flush=True)

    input_cfg = cfg['train_input_reader']
    n_epochs = input_cfg['max_num_epochs']
    batch_size = input_cfg['batch_size']

    steps_per_epoch = int(len(dataset) / batch_size / device_num)
    
    # PyTorch learning rate scheduler
    lr_cfg = train_cfg['learning_rate']
    initial_lr = lr_cfg['initial_learning_rate']
    decay_rate = lr_cfg['decay_rate']
    decay_epoch = lr_cfg['decay_epoch']
    
    optimizer = Adam(
        pointpillarsnet.parameters(),
        lr=initial_lr,
        weight_decay=train_cfg['weight_decay']
    )
    
    # Exponential decay scheduler
    scheduler = ExponentialLR(optimizer, gamma=decay_rate ** (1.0 / decay_epoch))

    # Wrap model with loss
    pointpillarsnet_wloss = PointPillarsWithLossCell(pointpillarsnet, cfg['model'])
    pointpillarsnet_wloss = pointpillarsnet_wloss.to(device)

    # Distributed training wrapper
    if device_num > 1:
        pointpillarsnet_wloss = DDP(
            pointpillarsnet_wloss, 
            device_ids=[rank % torch.cuda.device_count()],
            output_device=rank % torch.cuda.device_count(),
            find_unused_parameters=True
        )

    # DataLoader setup
    train_column_names = dataset.data_keys
    
    if device_num > 1:
        sampler = DistributedSampler(
            dataset, 
            num_replicas=device_num, 
            rank=rank,
            shuffle=True
        )
    else:
        sampler = None

    ds = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=1,  # 可根据需要调整
        pin_memory=True if device.type == 'cuda' else False,
        drop_last=True,
        collate_fn=custom_collate_fn  # 需要自定义的 collate function
    )

    if save_ckpt_log_flag:
        # Checkpoint configuration
        keep_checkpoint_max = train_cfg['keep_checkpoint_max']
        saved_checkpoints = []

    log_freq = train_cfg['log_frequency_step']
    old_progress = -1
    start = time()
    
    pointpillarsnet_wloss.train()
    
    for epoch in range(n_epochs):
        if device_num > 1:
            sampler.set_epoch(epoch)
            
        for i, data in enumerate(ds):
            global_step = epoch * steps_per_epoch + i
            
            # Move data to device
            voxels = data["voxels"].to(device)
            num_points = data["num_points"].to(device)
            coors = data["coordinates"].to(device)
            labels = data['labels'].to(device)
            reg_targets = data['reg_targets'].to(device)
            batch_anchors = data["anchors"].to(device)
            bev_map = data.get('bev_map', None)
            if bev_map is not None:
                bev_map = bev_map.to(device)

            # Forward pass
            optimizer.zero_grad()
            loss = pointpillarsnet_wloss(
                voxels, num_points, coors, bev_map, 
                labels, reg_targets, batch_anchors
            )
            
            # Backward pass
            if isinstance(loss, tuple):
                loss_value = loss[0]
            else:
                loss_value = loss
                
            if device_num > 1:
                loss_value = loss_value.mean()  # 平均梯度
            
            loss_value.backward()
            optimizer.step()
            
            # Learning rate decay
            if (global_step + 1) % (decay_epoch * steps_per_epoch) == 0:
                scheduler.step()

            # Logging and checkpoint
            if save_ckpt_log_flag:
                # Save checkpoint
                if (global_step + 1) % steps_per_epoch == 0 or (global_step + 1) == n_epochs * steps_per_epoch:
                    checkpoint_path = os.path.join(
                        args.save_path, 
                        f'pointpillars_epoch{epoch+1}.ckpt'
                    )
                    
                    # 保存模型状态
                    state_dict = pointpillarsnet_wloss.module.network.state_dict() if isinstance(pointpillarsnet_wloss, DDP) else pointpillarsnet_wloss.network.state_dict()
                    
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': state_dict,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'loss': loss_value.item(),
                    }, checkpoint_path)
                    
                    saved_checkpoints.append(checkpoint_path)
                    
                    # Keep only max checkpoints
                    if len(saved_checkpoints) > keep_checkpoint_max:
                        oldest_ckpt = saved_checkpoints.pop(0)
                        if os.path.exists(oldest_ckpt):
                            os.remove(oldest_ckpt)
                
                if global_step % log_freq == 0:
                    time_used = time() - start
                    fps = (global_step - old_progress) * batch_size * device_num / time_used
                    date_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    print(f'{date_time} epoch:{epoch}, iter:{global_step}, '
                          f'loss:{loss_value.item():.4f}, fps:{round(fps, 2)} imgs/sec',
                          flush=True)
                    start = time()
                    old_progress = global_step

    if device_num > 1:
        dist.destroy_process_group()


def custom_collate_fn(batch):
    """
    batch: list of dicts, 每个 dict 是 __getitem__ 返回的
    """
    keys = batch[0].keys()
    collated = {}
    
    for key in keys:
        values = [item[key] for item in batch]
        
        # 如果值是 numpy array，先转为 tensor 再 stack
        if isinstance(values[0], np.ndarray):
            # 注意：如果 arrays 长度不同（变长数据），不能用 stack，需要用 list
            try:
                values = np.stack(values, axis=0)
                collated[key] = torch.from_numpy(values)
            except ValueError:
                # 变长数据（如不同数量的 voxels），保持为 list
                collated[key] = [torch.from_numpy(v) if isinstance(v, np.ndarray) else v for v in values]
        
        # 如果值已经是 tensor
        elif isinstance(values[0], torch.Tensor):
            try:
                collated[key] = torch.stack(values, dim=0)
            except RuntimeError:
                # 变长 tensor，保持为 list
                collated[key] = values
        
        # 其他类型（如 int, float）
        else:
            collated[key] = values
            
    return collated


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_path', required=True, help='Path to config file.')
    parser.add_argument('--save_path', required=True, help='Path to save checkpoints.')
    parser.add_argument('--device_target', default='GPU', help='device target')
    parser.add_argument('--is_distributed', default=0, help='distributed train')
    parser.add_argument('--local_rank', default=0, type=int, help='local rank for ddp')
    parse_args = parser.parse_args()
    run_train(parse_args)