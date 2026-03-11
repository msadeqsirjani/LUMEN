#!/usr/bin/env python3
"""
Knowledge distillation training for LUMEN.

A pre-trained LUMEN teacher guides the student.
Follows MambaLiteSR (arXiv:2502.14090): L = α·L1(student, teacher) + (1-α)·L1(student, GT)

Reads the same options/ YAML as basicsr/train.py (options/train/ schema).

Usage:
    python scripts/train_distill.py options/train/train_LUMEN_x4_DIV2K.yml \\
        --teacher-weights pretrained/LUMEN_x4.pth
    python scripts/train_distill.py options/train/train_LUMEN_x4_DIV2K.yml \\
        --teacher-weights pretrained/LUMEN_x4.pth --alpha 0.8
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from basicsr.archs.lumen.model import LUMEN
from basicsr.archs import build_network
from basicsr.data.sr_pair_dataset import SRPairDataset
from basicsr.distill import DistillationLoss
from basicsr.utils import psnr, ssim, AverageMeter, get_logger, save_checkpoint, load_checkpoint

def parse_args():
    p = argparse.ArgumentParser(description='LUMEN Distillation Training')
    p.add_argument('config', type=str, help='options/train/train_LUMEN_x4_DIV2K.yml')
    p.add_argument('--teacher-weights', type=str, required=True)
    p.add_argument('--alpha', type=float, default=0.8,
                   help='KD weight — MambaLiteSR default: 0.8')
    p.add_argument('--resume', type=str, default=None)
    p.add_argument('--gpu', type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Paths (options/ schema)
    save_dir = os.path.join(cfg.get('path', {}).get('experiments_root', 'experiments'), cfg['name'])
    os.makedirs(save_dir, exist_ok=True)
    logger = get_logger('lumen_distill', os.path.join(save_dir, 'distill.log'))

    # Teacher (frozen)
    teacher = LUMEN(upscale=cfg['scale']).to(device).eval()
    load_checkpoint(args.teacher_weights, teacher)
    for p in teacher.parameters():
        p.requires_grad_(False)
    logger.info(f"Teacher: LUMEN ({sum(p.numel() for p in teacher.parameters()):,} params)")

    # Student
    student = build_network(cfg['network_g']).to(device)
    logger.info(f"Student: {sum(p.numel() for p in student.parameters() if p.requires_grad):,} params")

    # Dataset (options/ schema: datasets.train.dataroot_gt / dataroot_lq)
    td = cfg['datasets']['train']
    train_ds = SRPairDataset(
        hr_dir=td['dataroot_gt'],
        lr_dir=td['dataroot_lq'],
        patch_size=td.get('gt_size', 256),
        scale=cfg['scale'],
        augment=td.get('use_hflip', True) or td.get('use_rot', True),
        cache=False,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=td.get('batch_size', 32),
        shuffle=True,
        num_workers=td.get('num_workers', 4),
        pin_memory=True,
        drop_last=True,
    )

    # Optimiser & scheduler (options/ schema: train.optim_g / train.scheduler)
    tcfg = cfg['train']
    opt_cfg = tcfg['optim_g']
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=float(opt_cfg['lr']),
        weight_decay=float(opt_cfg.get('weight_decay', 1e-4)),
        betas=tuple(opt_cfg.get('betas', [0.9, 0.999])),
    )
    total_iter  = int(tcfg['total_iter'])
    warmup_iter = int(tcfg.get('warmup_iter', 0))
    sched_cfg   = tcfg.get('scheduler', {})
    sched_type  = sched_cfg.get('type', 'MultiStepLR').lower()
    if sched_type == 'cosineannealinglr':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(total_iter - warmup_iter, 1), eta_min=1e-6)
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=sched_cfg.get('milestones', [total_iter // 2]),
            gamma=float(sched_cfg.get('gamma', 0.5)),
        )

    criterion = DistillationLoss(alpha=args.alpha)
    val_freq  = int(cfg.get('val', {}).get('val_freq', 5000))
    base_lr   = float(opt_cfg['lr'])

    # Val dataset (options/ schema: datasets.val)
    vd = cfg['datasets'].get('val', {})
    val_ds = SRPairDataset(
        hr_dir=vd['dataroot_gt'], lr_dir=vd['dataroot_lq'],
        patch_size=0, scale=cfg['scale'], augment=False,
    ) if vd else None

    start_iter, best_psnr = 0, 0.0
    if args.resume:
        state = load_checkpoint(args.resume, student)
        start_iter = state.get('epoch', 0)       # stored as iter count
        best_psnr  = state.get('best_psnr', 0.0)
        logger.info(f"Resumed from iter {start_iter}, best PSNR {best_psnr:.2f}")

    cur_iter = start_iter
    loss_m, kd_m, gt_m = AverageMeter(), AverageMeter(), AverageMeter()

    student.train()
    while cur_iter < total_iter:
        for batch in train_loader:
            if cur_iter >= total_iter:
                break

            # LR warmup
            if cur_iter < warmup_iter:
                lr = base_lr * (cur_iter + 1) / warmup_iter
                for pg in optimizer.param_groups:
                    pg['lr'] = lr
            elif cur_iter == warmup_iter:
                for pg in optimizer.param_groups:
                    pg['lr'] = base_lr

            lq = batch['lq'].to(device, non_blocking=True)
            gt = batch['gt'].to(device, non_blocking=True)
            with torch.no_grad():
                teacher_out = teacher(lq)
            student_out = student(lq)
            loss, kd, gt_loss = criterion(student_out, teacher_out, gt)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            if cur_iter >= warmup_iter:
                scheduler.step()

            n = lq.size(0)
            loss_m.update(loss.item(), n)
            kd_m.update(kd.item(), n)
            gt_m.update(gt_loss.item(), n)
            cur_iter += 1

            if cur_iter % 100 == 0:
                logger.info(
                    f"[{cur_iter:7d}/{total_iter}]  "
                    f"Loss: {loss_m.avg:.5f}  KD: {kd_m.avg:.5f}  GT: {gt_m.avg:.5f}  "
                    f"LR: {optimizer.param_groups[0]['lr']:.2e}"
                )
                loss_m.reset(); kd_m.reset(); gt_m.reset()

            if cur_iter % val_freq == 0 or cur_iter == total_iter:
                if val_ds is not None:
                    student.eval()
                    pm, sm = AverageMeter(), AverageMeter()
                    with torch.no_grad():
                        for s in val_ds:
                            lq_ = s['lq'].unsqueeze(0).to(device)
                            gt_ = s['gt'].unsqueeze(0).to(device)
                            pred = student(lq_).clamp(0, 1)
                            pm.update(psnr(pred, gt_))
                            sm.update(ssim(pred, gt_))
                    logger.info(f"[{cur_iter}] Val  PSNR: {pm.avg:.2f}  SSIM: {sm.avg:.4f}")
                    is_best = pm.avg > best_psnr
                    if is_best:
                        best_psnr = pm.avg
                        logger.info(f"  New best: {best_psnr:.2f}")
                    save_checkpoint(student, optimizer, scheduler, cur_iter, best_psnr,
                                    os.path.join(save_dir, 'ckpt_latest.pth'), is_best)
                    student.train()

    logger.info(f"Done. Best PSNR: {best_psnr:.2f}")


if __name__ == '__main__':
    main()
