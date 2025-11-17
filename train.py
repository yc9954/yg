"""
학습 스크립트
음악 조건부 댄스 생성 모델을 학습합니다.
"""

import os
import argparse
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

from models import MusicConditionedDanceGenerator
from dance_datasets import DanceMotionDataset, create_dummy_dataset
from utils import TextEmbedder, COCO_SKELETON, compute_bone_lengths


class DanceGeneratorTrainer:
    """댄스 생성 모델 학습기"""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        text_embedder: TextEmbedder,
        config: dict,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.text_embedder = text_embedder
        self.config = config
        self.device = device

        # Optimizer
        if config['optimizer'] == 'adam':
            self.optimizer = Adam(
                model.parameters(),
                lr=config['lr'],
                betas=(0.9, 0.999),
                weight_decay=config.get('weight_decay', 0.0)
            )
        elif config['optimizer'] == 'adamw':
            self.optimizer = AdamW(
                model.parameters(),
                lr=config['lr'],
                weight_decay=config.get('weight_decay', 1e-4)
            )

        # Scheduler
        if config.get('scheduler') == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=config['epochs'],
                eta_min=config.get('min_lr', 1e-6)
            )
        elif config.get('scheduler') == 'step':
            self.scheduler = StepLR(
                self.optimizer,
                step_size=config.get('step_size', 30),
                gamma=config.get('gamma', 0.1)
            )
        else:
            self.scheduler = None

        # Loss weights
        self.loss_weights = {
            'pose': config.get('loss_weight_pose', 1.0),
            'velocity': config.get('loss_weight_velocity', 0.5),
            'bone': config.get('loss_weight_bone', 0.1)
        }

        # Logging
        self.global_step = 0
        self.best_val_loss = float('inf')

        # Output directory
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        with open(self.output_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)

    def compute_losses(
        self,
        pred_poses: torch.Tensor,
        target_poses: torch.Tensor,
        target_vel: torch.Tensor
    ) -> dict:
        """
        손실 함수 계산

        Args:
            pred_poses: (B, T, J, D)
            target_poses: (B, T, J, D)
            target_vel: (B, T, J, D)

        Returns:
            losses 딕셔너리
        """
        # 1. Pose reconstruction loss (MSE)
        loss_pose = F.mse_loss(pred_poses, target_poses)

        # 2. Velocity loss
        pred_vel = torch.zeros_like(pred_poses)
        pred_vel[:, 1:] = pred_poses[:, 1:] - pred_poses[:, :-1]
        loss_velocity = F.mse_loss(pred_vel, target_vel)

        # 3. Bone length consistency loss
        # 예측과 타겟의 본 길이가 유사해야 함
        pred_bone_lengths = compute_bone_lengths(
            pred_poses.cpu().numpy(),
            COCO_SKELETON
        )
        target_bone_lengths = compute_bone_lengths(
            target_poses.cpu().numpy(),
            COCO_SKELETON
        )
        pred_bone_lengths = torch.from_numpy(pred_bone_lengths).to(self.device)
        target_bone_lengths = torch.from_numpy(target_bone_lengths).to(self.device)

        loss_bone = F.mse_loss(pred_bone_lengths, target_bone_lengths)

        # Total loss
        total_loss = (
            self.loss_weights['pose'] * loss_pose +
            self.loss_weights['velocity'] * loss_velocity +
            self.loss_weights['bone'] * loss_bone
        )

        return {
            'total': total_loss,
            'pose': loss_pose,
            'velocity': loss_velocity,
            'bone': loss_bone
        }

    def train_epoch(self, epoch: int) -> dict:
        """한 에폭 학습"""
        self.model.train()

        epoch_losses = {
            'total': 0.0,
            'pose': 0.0,
            'velocity': 0.0,
            'bone': 0.0
        }

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")

        for batch_idx, batch in enumerate(pbar):
            # 데이터 준비
            audio_features = batch['audio_features'].to(self.device)
            poses = batch['poses'].to(self.device)
            pose_vel = batch['pose_vel'].to(self.device)
            style_texts = batch['style_texts']

            # 스타일 임베딩
            with torch.no_grad():
                style_emb = self.text_embedder.encode(style_texts).to(self.device)

            # Forward
            pred_poses = self.model(poses, audio_features, style_emb)

            # 손실 계산
            losses = self.compute_losses(pred_poses, poses, pose_vel)

            # Backward
            self.optimizer.zero_grad()
            losses['total'].backward()

            # Gradient clipping
            if self.config.get('grad_clip'):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['grad_clip']
                )

            self.optimizer.step()

            # Logging
            for key in epoch_losses:
                epoch_losses[key] += losses[key].item()

            self.global_step += 1

            # Update progress bar
            pbar.set_postfix({
                'loss': losses['total'].item(),
                'pose': losses['pose'].item(),
                'vel': losses['velocity'].item()
            })

        # 평균 손실
        num_batches = len(self.train_loader)
        for key in epoch_losses:
            epoch_losses[key] /= num_batches

        return epoch_losses

    @torch.no_grad()
    def validate(self) -> dict:
        """검증"""
        self.model.eval()

        val_losses = {
            'total': 0.0,
            'pose': 0.0,
            'velocity': 0.0,
            'bone': 0.0
        }

        for batch in tqdm(self.val_loader, desc="Validation"):
            audio_features = batch['audio_features'].to(self.device)
            poses = batch['poses'].to(self.device)
            pose_vel = batch['pose_vel'].to(self.device)
            style_texts = batch['style_texts']

            style_emb = self.text_embedder.encode(style_texts).to(self.device)

            pred_poses = self.model(poses, audio_features, style_emb)

            losses = self.compute_losses(pred_poses, poses, pose_vel)

            for key in val_losses:
                val_losses[key] += losses[key].item()

        # 평균
        num_batches = len(self.val_loader)
        for key in val_losses:
            val_losses[key] /= num_batches

        return val_losses

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """체크포인트 저장"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'config': self.config
        }

        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        # 최신 체크포인트
        checkpoint_path = self.output_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)

        # 베스트 모델
        if is_best:
            best_path = self.output_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"Best model saved at epoch {epoch}")

    def train(self):
        """전체 학습 루프"""
        print("=" * 50)
        print("Training started")
        print(f"Device: {self.device}")
        print(f"Epochs: {self.config['epochs']}")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        print("=" * 50)

        for epoch in range(1, self.config['epochs'] + 1):
            # 학습
            train_losses = self.train_epoch(epoch)

            # 검증
            if epoch % self.config.get('val_interval', 1) == 0:
                val_losses = self.validate()

                print(f"\nEpoch {epoch}:")
                print(f"  Train Loss: {train_losses['total']:.4f}")
                print(f"    - Pose: {train_losses['pose']:.4f}")
                print(f"    - Velocity: {train_losses['velocity']:.4f}")
                print(f"    - Bone: {train_losses['bone']:.4f}")
                print(f"  Val Loss: {val_losses['total']:.4f}")

                # 베스트 모델 체크
                is_best = val_losses['total'] < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_losses['total']

                # 체크포인트 저장
                if epoch % self.config.get('save_interval', 10) == 0 or is_best:
                    self.save_checkpoint(epoch, is_best)

            # Scheduler step
            if self.scheduler:
                self.scheduler.step()

        print("\nTraining completed!")
        print(f"Best val loss: {self.best_val_loss:.4f}")


def collate_fn(batch):
    """배치 collate 함수"""
    audio_features = torch.stack([item['audio_features'] for item in batch])
    poses = torch.stack([item['poses'] for item in batch])
    pose_vel = torch.stack([item['pose_vel'] for item in batch])
    style_texts = [item['style_text'] for item in batch]

    return {
        'audio_features': audio_features,
        'poses': poses,
        'pose_vel': pose_vel,
        'style_texts': style_texts
    }


def main():
    parser = argparse.ArgumentParser(description='Train dance generation model')

    # Data
    parser.add_argument('--data_dir', type=str, default='/tmp/dummy_dance_dataset',
                        help='Data directory')
    parser.add_argument('--create_dummy', action='store_true',
                        help='Create dummy dataset for testing')
    parser.add_argument('--num_dummy_samples', type=int, default=100,
                        help='Number of dummy samples')

    # Model
    parser.add_argument('--num_joints', type=int, default=17)
    parser.add_argument('--pose_dim', type=int, default=2)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--dim_feedforward', type=int, default=2048)
    parser.add_argument('--dropout', type=float, default=0.1)

    # Training
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'adamw'])
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'step', 'none'])
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--grad_clip', type=float, default=1.0)

    # Loss weights
    parser.add_argument('--loss_weight_pose', type=float, default=1.0)
    parser.add_argument('--loss_weight_velocity', type=float, default=0.5)
    parser.add_argument('--loss_weight_bone', type=float, default=0.1)

    # Misc
    parser.add_argument('--seq_length', type=int, default=64)
    parser.add_argument('--fps', type=float, default=30.0)
    parser.add_argument('--val_split', type=float, default=0.1)
    parser.add_argument('--val_interval', type=int, default=1)
    parser.add_argument('--save_interval', type=int, default=10)
    parser.add_argument('--output_dir', type=str, default='./checkpoints')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)

    args = parser.parse_args()

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 더미 데이터셋 생성 (테스트용)
    if args.create_dummy:
        print(f"Creating dummy dataset with {args.num_dummy_samples} samples...")
        create_dummy_dataset(args.data_dir, args.num_dummy_samples)

    # Dataset
    print("Loading dataset...")
    full_dataset = DanceMotionDataset(
        data_dir=args.data_dir,
        seq_length=args.seq_length,
        fps=args.fps,
        num_joints=args.num_joints,
        normalize_pose=True,
        cache_audio_features=True,
        train=True
    )

    # Train/Val split
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size]
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    # Text embedder
    print("Loading text embedder...")
    text_embedder = TextEmbedder('bert', device=device)

    # Model
    print("Creating model...")
    model = MusicConditionedDanceGenerator(
        num_joints=args.num_joints,
        pose_dim=args.pose_dim,
        audio_feature_dim=36,
        style_emb_dim=text_embedder.get_hidden_size(),
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_decoder_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Config
    config = vars(args)

    # Trainer
    trainer = DanceGeneratorTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        text_embedder=text_embedder,
        config=config,
        device=device
    )

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
