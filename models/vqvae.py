"""
VQ-VAE (Vector Quantized Variational AutoEncoder)
포즈 시퀀스를 이산 토큰으로 양자화합니다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class VectorQuantizer(nn.Module):
    """
    Vector Quantization 레이어
    연속 잠재 벡터를 코드북의 이산 벡터로 매핑합니다.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25
    ):
        """
        Args:
            num_embeddings: 코드북 크기
            embedding_dim: 임베딩 차원
            commitment_cost: commitment loss 가중치
        """
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        # 코드북
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z: shape (B, T, D) 또는 (B, D, H, W)

        Returns:
            quantized: 양자화된 벡터
            loss: VQ 손실
            indices: 코드북 인덱스
        """
        # Flatten
        if z.dim() == 3:  # (B, T, D)
            B, T, D = z.shape
            z_flat = z.reshape(-1, D)  # (B*T, D)
        else:  # (B, D, H, W)
            B, D, H, W = z.shape
            z_flat = z.permute(0, 2, 3, 1).reshape(-1, D)  # (B*H*W, D)

        # 코드북과의 거리 계산
        distances = (
            torch.sum(z_flat**2, dim=1, keepdim=True) +
            torch.sum(self.embedding.weight**2, dim=1) -
            2 * torch.matmul(z_flat, self.embedding.weight.t())
        )  # (B*T, num_embeddings)

        # 가장 가까운 코드 찾기
        indices = torch.argmin(distances, dim=1)  # (B*T,)
        quantized_flat = self.embedding(indices)  # (B*T, D)

        # Reshape back
        if z.dim() == 3:
            quantized = quantized_flat.reshape(B, T, D)
        else:
            quantized = quantized_flat.reshape(B, H, W, D).permute(0, 3, 1, 2)

        # Loss 계산
        # 1. Codebook loss: 코드북을 z에 가깝게
        e_latent_loss = F.mse_loss(quantized.detach(), z)

        # 2. Commitment loss: z를 양자화된 벡터에 가깝게
        q_latent_loss = F.mse_loss(quantized, z.detach())

        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight-through estimator
        quantized = z + (quantized - z).detach()

        return quantized, loss, indices


class ResidualBlock(nn.Module):
    """Residual Block"""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(),
            nn.Linear(channels, channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class PoseVQVAE(nn.Module):
    """
    포즈 시퀀스를 위한 VQ-VAE
    (B, T, num_joints, 2) -> (B, T, latent_dim) -> (B, T, num_embeddings)
    """

    def __init__(
        self,
        num_joints: int = 17,
        input_dim: int = 2,
        latent_dim: int = 256,
        num_embeddings: int = 512,
        commitment_cost: float = 0.25,
        num_residual_blocks: int = 2
    ):
        """
        Args:
            num_joints: 관절 개수
            input_dim: 관절 좌표 차원 (2 for 2D, 3 for 3D)
            latent_dim: 잠재 공간 차원
            num_embeddings: 코드북 크기
            commitment_cost: VQ commitment cost
            num_residual_blocks: 잔차 블록 개수
        """
        super().__init__()

        self.num_joints = num_joints
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings

        pose_dim = num_joints * input_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(pose_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            *[ResidualBlock(512) for _ in range(num_residual_blocks)],
            nn.Linear(512, latent_dim)
        )

        # Vector Quantizer
        self.vq = VectorQuantizer(num_embeddings, latent_dim, commitment_cost)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            *[ResidualBlock(512) for _ in range(num_residual_blocks)],
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, pose_dim)
        )

    def encode(self, poses: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            poses: (B, T, num_joints, input_dim)

        Returns:
            quantized: (B, T, latent_dim)
            indices: (B, T)
        """
        B, T, J, D = poses.shape
        assert J == self.num_joints and D == self.input_dim

        # Flatten joints
        x = poses.reshape(B, T, -1)  # (B, T, J*D)

        # Encode
        z = self.encoder(x)  # (B, T, latent_dim)

        # Quantize
        quantized, _, indices = self.vq(z)  # (B, T, latent_dim), (B*T,)
        indices = indices.reshape(B, T)

        return quantized, indices

    def decode(self, quantized: torch.Tensor) -> torch.Tensor:
        """
        Args:
            quantized: (B, T, latent_dim)

        Returns:
            poses: (B, T, num_joints, input_dim)
        """
        # Decode
        x = self.decoder(quantized)  # (B, T, J*D)

        # Reshape
        B, T, _ = x.shape
        poses = x.reshape(B, T, self.num_joints, self.input_dim)

        return poses

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """
        코드북 인덱스에서 직접 포즈 디코딩

        Args:
            indices: (B, T)

        Returns:
            poses: (B, T, num_joints, input_dim)
        """
        # 코드북에서 임베딩 가져오기
        quantized = self.vq.embedding(indices)  # (B, T, latent_dim)

        # 디코딩
        return self.decode(quantized)

    def forward(self, poses: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            poses: (B, T, num_joints, input_dim)

        Returns:
            recon_poses: 재구성된 포즈
            vq_loss: VQ 손실
            indices: 코드북 인덱스
        """
        B, T, J, D = poses.shape

        # Flatten
        x = poses.reshape(B, T, -1)

        # Encode
        z = self.encoder(x)

        # Quantize
        quantized, vq_loss, indices = self.vq(z)
        indices = indices.reshape(B, T)

        # Decode
        recon_x = self.decoder(quantized)

        # Reshape
        recon_poses = recon_x.reshape(B, T, J, D)

        return recon_poses, vq_loss, indices


class PartBasedVQVAE(nn.Module):
    """
    신체 부위별 VQ-VAE
    상체, 하체, 손 등을 별도의 코드북으로 인코딩 (X-Dancer 스타일)
    """

    BODY_PARTS = {
        'upper': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # 머리, 어깨, 팔
        'lower': [11, 12, 13, 14, 15, 16]  # 엉덩이, 다리
    }

    def __init__(
        self,
        num_joints: int = 17,
        input_dim: int = 2,
        latent_dim: int = 128,
        num_embeddings: int = 512,
        commitment_cost: float = 0.25
    ):
        super().__init__()

        self.num_joints = num_joints
        self.input_dim = input_dim

        # 각 부위별 VQ-VAE
        self.vqvaes = nn.ModuleDict()
        for part_name, joint_indices in self.BODY_PARTS.items():
            self.vqvaes[part_name] = PoseVQVAE(
                num_joints=len(joint_indices),
                input_dim=input_dim,
                latent_dim=latent_dim,
                num_embeddings=num_embeddings,
                commitment_cost=commitment_cost
            )

    def forward(self, poses: torch.Tensor):
        """
        Args:
            poses: (B, T, num_joints, input_dim)

        Returns:
            recon_poses: 재구성된 포즈
            total_vq_loss: 총 VQ 손실
            indices_dict: 부위별 인덱스 딕셔너리
        """
        B, T, J, D = poses.shape

        recon_parts = []
        total_vq_loss = 0.0
        indices_dict = {}

        for part_name, joint_indices in self.BODY_PARTS.items():
            # 해당 부위 추출
            part_poses = poses[:, :, joint_indices, :]  # (B, T, num_joints_part, D)

            # VQ-VAE
            recon_part, vq_loss, indices = self.vqvaes[part_name](part_poses)

            recon_parts.append((joint_indices, recon_part))
            total_vq_loss += vq_loss
            indices_dict[part_name] = indices

        # 재결합
        recon_poses = torch.zeros_like(poses)
        for joint_indices, recon_part in recon_parts:
            recon_poses[:, :, joint_indices, :] = recon_part

        return recon_poses, total_vq_loss, indices_dict


def test_vqvae():
    """테스트 함수"""
    print("VQ-VAE 테스트")
    print("=" * 50)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1. 기본 VQ-VAE
    print("\n1. Basic VQ-VAE")
    model = PoseVQVAE(
        num_joints=17,
        input_dim=2,
        latent_dim=256,
        num_embeddings=512
    ).to(device)

    B, T = 4, 64
    poses = torch.randn(B, T, 17, 2).to(device)

    recon_poses, vq_loss, indices = model(poses)

    print(f"Input shape: {poses.shape}")
    print(f"Reconstructed shape: {recon_poses.shape}")
    print(f"VQ loss: {vq_loss.item():.4f}")
    print(f"Indices shape: {indices.shape}")
    print(f"Unique codes used: {indices.unique().numel()} / {model.num_embeddings}")

    # Reconstruction error
    recon_error = F.mse_loss(recon_poses, poses)
    print(f"Reconstruction error: {recon_error.item():.4f}")

    # 2. Part-based VQ-VAE
    print("\n2. Part-based VQ-VAE")
    part_model = PartBasedVQVAE(
        num_joints=17,
        input_dim=2,
        latent_dim=128,
        num_embeddings=256
    ).to(device)

    recon_poses, total_vq_loss, indices_dict = part_model(poses)

    print(f"Total VQ loss: {total_vq_loss.item():.4f}")
    for part_name, indices in indices_dict.items():
        print(f"  {part_name}: {indices.shape}, unique={indices.unique().numel()}")

    # 파라미터 수
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")

    print("\n테스트 완료!")


if __name__ == "__main__":
    test_vqvae()
