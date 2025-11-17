"""
Transformer 기반 댄스 생성 모델
음악과 스타일 텍스트를 조건으로 포즈 시퀀스를 생성합니다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class PositionalEncoding(nn.Module):
    """사인-코사인 위치 인코딩"""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 위치 인코딩 계산
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """멀티헤드 어텐션"""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query, key, value: (B, T, d_model)
            mask: (B, T, T) or (T, T)

        Returns:
            output: (B, T, d_model)
            attn_weights: (B, num_heads, T, T)
        """
        B, T, _ = query.shape

        # Linear projections
        Q = self.q_proj(query).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        # Q, K, V: (B, num_heads, T, head_dim)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, num_heads, T, T)

        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention
        output = torch.matmul(attn_weights, V)  # (B, num_heads, T, head_dim)
        output = output.transpose(1, 2).reshape(B, T, self.d_model)

        output = self.out_proj(output)

        return output, attn_weights


class TransformerDecoderLayer(nn.Module):
    """Transformer 디코더 레이어"""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            tgt: (B, T, d_model) - 타겟 시퀀스
            memory: (B, S, d_model) - 인코더 출력 (음악/스타일 조건)
            tgt_mask: (T, T) - 타겟 마스크 (causal)
            memory_mask: (T, S) - 메모리 마스크

        Returns:
            output: (B, T, d_model)
        """
        # Self-attention
        tgt2, _ = self.self_attn(tgt, tgt, tgt, tgt_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Cross-attention
        tgt2, _ = self.cross_attn(tgt, memory, memory, memory_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # Feedforward
        tgt2 = self.ffn(tgt)
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt


class MusicConditionedDanceGenerator(nn.Module):
    """
    음악 조건부 댄스 생성 Transformer 모델
    X-Dancer와 유사한 구조
    """

    def __init__(
        self,
        num_joints: int = 17,
        pose_dim: int = 2,
        audio_feature_dim: int = 36,  # chroma(12) + mfcc(20) + onset(1) + rms(1) + beat(1) + tempo(1)
        style_emb_dim: int = 768,  # BERT hidden size
        d_model: int = 512,
        num_heads: int = 8,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 1000
    ):
        """
        Args:
            num_joints: 관절 개수
            pose_dim: 포즈 차원 (2D or 3D)
            audio_feature_dim: 오디오 특징 차원
            style_emb_dim: 스타일 임베딩 차원
            d_model: 모델 차원
            num_heads: 어텐션 헤드 수
            num_decoder_layers: 디코더 레이어 수
            dim_feedforward: FFN 차원
            dropout: 드롭아웃 비율
            max_seq_len: 최대 시퀀스 길이
        """
        super().__init__()

        self.num_joints = num_joints
        self.pose_dim = pose_dim
        self.d_model = d_model

        pose_input_dim = num_joints * pose_dim

        # === Input Projections ===

        # 포즈 임베딩
        self.pose_embedding = nn.Linear(pose_input_dim, d_model)

        # 오디오 특징 인코더 (프레임별)
        self.audio_encoder = nn.Sequential(
            nn.Linear(audio_feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, d_model)
        )

        # 스타일 임베딩 프로젝션
        self.style_proj = nn.Linear(style_emb_dim, d_model)

        # === Positional Encoding ===
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        # === Transformer Decoder ===
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, dim_feedforward, dropout)
            for _ in range(num_decoder_layers)
        ])

        # === Output Head ===
        self.output_head = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, pose_input_dim)
        )

        # 초기화
        self._init_weights()

    def _init_weights(self):
        """파라미터 초기화"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Causal mask 생성 (autoregressive를 위해)

        Returns:
            mask: (seq_len, seq_len)
        """
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask == 0  # 1은 attend, 0은 mask

    def forward(
        self,
        poses: torch.Tensor,
        audio_features: torch.Tensor,
        style_emb: torch.Tensor,
        use_causal_mask: bool = True
    ) -> torch.Tensor:
        """
        Args:
            poses: (B, T, num_joints, pose_dim) - 입력 포즈 시퀀스 (teacher forcing)
            audio_features: (B, T, audio_feature_dim) - 프레임별 오디오 특징
            style_emb: (B, style_emb_dim) - 스타일 임베딩 (전역)
            use_causal_mask: causal mask 사용 여부

        Returns:
            pred_poses: (B, T, num_joints, pose_dim) - 예측 포즈
        """
        B, T, J, D = poses.shape
        device = poses.device

        # === 1. Pose Embedding ===
        poses_flat = poses.reshape(B, T, -1)  # (B, T, J*D)
        tgt = self.pose_embedding(poses_flat)  # (B, T, d_model)
        tgt = self.pos_encoding(tgt)

        # === 2. Audio Encoding ===
        audio_emb = self.audio_encoder(audio_features)  # (B, T, d_model)
        audio_emb = self.pos_encoding(audio_emb)

        # === 3. Style Embedding ===
        style_emb = self.style_proj(style_emb)  # (B, d_model)
        style_emb = style_emb.unsqueeze(1)  # (B, 1, d_model)

        # === 4. Memory: 음악 + 스타일 결합 ===
        # 스타일을 시퀀스의 첫 번째 토큰으로 추가
        memory = torch.cat([style_emb, audio_emb], dim=1)  # (B, 1+T, d_model)

        # === 5. Causal Mask ===
        tgt_mask = None
        if use_causal_mask:
            tgt_mask = self.generate_causal_mask(T, device)

        # === 6. Transformer Decoder ===
        output = tgt
        for layer in self.decoder_layers:
            output = layer(output, memory, tgt_mask=tgt_mask)

        # === 7. Output Projection ===
        pred_poses_flat = self.output_head(output)  # (B, T, J*D)
        pred_poses = pred_poses_flat.reshape(B, T, J, D)

        return pred_poses

    @torch.no_grad()
    def generate(
        self,
        audio_features: torch.Tensor,
        style_emb: torch.Tensor,
        init_poses: Optional[torch.Tensor] = None,
        num_frames: Optional[int] = None,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Autoregressive 생성

        Args:
            audio_features: (B, T, audio_feature_dim)
            style_emb: (B, style_emb_dim)
            init_poses: (B, init_len, num_joints, pose_dim) - 초기 포즈 (선택)
            num_frames: 생성할 프레임 수 (None이면 audio_features 길이와 동일)
            temperature: 샘플링 온도

        Returns:
            generated_poses: (B, T, num_joints, pose_dim)
        """
        B = audio_features.size(0)
        T_audio = audio_features.size(1)
        device = audio_features.device

        if num_frames is None:
            num_frames = T_audio

        # 초기 포즈 설정
        if init_poses is None:
            # 제로 포즈로 시작
            generated = torch.zeros(
                B, 1, self.num_joints, self.pose_dim, device=device
            )
        else:
            generated = init_poses

        # Autoregressive 생성
        for t in range(generated.size(1), num_frames):
            # 현재까지 생성된 포즈로 다음 프레임 예측
            # 오디오는 전체 시퀀스 사용 (또는 현재까지만)
            audio_slice = audio_features[:, :t+1, :]

            # Forward
            pred_poses = self.forward(
                generated,
                audio_slice,
                style_emb,
                use_causal_mask=True
            )

            # 마지막 프레임만 가져오기
            next_pose = pred_poses[:, -1:, :, :]  # (B, 1, J, D)

            # Temperature 적용 (선택적)
            if temperature != 1.0:
                next_pose = next_pose / temperature

            # Concatenate
            generated = torch.cat([generated, next_pose], dim=1)

        return generated


def test_transformer():
    """테스트 함수"""
    print("Transformer Dance Generator 테스트")
    print("=" * 50)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # 모델 초기화
    model = MusicConditionedDanceGenerator(
        num_joints=17,
        pose_dim=2,
        audio_feature_dim=36,
        style_emb_dim=768,
        d_model=512,
        num_heads=8,
        num_decoder_layers=6
    ).to(device)

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 더미 데이터
    B, T = 2, 64
    poses = torch.randn(B, T, 17, 2).to(device)
    audio_features = torch.randn(B, T, 36).to(device)
    style_emb = torch.randn(B, 768).to(device)

    # Forward pass
    print("\n1. Forward pass (teacher forcing)")
    pred_poses = model(poses, audio_features, style_emb)
    print(f"Input poses: {poses.shape}")
    print(f"Predicted poses: {pred_poses.shape}")

    loss = F.mse_loss(pred_poses, poses)
    print(f"MSE loss: {loss.item():.4f}")

    # Autoregressive generation
    print("\n2. Autoregressive generation")
    init_poses = torch.randn(B, 10, 17, 2).to(device)
    generated = model.generate(
        audio_features,
        style_emb,
        init_poses=init_poses,
        num_frames=T
    )
    print(f"Generated poses: {generated.shape}")

    # Causal mask 테스트
    print("\n3. Causal mask")
    mask = model.generate_causal_mask(10, device)
    print(f"Mask shape: {mask.shape}")
    print(f"Mask:\n{mask.int()}")

    print("\n테스트 완료!")


if __name__ == "__main__":
    test_transformer()
