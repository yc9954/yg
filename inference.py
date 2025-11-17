"""
추론 스크립트
학습된 모델로 댄스를 생성합니다.
"""

import argparse
import json
from pathlib import Path
import numpy as np

import torch

from models import MusicConditionedDanceGenerator
from utils import AudioFeatureExtractor, TextEmbedder, save_pose_sequence


class DanceGenerator:
    """댄스 생성기"""

    def __init__(
        self,
        checkpoint_path: str,
        device: str = 'cuda'
    ):
        """
        Args:
            checkpoint_path: 체크포인트 파일 경로
            device: 디바이스
        """
        self.device = device

        # 체크포인트 로드
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        self.config = checkpoint['config']

        # 모델 초기화
        self.model = MusicConditionedDanceGenerator(
            num_joints=self.config['num_joints'],
            pose_dim=self.config['pose_dim'],
            audio_feature_dim=36,
            style_emb_dim=768,  # BERT
            d_model=self.config['d_model'],
            num_heads=self.config['num_heads'],
            num_decoder_layers=self.config['num_layers'],
            dim_feedforward=self.config['dim_feedforward'],
            dropout=self.config['dropout']
        ).to(device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        print(f"Model loaded from epoch {checkpoint['epoch']}")

        # 유틸리티
        self.audio_extractor = AudioFeatureExtractor()
        self.text_embedder = TextEmbedder('bert', device=device)

    @torch.no_grad()
    def generate(
        self,
        audio_path: str,
        style_text: str,
        num_frames: int = None,
        fps: float = 30.0,
        init_poses: np.ndarray = None,
        temperature: float = 1.0
    ) -> np.ndarray:
        """
        댄스 생성

        Args:
            audio_path: 음악 파일 경로
            style_text: 스타일 설명 (예: "강렬한 K-pop 댄스")
            num_frames: 생성할 프레임 수 (None이면 음악 길이에 맞춤)
            fps: FPS
            init_poses: 초기 포즈 (None이면 제로 포즈)
            temperature: 샘플링 온도

        Returns:
            generated_poses: (T, num_joints, pose_dim)
        """
        print(f"Generating dance for: {audio_path}")
        print(f"Style: {style_text}")

        # 오디오 특징 추출
        if num_frames is None:
            # 오디오 길이에서 프레임 수 계산
            import librosa
            y, sr = librosa.load(audio_path)
            duration = librosa.get_duration(y=y, sr=sr)
            num_frames = int(duration * fps)

        print(f"Number of frames: {num_frames}")

        audio_features_dict = self.audio_extractor.extract_frame_level_features(
            audio_path,
            num_frames=num_frames,
            fps=fps
        )
        audio_features = audio_features_dict['audio_features'].unsqueeze(0).to(self.device)  # (1, T, 36)

        # 스타일 임베딩
        style_emb = self.text_embedder.encode([style_text]).to(self.device)  # (1, 768)

        # 초기 포즈
        if init_poses is not None:
            init_poses = torch.from_numpy(init_poses).float().unsqueeze(0).to(self.device)
        else:
            # 제로 포즈로 시작
            init_poses = torch.zeros(
                1, 10, self.config['num_joints'], self.config['pose_dim'], device=self.device
            )

        # 생성
        print("Generating...")
        generated = self.model.generate(
            audio_features,
            style_emb,
            init_poses=init_poses,
            num_frames=num_frames,
            temperature=temperature
        )

        # NumPy로 변환
        generated_poses = generated[0].cpu().numpy()  # (T, J, D)

        print(f"Generated shape: {generated_poses.shape}")

        return generated_poses


def main():
    parser = argparse.ArgumentParser(description='Generate dance from music')

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Checkpoint path')
    parser.add_argument('--audio_path', type=str, required=True,
                        help='Input audio file')
    parser.add_argument('--style_text', type=str, default='dance',
                        help='Dance style description')
    parser.add_argument('--output_path', type=str, default='output/generated_dance.json',
                        help='Output JSON path')
    parser.add_argument('--num_frames', type=int, default=None,
                        help='Number of frames (None = auto from audio)')
    parser.add_argument('--fps', type=float, default=30.0,
                        help='FPS')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'])

    args = parser.parse_args()

    # 출력 디렉토리 생성
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generator 초기화
    generator = DanceGenerator(
        checkpoint_path=args.checkpoint,
        device=args.device
    )

    # 생성
    generated_poses = generator.generate(
        audio_path=args.audio_path,
        style_text=args.style_text,
        num_frames=args.num_frames,
        fps=args.fps,
        temperature=args.temperature
    )

    # 저장
    print(f"Saving to {output_path}")
    save_pose_sequence(
        generated_poses,
        str(output_path),
        fps=args.fps,
        metadata={
            'audio_path': args.audio_path,
            'style_text': args.style_text,
            'temperature': args.temperature
        }
    )

    print("Done!")


if __name__ == "__main__":
    main()
