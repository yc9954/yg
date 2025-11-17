"""
댄스 모션 데이터셋 클래스
AIST++, HumanML3D 등의 데이터를 PyTorch Dataset으로 래핑합니다.
"""

import os
import json
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import sys

# 상위 디렉토리의 utils 모듈 import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import AudioFeatureExtractor, TextEmbedder, PoseNormalizer


class DanceMotionDataset(Dataset):
    """
    일반 댄스 모션 데이터셋
    (음악 파일, 스타일 텍스트, 포즈 시퀀스) 튜플을 로드합니다.
    """

    def __init__(
        self,
        data_dir: str,
        seq_length: int = 64,
        fps: float = 30.0,
        num_joints: int = 17,
        audio_sr: int = 22050,
        normalize_pose: bool = True,
        cache_audio_features: bool = True,
        train: bool = True
    ):
        """
        Args:
            data_dir: 데이터 디렉토리 (metadata.json 포함)
            seq_length: 시퀀스 길이 (프레임 수)
            fps: 비디오 FPS
            num_joints: 관절 개수
            audio_sr: 오디오 샘플링 레이트
            normalize_pose: 포즈 정규화 여부
            cache_audio_features: 오디오 특징 캐싱 여부
            train: 학습 모드 여부
        """
        self.data_dir = Path(data_dir)
        self.seq_length = seq_length
        self.fps = fps
        self.num_joints = num_joints
        self.normalize_pose = normalize_pose
        self.train = train

        # 메타데이터 로드
        metadata_path = self.data_dir / 'metadata.json'
        if not metadata_path.exists():
            raise FileNotFoundError(f"metadata.json not found in {data_dir}")

        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        # 데이터 리스트
        self.data_list = self.metadata.get('data', [])
        if len(self.data_list) == 0:
            raise ValueError("No data found in metadata.json")

        # 유틸리티 초기화
        self.audio_extractor = AudioFeatureExtractor(sr=audio_sr)
        self.pose_normalizer = PoseNormalizer(num_joints)

        # 오디오 특징 캐시
        self.cache_audio_features = cache_audio_features
        self.audio_cache = {}

        print(f"Loaded {len(self.data_list)} sequences from {data_dir}")

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            딕셔너리:
                - audio_features: (seq_length, audio_dim)
                - style_text: str
                - poses: (seq_length, num_joints, 2)
                - pose_vel: (seq_length, num_joints, 2) - 속도
        """
        item = self.data_list[idx]

        # 파일 경로
        audio_path = self.data_dir / item['audio_path']
        pose_path = self.data_dir / item['pose_path']
        style_text = item.get('style', 'dance')

        # 오디오 특징 추출
        if self.cache_audio_features and str(audio_path) in self.audio_cache:
            audio_features = self.audio_cache[str(audio_path)]
        else:
            duration = self.seq_length / self.fps
            audio_features_dict = self.audio_extractor.extract_frame_level_features(
                str(audio_path),
                num_frames=self.seq_length,
                fps=self.fps
            )
            audio_features = audio_features_dict['audio_features']  # (T, 36)

            if self.cache_audio_features:
                self.audio_cache[str(audio_path)] = audio_features

        # 포즈 로드
        if pose_path.suffix == '.npy':
            poses = np.load(pose_path)  # (T, num_joints, 2 or 3)
        elif pose_path.suffix == '.json':
            from utils import load_pose_sequence
            poses, _ = load_pose_sequence(str(pose_path))
        else:
            raise ValueError(f"Unknown pose file format: {pose_path.suffix}")

        # 2D만 사용
        if poses.shape[-1] == 3:
            poses = poses[..., :2]

        # 시퀀스 길이 맞추기
        T = poses.shape[0]
        if T > self.seq_length:
            # 랜덤 크롭 (train) 또는 중앙 크롭 (test)
            if self.train:
                start_idx = np.random.randint(0, T - self.seq_length + 1)
            else:
                start_idx = (T - self.seq_length) // 2
            poses = poses[start_idx:start_idx + self.seq_length]
        elif T < self.seq_length:
            # 패딩
            pad_length = self.seq_length - T
            poses = np.pad(poses, ((0, pad_length), (0, 0), (0, 0)), mode='edge')

        # 정규화
        if self.normalize_pose:
            poses = self.pose_normalizer.normalize_poses(poses)

        # 속도 계산 (1차 미분)
        pose_vel = np.zeros_like(poses)
        pose_vel[1:] = poses[1:] - poses[:-1]

        # 텐서 변환
        poses = torch.from_numpy(poses).float()
        pose_vel = torch.from_numpy(pose_vel).float()

        return {
            'audio_features': audio_features,  # (T, 36)
            'style_text': style_text,
            'poses': poses,  # (T, num_joints, 2)
            'pose_vel': pose_vel,  # (T, num_joints, 2)
            'audio_path': str(audio_path),
            'pose_path': str(pose_path)
        }


class AIST_Dataset(Dataset):
    """
    AIST++ 데이터셋 로더
    https://google.github.io/aistplusplus_dataset/
    """

    GENRES = [
        'gBR', 'gPO', 'gLO', 'gMH', 'gLH',
        'gHO', 'gWA', 'gKR', 'gJS', 'gJB'
    ]

    def __init__(
        self,
        aist_dir: str,
        seq_length: int = 64,
        fps: float = 60.0,  # AIST++는 60 FPS
        normalize_pose: bool = True,
        split: str = 'train',
        split_ratio: float = 0.9
    ):
        """
        Args:
            aist_dir: AIST++ 데이터 디렉토리
            seq_length: 시퀀스 길이
            fps: FPS
            normalize_pose: 정규화 여부
            split: 'train' or 'test'
            split_ratio: train/test 분할 비율
        """
        self.aist_dir = Path(aist_dir)
        self.seq_length = seq_length
        self.fps = fps
        self.normalize_pose = normalize_pose
        self.split = split

        # 파일 리스트 로드
        self.motion_dir = self.aist_dir / 'motions'
        self.audio_dir = self.aist_dir / 'audio'

        if not self.motion_dir.exists():
            raise FileNotFoundError(f"Motions directory not found: {self.motion_dir}")

        # 모션 파일 리스트
        motion_files = sorted(list(self.motion_dir.glob('*.pkl')))

        # train/test 분할
        split_idx = int(len(motion_files) * split_ratio)
        if split == 'train':
            self.motion_files = motion_files[:split_idx]
        else:
            self.motion_files = motion_files[split_idx:]

        self.pose_normalizer = PoseNormalizer()
        self.audio_extractor = AudioFeatureExtractor()

        print(f"AIST++ {split}: {len(self.motion_files)} sequences")

    def __len__(self) -> int:
        return len(self.motion_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        motion_file = self.motion_files[idx]

        # 모션 데이터 로드
        with open(motion_file, 'rb') as f:
            motion_data = pickle.load(f)

        # AIST++는 3D SMPL 포즈를 제공
        # 여기서는 간단히 joint positions를 사용
        # 실제로는 SMPL로부터 관절 위치 추출 필요
        if 'smpl_poses' in motion_data:
            # SMPL 파라미터가 있다면 (실제 구현에서는 SMPL 모델 필요)
            # 여기서는 더미 데이터 생성
            T = motion_data.get('smpl_poses').shape[0]
            poses = np.random.randn(T, 17, 2)  # 실제로는 SMPL -> joints
        elif 'joints' in motion_data:
            poses = motion_data['joints'][:, :17, :2]  # 3D -> 2D
        else:
            # 기본값
            poses = np.random.randn(300, 17, 2)

        # 시퀀스 길이 조정
        T = poses.shape[0]
        if T > self.seq_length:
            start_idx = np.random.randint(0, T - self.seq_length + 1)
            poses = poses[start_idx:start_idx + self.seq_length]
        else:
            poses = np.pad(poses, ((0, self.seq_length - T), (0, 0), (0, 0)), mode='edge')

        # 정규화
        if self.normalize_pose:
            poses = self.pose_normalizer.normalize_poses(poses)

        # 오디오 특징 (파일명에서 추출)
        audio_name = motion_file.stem + '.wav'
        audio_path = self.audio_dir / audio_name

        if audio_path.exists():
            audio_features_dict = self.audio_extractor.extract_frame_level_features(
                str(audio_path), num_frames=self.seq_length, fps=self.fps
            )
            audio_features = audio_features_dict['audio_features']
        else:
            # 오디오 없으면 제로 특징
            audio_features = torch.zeros(self.seq_length, 36)

        # 장르 추출 (파일명에서)
        genre = motion_file.stem[:3]  # 예: gBR
        style_text = f"{genre} dance"

        poses = torch.from_numpy(poses).float()

        return {
            'audio_features': audio_features,
            'style_text': style_text,
            'poses': poses,
            'sequence_name': motion_file.stem
        }


def create_dummy_dataset(output_dir: str, num_samples: int = 100):
    """
    테스트용 더미 데이터셋 생성

    Args:
        output_dir: 출력 디렉토리
        num_samples: 샘플 수
    """
    import soundfile as sf

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_dir = output_dir / 'audio'
    pose_dir = output_dir / 'poses'
    audio_dir.mkdir(exist_ok=True)
    pose_dir.mkdir(exist_ok=True)

    metadata = {'data': []}

    styles = ['kpop', 'hip-hop', 'ballet', 'jazz', 'pop']

    for i in range(num_samples):
        # 더미 오디오 (3초, 22050 Hz)
        sr = 22050
        duration = 3.0
        t = np.linspace(0, duration, int(sr * duration))
        freq = 440 + np.random.randint(-100, 100)
        audio = np.sin(2 * np.pi * freq * t) * 0.5

        audio_filename = f'audio_{i:04d}.wav'
        audio_path = audio_dir / audio_filename
        sf.write(audio_path, audio, sr)

        # 더미 포즈 (90 프레임, 17 관절, 2D)
        poses = np.random.randn(90, 17, 2) * 0.5
        pose_filename = f'pose_{i:04d}.npy'
        pose_path = pose_dir / pose_filename
        np.save(pose_path, poses)

        # 메타데이터
        metadata['data'].append({
            'audio_path': f'audio/{audio_filename}',
            'pose_path': f'poses/{pose_filename}',
            'style': np.random.choice(styles),
            'duration': duration
        })

    # 메타데이터 저장
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Created dummy dataset with {num_samples} samples in {output_dir}")


def test_dataset():
    """데이터셋 테스트"""
    print("Dataset 테스트")
    print("=" * 50)

    # 더미 데이터셋 생성
    dummy_dir = '/tmp/dummy_dance_dataset'
    create_dummy_dataset(dummy_dir, num_samples=10)

    # 데이터셋 로드
    dataset = DanceMotionDataset(
        data_dir=dummy_dir,
        seq_length=64,
        fps=30.0,
        cache_audio_features=False
    )

    print(f"\nDataset size: {len(dataset)}")

    # 첫 번째 샘플
    sample = dataset[0]
    print(f"\nSample keys: {sample.keys()}")
    print(f"Audio features shape: {sample['audio_features'].shape}")
    print(f"Poses shape: {sample['poses'].shape}")
    print(f"Style text: {sample['style_text']}")

    # DataLoader 테스트
    from torch.utils.data import DataLoader

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

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn
    )

    batch = next(iter(dataloader))
    print(f"\nBatch:")
    print(f"  Audio features: {batch['audio_features'].shape}")
    print(f"  Poses: {batch['poses'].shape}")
    print(f"  Styles: {batch['style_texts']}")

    print("\n테스트 완료!")


if __name__ == "__main__":
    test_dataset()
