"""
포즈 전처리 모듈
2D/3D 스켈레톤 데이터를 정규화하고 변환합니다.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple, List
import json


# COCO 17 키포인트 정의
COCO_KEYPOINTS = [
    'nose',           # 0
    'left_eye',       # 1
    'right_eye',      # 2
    'left_ear',       # 3
    'right_ear',      # 4
    'left_shoulder',  # 5
    'right_shoulder', # 6
    'left_elbow',     # 7
    'right_elbow',    # 8
    'left_wrist',     # 9
    'right_wrist',    # 10
    'left_hip',       # 11
    'right_hip',      # 12
    'left_knee',      # 13
    'right_knee',     # 14
    'left_ankle',     # 15
    'right_ankle'     # 16
]

# 스켈레톤 연결 (본)
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # 얼굴
    (0, 5), (0, 6),  # 어깨로
    (5, 7), (7, 9),  # 왼팔
    (6, 8), (8, 10),  # 오른팔
    (5, 11), (6, 12),  # 몸통
    (11, 12),  # 엉덩이
    (11, 13), (13, 15),  # 왼다리
    (12, 14), (14, 16)   # 오른다리
]


class PoseNormalizer:
    """포즈 데이터 정규화 클래스"""

    def __init__(self, num_joints: int = 17):
        """
        Args:
            num_joints: 관절 개수
        """
        self.num_joints = num_joints

    def normalize_scale(
        self,
        poses: np.ndarray,
        reference_bone: Tuple[int, int] = (5, 11)  # left_shoulder to left_hip
    ) -> np.ndarray:
        """
        스켈레톤의 스케일을 정규화합니다.

        Args:
            poses: shape (T, num_joints, 2 or 3)
            reference_bone: 참조 본 (joint_idx1, joint_idx2)

        Returns:
            정규화된 포즈
        """
        # 참조 본의 길이 계산
        bone_vectors = poses[:, reference_bone[1], :] - poses[:, reference_bone[0], :]
        bone_lengths = np.linalg.norm(bone_vectors, axis=1, keepdims=True)  # (T, 1)

        # 평균 본 길이로 정규화
        mean_bone_length = bone_lengths.mean()
        if mean_bone_length > 0:
            scale_factor = 1.0 / mean_bone_length
            normalized_poses = poses * scale_factor
        else:
            normalized_poses = poses

        return normalized_poses

    def center_poses(
        self,
        poses: np.ndarray,
        root_joint: int = 0  # nose
    ) -> np.ndarray:
        """
        루트 관절을 원점으로 이동합니다.

        Args:
            poses: shape (T, num_joints, D)
            root_joint: 루트 관절 인덱스

        Returns:
            중심화된 포즈
        """
        root_positions = poses[:, root_joint:root_joint+1, :]  # (T, 1, D)
        centered_poses = poses - root_positions
        return centered_poses

    def normalize_poses(
        self,
        poses: np.ndarray,
        center: bool = True,
        scale: bool = True,
        root_joint: int = 0
    ) -> np.ndarray:
        """
        전체 정규화 파이프라인

        Args:
            poses: shape (T, num_joints, D)
            center: 중심화 여부
            scale: 스케일 정규화 여부
            root_joint: 루트 관절

        Returns:
            정규화된 포즈
        """
        normalized = poses.copy()

        if center:
            normalized = self.center_poses(normalized, root_joint)

        if scale:
            normalized = self.normalize_scale(normalized)

        return normalized

    def to_relative_coordinates(
        self,
        poses: np.ndarray,
        parent_indices: Optional[List[int]] = None
    ) -> np.ndarray:
        """
        절대 좌표를 부모 관절 기준 상대 좌표로 변환합니다.

        Args:
            poses: shape (T, num_joints, D)
            parent_indices: 각 관절의 부모 인덱스 리스트

        Returns:
            상대 좌표 포즈
        """
        if parent_indices is None:
            # COCO 기본 부모 관계
            parent_indices = [
                -1,  # 0: nose (root)
                0, 0, 1, 2,  # 1-4: eyes, ears
                0, 0,  # 5-6: shoulders
                5, 6,  # 7-8: elbows
                7, 8,  # 9-10: wrists
                5, 6,  # 11-12: hips
                11, 12,  # 13-14: knees
                13, 14   # 15-16: ankles
            ]

        relative_poses = poses.copy()
        for joint_idx in range(1, self.num_joints):
            parent_idx = parent_indices[joint_idx]
            if parent_idx >= 0:
                relative_poses[:, joint_idx] = (
                    poses[:, joint_idx] - poses[:, parent_idx]
                )

        return relative_poses


class PoseAugmenter:
    """포즈 데이터 증강 클래스"""

    def __init__(self):
        pass

    def random_rotation(
        self,
        poses: np.ndarray,
        max_angle: float = 15.0
    ) -> np.ndarray:
        """
        2D 포즈를 랜덤 회전합니다.

        Args:
            poses: shape (T, num_joints, 2)
            max_angle: 최대 회전 각도 (도)

        Returns:
            회전된 포즈
        """
        angle = np.random.uniform(-max_angle, max_angle)
        theta = np.radians(angle)

        # 2D 회전 행렬
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        rotation_matrix = np.array([
            [cos_theta, -sin_theta],
            [sin_theta, cos_theta]
        ])

        # 회전 적용
        T, J, D = poses.shape
        rotated = poses.reshape(T * J, D) @ rotation_matrix.T
        rotated = rotated.reshape(T, J, D)

        return rotated

    def random_scale(
        self,
        poses: np.ndarray,
        scale_range: Tuple[float, float] = (0.9, 1.1)
    ) -> np.ndarray:
        """
        랜덤 스케일링

        Args:
            poses: shape (T, num_joints, D)
            scale_range: 스케일 범위 (min, max)

        Returns:
            스케일된 포즈
        """
        scale = np.random.uniform(*scale_range)
        return poses * scale

    def add_noise(
        self,
        poses: np.ndarray,
        noise_std: float = 0.01
    ) -> np.ndarray:
        """
        가우시안 노이즈 추가

        Args:
            poses: shape (T, num_joints, D)
            noise_std: 노이즈 표준편차

        Returns:
            노이즈가 추가된 포즈
        """
        noise = np.random.normal(0, noise_std, poses.shape)
        return poses + noise

    def temporal_crop(
        self,
        poses: np.ndarray,
        crop_length: int
    ) -> np.ndarray:
        """
        시간축에서 랜덤 크롭

        Args:
            poses: shape (T, num_joints, D)
            crop_length: 크롭 길이

        Returns:
            크롭된 포즈
        """
        T = poses.shape[0]
        if T <= crop_length:
            return poses

        start_idx = np.random.randint(0, T - crop_length + 1)
        return poses[start_idx:start_idx + crop_length]


def save_pose_sequence(
    poses: np.ndarray,
    output_path: str,
    fps: float = 30.0,
    metadata: Optional[dict] = None
):
    """
    포즈 시퀀스를 JSON 파일로 저장합니다.

    Args:
        poses: shape (T, num_joints, 2 or 3)
        output_path: 출력 파일 경로
        fps: FPS
        metadata: 추가 메타데이터
    """
    T, num_joints, dim = poses.shape

    data = {
        'num_frames': T,
        'num_joints': num_joints,
        'dimension': dim,
        'fps': fps,
        'keypoints': COCO_KEYPOINTS[:num_joints],
        'skeleton': COCO_SKELETON,
        'poses': poses.tolist(),
    }

    if metadata:
        data['metadata'] = metadata

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


def load_pose_sequence(input_path: str) -> Tuple[np.ndarray, dict]:
    """
    JSON 파일에서 포즈 시퀀스를 로드합니다.

    Args:
        input_path: 입력 파일 경로

    Returns:
        poses: shape (T, num_joints, D)
        metadata: 메타데이터 딕셔너리
    """
    with open(input_path, 'r') as f:
        data = json.load(f)

    poses = np.array(data['poses'])

    metadata = {
        'num_frames': data['num_frames'],
        'num_joints': data['num_joints'],
        'dimension': data['dimension'],
        'fps': data.get('fps', 30.0),
        'keypoints': data.get('keypoints', COCO_KEYPOINTS),
        'skeleton': data.get('skeleton', COCO_SKELETON)
    }

    if 'metadata' in data:
        metadata.update(data['metadata'])

    return poses, metadata


def compute_bone_lengths(poses: np.ndarray, skeleton: List[Tuple[int, int]]) -> np.ndarray:
    """
    각 본의 길이를 계산합니다.

    Args:
        poses: shape (T, num_joints, D)
        skeleton: 본 연결 리스트

    Returns:
        shape (T, num_bones) 본 길이 배열
    """
    bone_lengths = []
    for joint1, joint2 in skeleton:
        vectors = poses[:, joint2] - poses[:, joint1]
        lengths = np.linalg.norm(vectors, axis=1)
        bone_lengths.append(lengths)

    return np.stack(bone_lengths, axis=1)  # (T, num_bones)


def test_pose_processing():
    """테스트 함수"""
    print("Pose Processing 테스트")
    print("=" * 50)

    # 랜덤 포즈 생성
    T, num_joints = 100, 17
    poses = np.random.randn(T, num_joints, 2)

    # 정규화
    normalizer = PoseNormalizer(num_joints)
    normalized = normalizer.normalize_poses(poses)
    print(f"Original pose shape: {poses.shape}")
    print(f"Normalized pose shape: {normalized.shape}")
    print(f"Root joint mean: {normalized[:, 0].mean(axis=0)}")  # 중심화 확인

    # 증강
    augmenter = PoseAugmenter()
    rotated = augmenter.random_rotation(poses)
    scaled = augmenter.random_scale(poses)
    noisy = augmenter.add_noise(poses)
    print(f"\nAugmented poses:")
    print(f"  Rotated: {rotated.shape}")
    print(f"  Scaled: {scaled.shape}")
    print(f"  Noisy: {noisy.shape}")

    # 저장 및 로드
    test_path = "/tmp/test_pose.json"
    save_pose_sequence(poses, test_path, fps=30.0, metadata={'test': True})
    loaded_poses, metadata = load_pose_sequence(test_path)
    print(f"\nSaved and loaded:")
    print(f"  Shape: {loaded_poses.shape}")
    print(f"  FPS: {metadata['fps']}")
    print(f"  Metadata: {metadata.get('test')}")

    # 본 길이 계산
    bone_lengths = compute_bone_lengths(poses, COCO_SKELETON)
    print(f"\nBone lengths shape: {bone_lengths.shape}")
    print(f"Mean bone length: {bone_lengths.mean():.4f}")

    print("\n테스트 완료!")


if __name__ == "__main__":
    test_pose_processing()
