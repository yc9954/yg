"""
평가 메트릭
Beat Align Score, FVD, Diversity 등을 계산합니다.
"""

import numpy as np
import torch
import torch.nn as nn
from scipy import linalg
from typing import List, Tuple
import librosa


class BeatAlignScore:
    """
    Beat Alignment Score (BAS)
    생성된 댄스와 음악 박자 간 동기화 정도를 측정합니다.
    """

    def __init__(self, fps: float = 30.0):
        self.fps = fps

    def extract_motion_beats(
        self,
        poses: np.ndarray,
        threshold: float = 0.5
    ) -> np.ndarray:
        """
        동작 변화점(motion beat)을 추출합니다.

        Args:
            poses: (T, num_joints, 2 or 3)
            threshold: 동작 변화 감지 임계값

        Returns:
            motion_beat_times: 동작 박자 시간 배열 (초)
        """
        T = poses.shape[0]

        # 프레임 간 전체 동작 변화량 계산
        motion_diff = np.zeros(T)
        motion_diff[1:] = np.linalg.norm(
            poses[1:] - poses[:-1],
            axis=(1, 2)
        )

        # 피크 감지
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(motion_diff, height=threshold)

        # 시간으로 변환
        motion_beat_times = peaks / self.fps

        return motion_beat_times

    def compute_bas(
        self,
        poses: np.ndarray,
        music_beat_times: np.ndarray
    ) -> float:
        """
        Beat Alignment Score 계산

        Args:
            poses: (T, num_joints, D)
            music_beat_times: 음악 박자 시간 배열 (초)

        Returns:
            BAS: 평균 박자 정렬 거리 (낮을수록 좋음)
        """
        # 동작 박자 추출
        motion_beats = self.extract_motion_beats(poses)

        if len(motion_beats) == 0 or len(music_beat_times) == 0:
            return float('inf')

        # 각 동작 박자에 대해 가장 가까운 음악 박자까지의 거리
        distances = []
        for motion_beat in motion_beats:
            dist = np.abs(music_beat_times - motion_beat).min()
            distances.append(dist)

        # 평균 거리
        bas = np.mean(distances)

        return bas


class FrechetDistance:
    """
    Frechet Distance (FD)
    두 분포 간의 거리를 측정합니다 (FVD와 유사한 개념).
    """

    @staticmethod
    def compute_statistics(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        특징 벡터의 평균과 공분산 계산

        Args:
            features: (N, D) - N개 샘플, D차원 특징

        Returns:
            mu: (D,) 평균
            sigma: (D, D) 공분산
        """
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        return mu, sigma

    @staticmethod
    def compute_frechet_distance(
        mu1: np.ndarray,
        sigma1: np.ndarray,
        mu2: np.ndarray,
        sigma2: np.ndarray,
        eps: float = 1e-6
    ) -> float:
        """
        Frechet Distance 계산

        Args:
            mu1, sigma1: 첫 번째 분포의 평균과 공분산
            mu2, sigma2: 두 번째 분포의 평균과 공분산
            eps: 수치 안정성을 위한 작은 값

        Returns:
            FD 값
        """
        # 평균 차이
        diff = mu1 - mu2

        # 공분산 곱의 제곱근
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

        # 수치 안정성
        if not np.isfinite(covmean).all():
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # 허수 부분 제거 (수치 오차)
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fd = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)

        return fd

    def compute_fd_for_poses(
        self,
        real_poses: np.ndarray,
        fake_poses: np.ndarray
    ) -> float:
        """
        포즈 시퀀스에 대한 FD 계산

        Args:
            real_poses: (N, T, J, D) - 실제 포즈
            fake_poses: (M, T, J, D) - 생성된 포즈

        Returns:
            FD 값
        """
        # Flatten
        N, T, J, D = real_poses.shape
        M = fake_poses.shape[0]

        real_features = real_poses.reshape(N, -1)  # (N, T*J*D)
        fake_features = fake_poses.reshape(M, -1)  # (M, T*J*D)

        # 통계 계산
        mu_real, sigma_real = self.compute_statistics(real_features)
        mu_fake, sigma_fake = self.compute_statistics(fake_features)

        # FD 계산
        fd = self.compute_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)

        return fd


class DiversityMetric:
    """
    생성 다양성 측정
    """

    @staticmethod
    def compute_diversity(poses: np.ndarray) -> float:
        """
        포즈 시퀀스의 다양성 계산

        Args:
            poses: (N, T, J, D) - N개의 시퀀스

        Returns:
            Diversity 값 (높을수록 다양함)
        """
        N, T, J, D = poses.shape

        # Flatten
        features = poses.reshape(N, -1)  # (N, T*J*D)

        # 쌍별 거리 계산
        dists = []
        for i in range(N):
            for j in range(i + 1, N):
                dist = np.linalg.norm(features[i] - features[j])
                dists.append(dist)

        # 평균 거리
        diversity = np.mean(dists) if len(dists) > 0 else 0.0

        return diversity

    @staticmethod
    def compute_temporal_diversity(poses: np.ndarray) -> float:
        """
        시간축 다양성 (각 시퀀스 내 변화량)

        Args:
            poses: (N, T, J, D)

        Returns:
            Temporal diversity
        """
        # 프레임 간 변화량
        diffs = poses[:, 1:] - poses[:, :-1]  # (N, T-1, J, D)
        temporal_div = np.linalg.norm(diffs, axis=(2, 3)).mean()

        return temporal_div


class MotionQualityMetrics:
    """
    동작 품질 메트릭
    """

    @staticmethod
    def compute_smoothness(poses: np.ndarray) -> float:
        """
        동작 부드러움 (가속도 기반)

        Args:
            poses: (T, J, D)

        Returns:
            Smoothness 값 (낮을수록 부드러움)
        """
        # 속도
        vel = poses[1:] - poses[:-1]  # (T-1, J, D)

        # 가속도
        acc = vel[1:] - vel[:-1]  # (T-2, J, D)

        # 가속도의 평균 크기 (낮을수록 부드러움)
        smoothness = np.linalg.norm(acc, axis=(1, 2)).mean()

        return smoothness

    @staticmethod
    def compute_foot_skating(
        poses: np.ndarray,
        ankle_indices: List[int] = [15, 16],  # left_ankle, right_ankle
        threshold: float = 0.01
    ) -> float:
        """
        발 미끄러짐 (foot skating) 측정

        Args:
            poses: (T, J, D)
            ankle_indices: 발목 관절 인덱스
            threshold: 발이 지면에 있다고 간주할 높이 임계값

        Returns:
            Foot skating 비율 (낮을수록 좋음)
        """
        skating_count = 0
        total_count = 0

        for ankle_idx in ankle_indices:
            ankle_pos = poses[:, ankle_idx, :]  # (T, D)

            # 높이 (y 좌표 또는 z 좌표)
            # 2D의 경우 y, 3D의 경우 z
            if ankle_pos.shape[1] == 2:
                height = ankle_pos[:, 1]
            else:
                height = ankle_pos[:, 2]

            # 발이 지면에 있는 프레임
            on_ground = height < threshold

            # 지면에 있으면서 움직이는 경우 (skating)
            for t in range(1, len(poses)):
                if on_ground[t] and on_ground[t-1]:
                    # 발 위치 변화
                    movement = np.linalg.norm(ankle_pos[t] - ankle_pos[t-1])
                    if movement > 0.01:  # 임계값 이상 움직임
                        skating_count += 1
                    total_count += 1

        skating_ratio = skating_count / max(total_count, 1)

        return skating_ratio


def evaluate_generated_dances(
    real_poses: np.ndarray,
    generated_poses: np.ndarray,
    music_beat_times: np.ndarray,
    fps: float = 30.0
) -> dict:
    """
    생성된 댄스에 대한 종합 평가

    Args:
        real_poses: (N, T, J, D) - 실제 댄스
        generated_poses: (M, T, J, D) - 생성된 댄스
        music_beat_times: 음악 박자 시간 배열 (초)
        fps: FPS

    Returns:
        평가 메트릭 딕셔너리
    """
    metrics = {}

    # 1. Beat Alignment Score
    bas_calculator = BeatAlignScore(fps)
    bas_scores = []
    for i in range(generated_poses.shape[0]):
        bas = bas_calculator.compute_bas(generated_poses[i], music_beat_times)
        bas_scores.append(bas)
    metrics['beat_align_score'] = np.mean(bas_scores)

    # 2. Frechet Distance
    fd_calculator = FrechetDistance()
    metrics['frechet_distance'] = fd_calculator.compute_fd_for_poses(
        real_poses, generated_poses
    )

    # 3. Diversity
    metrics['diversity'] = DiversityMetric.compute_diversity(generated_poses)
    metrics['temporal_diversity'] = DiversityMetric.compute_temporal_diversity(generated_poses)

    # 4. Motion Quality
    smoothness_scores = []
    skating_scores = []
    for i in range(generated_poses.shape[0]):
        smoothness_scores.append(MotionQualityMetrics.compute_smoothness(generated_poses[i]))
        skating_scores.append(MotionQualityMetrics.compute_foot_skating(generated_poses[i]))

    metrics['smoothness'] = np.mean(smoothness_scores)
    metrics['foot_skating'] = np.mean(skating_scores)

    return metrics


def test_metrics():
    """테스트 함수"""
    print("Evaluation Metrics 테스트")
    print("=" * 50)

    # 더미 데이터
    N, M, T, J, D = 10, 10, 100, 17, 2
    real_poses = np.random.randn(N, T, J, D) * 0.5
    generated_poses = np.random.randn(M, T, J, D) * 0.5

    # 음악 박자 (예: 3초, 120 BPM -> 2Hz -> 6 beats)
    music_beat_times = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])

    # 평가
    metrics = evaluate_generated_dances(
        real_poses,
        generated_poses,
        music_beat_times,
        fps=30.0
    )

    print("\n평가 결과:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    print("\n테스트 완료!")


if __name__ == "__main__":
    test_metrics()
