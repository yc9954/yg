"""
음악 특징 추출 모듈
librosa를 사용하여 BPM, Beat, Onset, Chroma 등을 추출합니다.
"""

import librosa
import numpy as np
import torch
from typing import Dict, Tuple, Optional


class AudioFeatureExtractor:
    """음악 파일에서 다양한 특징을 추출하는 클래스"""

    def __init__(self, sr: int = 22050, hop_length: int = 512, n_fft: int = 2048):
        """
        Args:
            sr: 샘플링 레이트
            hop_length: STFT hop length
            n_fft: FFT 윈도우 크기
        """
        self.sr = sr
        self.hop_length = hop_length
        self.n_fft = n_fft

    def extract_all_features(
        self,
        audio_path: str,
        duration: Optional[float] = None
    ) -> Dict[str, np.ndarray]:
        """
        오디오 파일에서 모든 특징을 추출합니다.

        Args:
            audio_path: 오디오 파일 경로
            duration: 로드할 오디오 길이 (초), None이면 전체

        Returns:
            특징 딕셔너리:
                - tempo: BPM 값
                - beats: 박자 프레임 인덱스
                - beat_times: 박자 시간 (초)
                - onset_env: 온셋 강도 envelope
                - chroma: 크로마 특징 (12, T)
                - mfcc: MFCC 특징 (20, T)
                - mel_spectrogram: 멜 스펙트로그램 (128, T)
                - rms: RMS 에너지
        """
        # 오디오 로드
        y, sr = librosa.load(audio_path, sr=self.sr, duration=duration)

        # 1. Tempo와 Beat 추출
        tempo, beat_frames = librosa.beat.beat_track(
            y=y, sr=sr, hop_length=self.hop_length
        )
        beat_times = librosa.frames_to_time(
            beat_frames, sr=sr, hop_length=self.hop_length
        )

        # 2. Onset Strength
        onset_env = librosa.onset.onset_strength(
            y=y, sr=sr, hop_length=self.hop_length
        )

        # 3. Chroma Features (12-dimensional pitch class)
        chroma = librosa.feature.chroma_cqt(
            y=y, sr=sr, hop_length=self.hop_length
        )

        # 4. MFCC (Mel-frequency cepstral coefficients)
        mfcc = librosa.feature.mfcc(
            y=y, sr=sr, n_mfcc=20, hop_length=self.hop_length, n_fft=self.n_fft
        )

        # 5. Mel Spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=128, hop_length=self.hop_length, n_fft=self.n_fft
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # 6. RMS Energy
        rms = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]

        # 7. Spectral Contrast
        spectral_contrast = librosa.feature.spectral_contrast(
            y=y, sr=sr, hop_length=self.hop_length, n_fft=self.n_fft
        )

        # 8. Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y, hop_length=self.hop_length)[0]

        return {
            'audio': y,
            'tempo': float(tempo),
            'beat_frames': beat_frames,
            'beat_times': beat_times,
            'onset_env': onset_env,
            'chroma': chroma,
            'mfcc': mfcc,
            'mel_spectrogram': mel_spec_db,
            'rms': rms,
            'spectral_contrast': spectral_contrast,
            'zcr': zcr,
            'sr': sr
        }

    def get_beat_sync_features(
        self,
        features: Dict[str, np.ndarray],
        aggregate: str = 'mean'
    ) -> Dict[str, np.ndarray]:
        """
        박자에 동기화된 특징을 추출합니다.

        Args:
            features: extract_all_features 출력
            aggregate: 집계 방법 ('mean', 'median')

        Returns:
            박자 동기화된 특징 딕셔너리
        """
        beat_frames = features['beat_frames']

        # 박자에 맞춰 특징 집계
        chroma_sync = librosa.util.sync(
            features['chroma'], beat_frames, aggregate=aggregate
        )
        mfcc_sync = librosa.util.sync(
            features['mfcc'], beat_frames, aggregate=aggregate
        )
        onset_sync = librosa.util.sync(
            features['onset_env'].reshape(1, -1), beat_frames, aggregate=aggregate
        )[0]

        return {
            'chroma_sync': chroma_sync,
            'mfcc_sync': mfcc_sync,
            'onset_sync': onset_sync,
            'beat_times': features['beat_times']
        }

    def frame_to_beat_alignment(
        self,
        num_frames: int,
        beat_times: np.ndarray,
        fps: float = 30.0
    ) -> np.ndarray:
        """
        비디오 프레임과 음악 박자 간 정렬 정보를 생성합니다.

        Args:
            num_frames: 비디오 프레임 수
            beat_times: 박자 시간 배열 (초)
            fps: 비디오 FPS

        Returns:
            각 프레임의 가장 가까운 박자까지의 거리 (shape: num_frames)
        """
        frame_times = np.arange(num_frames) / fps
        beat_alignment = np.zeros(num_frames)

        for i, t in enumerate(frame_times):
            # 가장 가까운 박자까지의 시간 거리
            distances = np.abs(beat_times - t)
            beat_alignment[i] = np.min(distances)

        return beat_alignment

    def get_beat_onehot(
        self,
        num_frames: int,
        beat_times: np.ndarray,
        fps: float = 30.0,
        tolerance: float = 0.1
    ) -> np.ndarray:
        """
        각 프레임이 박자인지 나타내는 one-hot 벡터를 생성합니다.

        Args:
            num_frames: 프레임 수
            beat_times: 박자 시간 (초)
            fps: FPS
            tolerance: 박자로 간주할 시간 허용 오차 (초)

        Returns:
            shape (num_frames,), 박자 프레임은 1, 아니면 0
        """
        frame_times = np.arange(num_frames) / fps
        beat_onehot = np.zeros(num_frames)

        for beat_time in beat_times:
            # 허용 오차 내의 프레임 찾기
            mask = np.abs(frame_times - beat_time) < tolerance
            beat_onehot[mask] = 1.0

        return beat_onehot

    def extract_frame_level_features(
        self,
        audio_path: str,
        num_frames: int,
        fps: float = 30.0
    ) -> Dict[str, torch.Tensor]:
        """
        프레임 단위로 정렬된 오디오 특징을 추출합니다.
        댄스 생성 모델의 입력으로 사용됩니다.

        Args:
            audio_path: 오디오 파일 경로
            num_frames: 생성할 프레임 수
            fps: 비디오 FPS

        Returns:
            프레임 단위 특징 텐서 딕셔너리
        """
        # 전체 특징 추출
        duration = num_frames / fps
        features = self.extract_all_features(audio_path, duration=duration)

        # 프레임 단위로 리샘플링
        target_length = num_frames

        # Chroma
        chroma = torch.from_numpy(
            librosa.util.fix_length(features['chroma'], size=target_length, axis=1)
        ).float().T  # (num_frames, 12)

        # MFCC
        mfcc = torch.from_numpy(
            librosa.util.fix_length(features['mfcc'], size=target_length, axis=1)
        ).float().T  # (num_frames, 20)

        # Onset
        onset = torch.from_numpy(
            librosa.util.fix_length(
                features['onset_env'].reshape(1, -1), size=target_length, axis=1
            )
        ).float().T  # (num_frames, 1)

        # RMS
        rms = torch.from_numpy(
            librosa.util.fix_length(
                features['rms'].reshape(1, -1), size=target_length, axis=1
            )
        ).float().T  # (num_frames, 1)

        # Beat alignment
        beat_onehot = self.get_beat_onehot(num_frames, features['beat_times'], fps)
        beat_onehot = torch.from_numpy(beat_onehot).float().unsqueeze(1)  # (num_frames, 1)

        # Tempo (전체 프레임에 동일)
        tempo = torch.full((num_frames, 1), features['tempo'] / 200.0)  # 정규화

        return {
            'chroma': chroma,           # (T, 12)
            'mfcc': mfcc,              # (T, 20)
            'onset': onset,            # (T, 1)
            'rms': rms,                # (T, 1)
            'beat': beat_onehot,       # (T, 1)
            'tempo': tempo,            # (T, 1)
            'audio_features': torch.cat([
                chroma, mfcc, onset, rms, beat_onehot, tempo
            ], dim=1)  # (T, 36) - 모든 특징 결합
        }


def test_audio_extractor():
    """테스트 함수"""
    print("Audio Feature Extractor 테스트")
    print("=" * 50)

    extractor = AudioFeatureExtractor()

    # 예시 오디오 생성 (1초 440Hz 사인파)
    import soundfile as sf
    sr = 22050
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration))
    y = np.sin(2 * np.pi * 440 * t)

    # 임시 파일 저장
    test_path = "/tmp/test_audio.wav"
    sf.write(test_path, y, sr)

    # 특징 추출
    features = extractor.extract_all_features(test_path)
    print(f"Tempo: {features['tempo']:.2f} BPM")
    print(f"Beats: {len(features['beat_frames'])} beats")
    print(f"Chroma shape: {features['chroma'].shape}")
    print(f"MFCC shape: {features['mfcc'].shape}")

    # 프레임 단위 특징
    frame_features = extractor.extract_frame_level_features(test_path, num_frames=90, fps=30)
    print(f"\nFrame-level features shape: {frame_features['audio_features'].shape}")
    print(f"Chroma: {frame_features['chroma'].shape}")
    print(f"Beat onehot: {frame_features['beat'].sum().item()} beats")

    print("\n테스트 완료!")


if __name__ == "__main__":
    test_audio_extractor()
