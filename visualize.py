"""
시각화 스크립트
생성된 포즈 시퀀스를 비디오로 렌더링합니다.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from pathlib import Path
import json

from utils import load_pose_sequence, COCO_SKELETON


class PoseVisualizer:
    """2D 포즈 시각화 클래스"""

    def __init__(
        self,
        figsize: tuple = (8, 8),
        skeleton: list = None
    ):
        """
        Args:
            figsize: 그림 크기
            skeleton: 스켈레톤 연결 리스트
        """
        self.figsize = figsize
        self.skeleton = skeleton or COCO_SKELETON

    def draw_pose(
        self,
        ax,
        pose: np.ndarray,
        color: str = 'blue',
        linewidth: float = 2.0,
        markersize: float = 5.0
    ):
        """
        하나의 포즈를 그립니다.

        Args:
            ax: Matplotlib axis
            pose: (num_joints, 2) 관절 좌표
            color: 색상
            linewidth: 선 굵기
            markersize: 마커 크기
        """
        # 관절 그리기
        ax.scatter(pose[:, 0], pose[:, 1], c=color, s=markersize**2, zorder=2)

        # 본 그리기
        for joint1, joint2 in self.skeleton:
            if joint1 < len(pose) and joint2 < len(pose):
                x_values = [pose[joint1, 0], pose[joint2, 0]]
                y_values = [pose[joint1, 1], pose[joint2, 1]]
                ax.plot(x_values, y_values, c=color, linewidth=linewidth, zorder=1)

    def create_video(
        self,
        poses: np.ndarray,
        output_path: str,
        fps: float = 30.0,
        xlim: tuple = (-1.5, 1.5),
        ylim: tuple = (-1.5, 1.5),
        title: str = "Generated Dance"
    ):
        """
        포즈 시퀀스를 비디오로 저장합니다.

        Args:
            poses: (T, num_joints, 2)
            output_path: 출력 비디오 경로
            fps: FPS
            xlim, ylim: 축 범위
            title: 제목
        """
        T = poses.shape[0]

        fig, ax = plt.subplots(figsize=self.figsize)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect('equal')
        ax.invert_yaxis()  # Y축 반전 (이미지 좌표계)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        def update(frame):
            ax.clear()
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_aspect('equal')
            ax.invert_yaxis()
            ax.set_title(f"{title} - Frame {frame}/{T}")
            ax.grid(True, alpha=0.3)

            self.draw_pose(ax, poses[frame], color='blue', linewidth=2.5, markersize=6)

            return ax,

        anim = animation.FuncAnimation(
            fig,
            update,
            frames=T,
            interval=1000/fps,
            blit=False
        )

        # 저장
        print(f"Saving video to {output_path}...")
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=fps, bitrate=1800)
        anim.save(output_path, writer=writer)
        plt.close()

        print("Video saved!")

    def create_gif(
        self,
        poses: np.ndarray,
        output_path: str,
        fps: float = 30.0,
        xlim: tuple = (-1.5, 1.5),
        ylim: tuple = (-1.5, 1.5),
        title: str = "Generated Dance"
    ):
        """
        포즈 시퀀스를 GIF로 저장합니다.

        Args:
            poses: (T, num_joints, 2)
            output_path: 출력 GIF 경로
            fps: FPS
            xlim, ylim: 축 범위
            title: 제목
        """
        T = poses.shape[0]

        fig, ax = plt.subplots(figsize=self.figsize)

        def update(frame):
            ax.clear()
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_aspect('equal')
            ax.invert_yaxis()
            ax.set_title(f"{title} - Frame {frame}/{T}")
            ax.grid(True, alpha=0.3)

            self.draw_pose(ax, poses[frame], color='blue', linewidth=2.5, markersize=6)

            return ax,

        anim = animation.FuncAnimation(
            fig,
            update,
            frames=T,
            interval=1000/fps,
            blit=False
        )

        # 저장
        print(f"Saving GIF to {output_path}...")
        anim.save(output_path, writer='pillow', fps=fps)
        plt.close()

        print("GIF saved!")

    def plot_comparison(
        self,
        poses1: np.ndarray,
        poses2: np.ndarray,
        frame_idx: int,
        labels: tuple = ("Pose 1", "Pose 2"),
        output_path: str = None
    ):
        """
        두 포즈를 나란히 비교합니다.

        Args:
            poses1, poses2: (T, num_joints, 2)
            frame_idx: 비교할 프레임 인덱스
            labels: 라벨
            output_path: 저장 경로 (None이면 표시)
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        for ax, pose, label in zip(axes, [poses1[frame_idx], poses2[frame_idx]], labels):
            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(-1.5, 1.5)
            ax.set_aspect('equal')
            ax.invert_yaxis()
            ax.set_title(f"{label} - Frame {frame_idx}")
            ax.grid(True, alpha=0.3)

            self.draw_pose(ax, pose, color='blue', linewidth=2.5, markersize=6)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Comparison saved to {output_path}")
        else:
            plt.show()

        plt.close()


class PoseWithAudioVisualizer(PoseVisualizer):
    """오디오와 함께 포즈를 시각화"""

    def create_video_with_audio(
        self,
        poses: np.ndarray,
        audio_path: str,
        output_path: str,
        fps: float = 30.0,
        **kwargs
    ):
        """
        오디오와 함께 비디오 생성

        Args:
            poses: (T, num_joints, 2)
            audio_path: 오디오 파일 경로
            output_path: 출력 경로
            fps: FPS
        """
        import subprocess
        import tempfile

        # 먼저 비디오만 생성 (임시 파일)
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            tmp_video_path = tmp.name

        self.create_video(poses, tmp_video_path, fps=fps, **kwargs)

        # ffmpeg로 오디오 합성
        print("Merging audio...")
        cmd = [
            'ffmpeg',
            '-i', tmp_video_path,
            '-i', audio_path,
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-strict', 'experimental',
            '-shortest',
            '-y',
            output_path
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"Video with audio saved to {output_path}")
        except subprocess.CalledProcessError as e:
            print(f"Error merging audio: {e.stderr.decode()}")
            print(f"Video without audio saved to {tmp_video_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize pose sequence')

    parser.add_argument('--pose_sequence', type=str, required=True,
                        help='Pose sequence JSON file')
    parser.add_argument('--output_video', type=str, default='output/dance_video.mp4',
                        help='Output video path')
    parser.add_argument('--output_gif', type=str, default=None,
                        help='Output GIF path (optional)')
    parser.add_argument('--audio_path', type=str, default=None,
                        help='Audio file to merge (optional)')
    parser.add_argument('--fps', type=float, default=30.0,
                        help='FPS')
    parser.add_argument('--xlim', type=float, nargs=2, default=[-1.5, 1.5],
                        help='X-axis limits')
    parser.add_argument('--ylim', type=float, nargs=2, default=[-1.5, 1.5],
                        help='Y-axis limits')
    parser.add_argument('--title', type=str, default='Generated Dance',
                        help='Video title')

    args = parser.parse_args()

    # 출력 디렉토리 생성
    output_path = Path(args.output_video)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 포즈 로드
    print(f"Loading pose sequence from {args.pose_sequence}")
    poses, metadata = load_pose_sequence(args.pose_sequence)
    print(f"Loaded {poses.shape[0]} frames, {poses.shape[1]} joints")

    # 시각화
    if args.audio_path:
        visualizer = PoseWithAudioVisualizer()
        visualizer.create_video_with_audio(
            poses,
            args.audio_path,
            str(output_path),
            fps=args.fps,
            xlim=tuple(args.xlim),
            ylim=tuple(args.ylim),
            title=args.title
        )
    else:
        visualizer = PoseVisualizer()
        visualizer.create_video(
            poses,
            str(output_path),
            fps=args.fps,
            xlim=tuple(args.xlim),
            ylim=tuple(args.ylim),
            title=args.title
        )

    # GIF 생성 (선택)
    if args.output_gif:
        gif_path = Path(args.output_gif)
        gif_path.parent.mkdir(parents=True, exist_ok=True)
        visualizer.create_gif(
            poses,
            str(gif_path),
            fps=args.fps,
            xlim=tuple(args.xlim),
            ylim=tuple(args.ylim),
            title=args.title
        )

    print("Done!")


if __name__ == "__main__":
    main()
