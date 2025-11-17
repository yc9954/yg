# AI Dance Generation from Music and Text

음악 파일과 자연어 댄스 스타일 설명을 입력받아 2D 인체 스켈레톤 좌표 시퀀스를 생성하는 AI 모델입니다.

## 프로젝트 개요

- **입력**: 음악 파일 (mp3/wav) + 자연어 댄스 스타일 설명 (예: "걸크러시한 K-pop 댄스")
- **출력**: 2D 인체 스켈레톤 좌표 시퀀스 (JSON 또는 NumPy 배열)
- **모델**: Transformer 기반 (VQ-VAE + GPT-like 구조) + Diffusion 기반 생성기
- **참고 논문**: X-Dancer (ICCV 2025), DGSDP, GCDance

## 주요 기능

### 1. 입력 처리
- **음악 분석**: librosa를 사용한 BPM, Beat, Onset, Chroma 특징 추출
- **텍스트 임베딩**: BERT/CLIP를 사용한 스타일 텍스트 벡터화
- **실시간 처리**: YouTube 비디오에서 실시간 오디오 추출 지원

### 2. 모델 아키텍처
- **VQ-VAE 포즈 토크나이저**: 2D 스켈레톤을 토큰으로 변환
- **Transformer 디코더**: 음악과 스타일을 조건으로 시퀀스 생성
- **Diffusion 기반 고해상도 변환** (선택적)

### 3. 학습 데이터셋
- AIST++ (10개 장르, 1408 시퀀스, 5.2시간)
- HumanML3D (14,616개 동작 클립, 44,970개 텍스트 설명)
- BABEL (43시간 mocap 데이터)

### 4. 평가 지표
- **Beat Align Score (BAS)**: 음악-동작 리듬 정렬도
- **Frechet Video Distance (FVD)**: 생성 품질
- **Diversity (DIV)**: 생성 다양성
- **Motion Plausibility**: 자연스러움 평가

## 디렉토리 구조

```
yg/
├── data/                      # 데이터셋 디렉토리
│   ├── download_aist.py       # AIST++ 다운로드 스크립트
│   └── preprocess.py          # 데이터 전처리
├── models/                    # 모델 아키텍처
│   ├── vqvae.py              # VQ-VAE 포즈 토크나이저
│   ├── transformer.py         # Transformer 댄스 생성기
│   └── diffusion.py          # Diffusion 모델 (선택적)
├── utils/                     # 유틸리티 함수
│   ├── audio_features.py     # 음악 특징 추출
│   ├── text_embedding.py     # 텍스트 임베딩
│   └── pose_processing.py    # 포즈 전처리
├── datasets/                  # 데이터셋 클래스
│   └── dance_dataset.py      # PyTorch 데이터셋
├── train.py                   # 학습 스크립트
├── evaluate.py                # 평가 스크립트
├── inference.py               # 추론 스크립트
├── visualize.py              # 시각화 도구
├── notebooks/                 # Jupyter/Colab 노트북
│   └── colab_demo.ipynb      # Colab 데모
├── requirements.txt           # 의존성
└── README.md                 # 본 파일
```

## 설치 방법

### 로컬 환경
```bash
pip install -r requirements.txt
```

### Google Colab
```python
!pip install torch torchvision torchaudio librosa transformers mediapipe tqdm einops
```

## 사용 방법

### 1. 데이터 준비
```bash
# AIST++ 데이터셋 다운로드
python data/download_aist.py --output_dir ./data/aist_plusplus

# 데이터 전처리
python data/preprocess.py --dataset aist --input_dir ./data/aist_plusplus --output_dir ./data/processed
```

### 2. 학습
```bash
python train.py \
  --dataset_path ./data/processed \
  --batch_size 16 \
  --epochs 100 \
  --lr 1e-4 \
  --output_dir ./checkpoints
```

### 3. 추론
```bash
python inference.py \
  --checkpoint ./checkpoints/best_model.pth \
  --audio_path ./samples/music.mp3 \
  --style_text "강렬한 K-pop 댄스" \
  --output_path ./output/dance.json
```

### 4. 시각화
```bash
python visualize.py \
  --pose_sequence ./output/dance.json \
  --audio_path ./samples/music.mp3 \
  --output_video ./output/dance_video.mp4
```

## Google Colab에서 실행

Google Colab에서 전체 파이프라인을 실행하려면:
1. `notebooks/colab_demo.ipynb` 열기
2. GPU 런타임 설정 (T4/P100/V100 권장)
3. 셀 순서대로 실행

## 모델 아키텍처 상세

### Transformer 기반 생성기
- **입력**: 음악 특징 (tempo, chroma, onset) + 스타일 임베딩
- **출력**: 2D 스켈레톤 좌표 시퀀스 (T, 17, 2)
- **구조**:
  - Audio Encoder: 오디오 특징 → hidden representation
  - Style Encoder: BERT 텍스트 임베딩 → style vector
  - Transformer Decoder: 조건부 시퀀스 생성
  - Pose Decoder: hidden → 관절 좌표

### VQ-VAE 토크나이저 (선택적)
- 연속 포즈 공간을 이산 토큰으로 양자화
- 부위별 코드북 (상체, 하체, 손 등)
- GPT-like 모델과 결합 가능

## 하이퍼파라미터

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| batch_size | 16 | 배치 크기 |
| lr | 1e-4 | 학습률 |
| hidden_size | 512 | Transformer 차원 |
| num_heads | 8 | 어텐션 헤드 수 |
| num_layers | 6 | Transformer 레이어 수 |
| num_joints | 17 | 관절 개수 (COCO 포즈) |
| seq_length | 64 | 시퀀스 길이 |

## 평가 결과 예시

| 메트릭 | 점수 |
|--------|------|
| Beat Align Score (BAS) | 0.082 |
| FVD | 127.3 |
| Diversity | 8.42 |

## 참고 자료

### 논문
- [X-Dancer (ICCV 2025)](https://openaccess.thecvf.com/content/ICCV2025/papers/Chen_X-Dancer_Expressive_Music_to_Human_Dance_Video_Generation_ICCV_2025_paper.pdf)
- [DGSDP (2024)](https://arxiv.org/html/2406.07871v1)
- [AIST++ (2021)](https://arxiv.org/abs/2101.08779)
- [HumanML3D (CVPR 2022)](https://openaccess.thecvf.com/content/CVPR2022/papers/Guo_Generating_Diverse_and_Natural_3D_Human_Motions_From_Text_CVPR_2022_paper.pdf)

### 데이터셋
- [AIST++](https://google.github.io/aistplusplus_dataset/)
- [HumanML3D](https://github.com/EricGuo5513/HumanML3D)
- [BABEL](https://babel.is.tue.mpg.de/)

## 라이센스

MIT License

## 기여

이슈 및 PR 환영합니다!

## 연락처

프로젝트 관련 문의: [이메일 주소]
