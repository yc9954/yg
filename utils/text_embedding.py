"""
텍스트 임베딩 모듈
BERT, CLIP, T5 등을 사용하여 댄스 스타일 텍스트를 벡터화합니다.
"""

import torch
import torch.nn as nn
from transformers import (
    BertModel, BertTokenizer,
    CLIPTextModel, CLIPTokenizer,
    T5EncoderModel, T5Tokenizer
)
from typing import List, Union, Optional


class TextEmbedder:
    """텍스트를 임베딩 벡터로 변환하는 클래스"""

    def __init__(
        self,
        model_type: str = 'bert',
        model_name: Optional[str] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Args:
            model_type: 'bert', 'clip', 't5' 중 선택
            model_name: 사전학습 모델 이름 (None이면 기본값 사용)
            device: 'cuda' or 'cpu'
        """
        self.model_type = model_type.lower()
        self.device = device

        # 모델 및 토크나이저 로드
        if self.model_type == 'bert':
            model_name = model_name or 'bert-base-uncased'
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            self.model = BertModel.from_pretrained(model_name)
            self.hidden_size = self.model.config.hidden_size

        elif self.model_type == 'clip':
            model_name = model_name or 'openai/clip-vit-base-patch32'
            self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
            self.model = CLIPTextModel.from_pretrained(model_name)
            self.hidden_size = self.model.config.hidden_size

        elif self.model_type == 't5':
            model_name = model_name or 't5-base'
            self.tokenizer = T5Tokenizer.from_pretrained(model_name)
            self.model = T5EncoderModel.from_pretrained(model_name)
            self.hidden_size = self.model.config.d_model

        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def encode(
        self,
        texts: Union[str, List[str]],
        pooling: str = 'cls',
        max_length: int = 77
    ) -> torch.Tensor:
        """
        텍스트를 임베딩 벡터로 변환합니다.

        Args:
            texts: 단일 텍스트 또는 텍스트 리스트
            pooling: 'cls' (첫 토큰), 'mean' (평균 풀링), 'max' (맥스 풀링)
            max_length: 최대 토큰 길이

        Returns:
            shape (batch_size, hidden_size) 임베딩 텐서
        """
        if isinstance(texts, str):
            texts = [texts]

        # 토크나이징
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )

        # 디바이스로 이동
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        # 모델 forward
        if self.model_type == 'bert':
            outputs = self.model(**encoded)
            hidden_states = outputs.last_hidden_state  # (B, seq_len, hidden)

        elif self.model_type == 'clip':
            outputs = self.model(**encoded)
            hidden_states = outputs.last_hidden_state

        elif self.model_type == 't5':
            outputs = self.model(**encoded)
            hidden_states = outputs.last_hidden_state

        # 풀링
        if pooling == 'cls':
            # 첫 번째 토큰 ([CLS]) 사용
            embeddings = hidden_states[:, 0, :]

        elif pooling == 'mean':
            # 어텐션 마스크를 고려한 평균 풀링
            attention_mask = encoded['attention_mask'].unsqueeze(-1)
            masked_hidden = hidden_states * attention_mask
            sum_hidden = masked_hidden.sum(dim=1)
            sum_mask = attention_mask.sum(dim=1)
            embeddings = sum_hidden / sum_mask.clamp(min=1e-9)

        elif pooling == 'max':
            # 맥스 풀링
            embeddings = hidden_states.max(dim=1)[0]

        else:
            raise ValueError(f"Unknown pooling: {pooling}")

        return embeddings  # (batch_size, hidden_size)

    def get_hidden_size(self) -> int:
        """임베딩 차원 반환"""
        return self.hidden_size


class StyleEncoder(nn.Module):
    """
    학습 가능한 스타일 인코더
    사전정의된 스타일 카테고리를 임베딩으로 변환합니다.
    """

    STYLE_CATEGORIES = [
        'pop', 'hip-hop', 'kpop', 'ballet', 'contemporary',
        'jazz', 'breakdance', 'salsa', 'tango', 'waltz',
        'street', 'house', 'waacking', 'locking', 'popping',
        'krump', 'voguing', 'freestyle'
    ]

    def __init__(self, embedding_dim: int = 256):
        """
        Args:
            embedding_dim: 임베딩 차원
        """
        super().__init__()
        self.num_styles = len(self.STYLE_CATEGORIES)
        self.style_to_idx = {style: i for i, style in enumerate(self.STYLE_CATEGORIES)}

        # 학습 가능한 스타일 임베딩
        self.embedding = nn.Embedding(self.num_styles, embedding_dim)

        # 초기화
        nn.init.normal_(self.embedding.weight, mean=0, std=0.02)

    def forward(self, style_names: List[str]) -> torch.Tensor:
        """
        Args:
            style_names: 스타일 이름 리스트

        Returns:
            shape (batch_size, embedding_dim)
        """
        # 스타일 이름을 인덱스로 변환
        indices = []
        for name in style_names:
            name_lower = name.lower()
            # 가장 유사한 스타일 찾기
            if name_lower in self.style_to_idx:
                idx = self.style_to_idx[name_lower]
            else:
                # 부분 매칭
                matched = False
                for style, idx in self.style_to_idx.items():
                    if style in name_lower or name_lower in style:
                        indices.append(idx)
                        matched = True
                        break
                if not matched:
                    indices.append(0)  # 기본값: 첫 번째 스타일
                continue
            indices.append(idx)

        indices = torch.tensor(indices, dtype=torch.long, device=self.embedding.weight.device)
        return self.embedding(indices)


class HybridStyleEmbedder(nn.Module):
    """
    사전학습 텍스트 임베더와 학습 가능한 스타일 임베딩을 결합합니다.
    """

    def __init__(
        self,
        text_model_type: str = 'bert',
        style_embedding_dim: int = 256,
        fusion_type: str = 'concat'
    ):
        """
        Args:
            text_model_type: 'bert', 'clip', 't5'
            style_embedding_dim: 스타일 임베딩 차원
            fusion_type: 'concat', 'add', 'gated' 중 선택
        """
        super().__init__()

        self.text_embedder = TextEmbedder(text_model_type)
        self.style_encoder = StyleEncoder(style_embedding_dim)
        self.fusion_type = fusion_type

        text_dim = self.text_embedder.get_hidden_size()

        if fusion_type == 'concat':
            self.output_dim = text_dim + style_embedding_dim
            self.fusion = None

        elif fusion_type == 'add':
            # 차원 맞추기
            self.output_dim = max(text_dim, style_embedding_dim)
            self.text_proj = nn.Linear(text_dim, self.output_dim)
            self.style_proj = nn.Linear(style_embedding_dim, self.output_dim)

        elif fusion_type == 'gated':
            # Gated fusion
            self.output_dim = text_dim
            self.gate = nn.Sequential(
                nn.Linear(text_dim + style_embedding_dim, text_dim),
                nn.Sigmoid()
            )
            self.style_proj = nn.Linear(style_embedding_dim, text_dim)

    def forward(self, texts: List[str], style_names: List[str]) -> torch.Tensor:
        """
        Args:
            texts: 텍스트 설명 리스트
            style_names: 스타일 이름 리스트

        Returns:
            융합된 임베딩 (batch_size, output_dim)
        """
        # 텍스트 임베딩
        text_emb = self.text_embedder.encode(texts)  # (B, text_dim)

        # 스타일 임베딩
        style_emb = self.style_encoder(style_names)  # (B, style_dim)

        # 융합
        if self.fusion_type == 'concat':
            output = torch.cat([text_emb, style_emb], dim=1)

        elif self.fusion_type == 'add':
            text_proj = self.text_proj(text_emb)
            style_proj = self.style_proj(style_emb)
            output = text_proj + style_proj

        elif self.fusion_type == 'gated':
            combined = torch.cat([text_emb, style_emb], dim=1)
            gate = self.gate(combined)
            style_proj = self.style_proj(style_emb)
            output = gate * text_emb + (1 - gate) * style_proj

        return output


def test_text_embedder():
    """테스트 함수"""
    print("Text Embedder 테스트")
    print("=" * 50)

    # BERT 테스트
    print("\n1. BERT Embedder")
    embedder = TextEmbedder('bert')
    texts = [
        "강렬한 K-pop 댄스",
        "부드러운 발레 동작",
        "energetic hip-hop dance"
    ]
    embeddings = embedder.encode(texts, pooling='cls')
    print(f"Input: {texts}")
    print(f"Output shape: {embeddings.shape}")
    print(f"Hidden size: {embedder.get_hidden_size()}")

    # Style Encoder 테스트
    print("\n2. Style Encoder")
    style_encoder = StyleEncoder(embedding_dim=256)
    styles = ['kpop', 'ballet', 'hip-hop']
    style_emb = style_encoder(styles)
    print(f"Styles: {styles}")
    print(f"Output shape: {style_emb.shape}")

    # Hybrid 테스트
    print("\n3. Hybrid Style Embedder")
    hybrid = HybridStyleEmbedder(fusion_type='gated')
    output = hybrid(texts, styles)
    print(f"Output shape: {output.shape}")

    print("\n테스트 완료!")


if __name__ == "__main__":
    test_text_embedder()
