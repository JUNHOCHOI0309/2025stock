"""
timeseries_model.py
-------------------
시계열 예측 모델 정의 파일.

이 프로젝트는 실시간으로 들어오는 OHLCV 데이터를 기반으로 1-step 미래 가격을 예측해야 하므로,
다음과 같은 조건을 충족하는 모델이 필요함:

 - 입력: (sequence_length, batch_size, features=5)  → OHLCV
 - 출력: 다음 close 가격 1-step 예측
 - GPU 학습 최적화
 - 실시간 Online Learning이 가능하도록 구조 단순화

여기서는 Transformer Encoder 기반의 경량 모델을 사용함.
"""

import torch
import torch.nn as nn


class TimeSeriesTransformer(nn.Module):
    """
    Transformer 기반 시계열 예측 모델.
    - OHLCV(5개 feature)를 입력받아 다음 close 가격을 예측함.
    """

    def __init__(self, input_dim=5, embed_dim=64, num_heads=4, num_layers=3):
        super().__init__()

        # OHLCV 5개 값 → 임베딩 (차원 증가)
        self.embedding = nn.Linear(input_dim, embed_dim)

        # Transformer 인코더 레이어 구성
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=0.1,
            batch_first=True  # shape: (batch, seq, features)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 마지막 시점(hidden state)을 사용하여 미래 가격 예측
        self.fc_out = nn.Linear(embed_dim, 1)

    def forward(self, x):
        """
        x: (batch, seq_len, 5)
        return: (batch, 1)
        """
        x = self.embedding(x)          # (batch, seq, embed_dim)
        h = self.transformer(x)        # (batch, seq, embed_dim)
        last_hidden = h[:, -1, :]      # 마지막 시점의 hidden state
        out = self.fc_out(last_hidden) # 미래 close 가격 예측값
        return out
