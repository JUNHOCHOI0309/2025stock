"""
trainer.py
-----------
실시간으로 들어오는 OHLCV 데이터를 사용해
TimeSeriesTransformer 모델을 온라인 학습시키는 모듈.

주요 기능:
 - 학습 버퍼 관리
 - mini-batch 생성
 - GPU 학습 수행(4070Ti 최적화)
 - 온라인 학습(매 데이터마다 1~2 step만 학습)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class OnlineTrainer:
    """
    Transformer 기반 시계열 모델을 실시간 학습시키는 Trainer 클래스.

    매 tick(또는 1분)마다:
      1) 최신 OHLCV를 버퍼에 추가
      2) seq_len 길이의 입력 데이터 생성
      3) 모델 forward + loss 계산
      4) backward + optimizer step 수행
    """

    def __init__(
        self,
        model,
        lr=1e-4,
        seq_len=60,
        device=None
    ):
        self.model = model
        self.seq_len = seq_len
        self.buffer = []  # 실시간 OHLCV 누적 버퍼

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr)

    def add_data(self, ohlcv_row):
        """
        새로운 1개 OHLCV 데이터(1분 캔들 등)를 버퍼에 추가.
        ohlcv_row: [open, high, low, close, volume]
        """
        self.buffer.append(ohlcv_row)

    def can_train(self):
        """학습 가능한 최소 데이터가 있는지 확인"""
        return len(self.buffer) > self.seq_len

    def train_step(self):
        """
        온라인 학습 1 step 수행.
        - 최근 seq_len 만큼을 input으로 사용
        - 그 다음 close 값을 target으로 사용
        """
        if not self.can_train():
            return None  # 데이터 부족

        # 입력 X : 최근 seq_len OHLCV
        window = np.array(self.buffer[-self.seq_len - 1 : -1])  # shape=(seq,5)
        
        # target y : 가장 최근 close 값
        target = np.array([self.buffer[-1][3]])  # close 값만 사용

        # Tensor 변환
        x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(self.device)
        y = torch.tensor(target, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Forward + Loss
        self.optimizer.zero_grad()
        pred = self.model(x)  # (1,1)
        loss = self.loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        self.optimizer.step()

        return float(loss.item())
