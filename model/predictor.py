"""
predictor.py
------------
모델을 로드하고 추론(예측)을 수행하는 모듈.

실시간 트레이딩 시스템은 매 분/초마다 새로운 입력 데이터를 받아 모델을 통해 예측값을 얻어야 하므로,
predict() 함수는 매우 효율적이어야 함.

주요 기능:
 - GPU 자동 선택 (CUDA 사용)
 - 입력 OHLCV(최근 sequence_length 길이)를 모델에 전달
 - 미래 close 가격 또는 상승 확률 계산
"""

import torch
import numpy as np


class Predictor:
    """
    실시간 예측 전용 클래스.
    - TimeSeriesTransformer 모델을 입력받아 GPU/CPU에서 추론 실행
    """

    def __init__(self, model, device=None):
        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def predict(self, window_ohlcv: np.ndarray):
        """
        window_ohlcv: 최근 sequence_length 만큼의 OHLCV 배열
                      shape = (seq_len, 5)

        return: 예측된 미래 close 가격(float)
        """
        x = torch.tensor(window_ohlcv, dtype=torch.float32).unsqueeze(0) # (1, seq, 5)
        x = x.to(self.device)

        with torch.no_grad():
            y_pred = self.model(x)   # (1, 1)

        return float(y_pred.item())
