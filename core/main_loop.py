"""
main_loop.py
------------
AI 자동매매 시스템의 핵심 엔진.

이 파일은 다음 요소를 하나의 흐름으로 통합한다:

1) 실시간 OHLCV 데이터 수집 (Candle Stream)
2) Transformer 모델로 시계열 예측 수행
3) Gemini API로 전략·리스크 분석
4) 시그널 생성 (매수/매도 판단)
5) Binance API로 자동 매매 실행
6) 새로운 데이터로 모델 온라인 학습(Online Learning)
7) 위 과정을 지속적으로 반복

즉, 시스템 전체의 "메인 런타임" 역할을 담당한다.
"""

import time
import numpy as np

from core.config import config
from data.live_stream import MultiSymbolCandleStream
from model.timeseries_model import TimeSeriesTransformer
from model.predictor import Predictor
from model.trainer import OnlineTrainer

# (아직 만들지 않았지만 main_loop 연동을 위해 placeholder 생성)
# /ai/gemini_analyzer.py 가 완성되면 여기 교체됨
class GeminiAnalyzerMock:
    """Gemini 판단 모듈이 완성되기 전 임시 구조"""
    def analyze(self, symbol, predicted_price, last_close):
        """
        임시 전략:
        - 예측값이 현재가보다 높으면 → BUY
        - 예측값이 낮으면 → SELL
        """
        if predicted_price > last_close:
            return "BUY"
        else:
            return "SELL"


# /trading/binance_client.py placeholder (나중에 실제 기능으로 교체됨)
class BinanceTradingMock:
    """Binance 자동매매 기능이 완성되기 전 임시 모듈"""
    def execute(self, symbol, signal):
        print(f"[TRADE] {symbol}: EXECUTE {signal}")


def run_main_loop():
    """
    전체 자동매매 시스템의 메인 실행 루프.
    이 함수는 무한 루프로 실행되며 실시간 예측 + 분석 + 매매를 수행함.
    """

    print("🚀 Starting AI Trading Main Loop...")
    print("   • Transformer 모델 로딩")
    print("   • 실시간 OHLCV 스트림 연결 중...")

    # ----------------------------------------
    # 1) 실시간 OHLCV 스트림 준비
    # ----------------------------------------
    candle_stream = MultiSymbolCandleStream(
        symbols=[s.lower() for s in config.TRADE_SYMBOLS],
        interval="1m"
    )
    candle_stream.start()

    # ----------------------------------------
    # 2) 모델 · Predictor · Trainer 초기화
    # ----------------------------------------
    model = TimeSeriesTransformer(
        input_dim=5,
        embed_dim=config.MODEL_EMBED,
        num_heads=config.MODEL_HEADS,
        num_layers=config.MODEL_LAYERS
    )

    predictor = Predictor(model)
    trainer = OnlineTrainer(
        model=model,
        lr=config.LEARNING_RATE,
        seq_len=config.SEQ_LEN
    )

    # ----------------------------------------
    # 3) 분석 & 매매 모듈 (현재는 Mock)
    # ----------------------------------------
    gemini = GeminiAnalyzerMock()
    trader = BinanceTradingMock()

    # ----------------------------------------
    # 4) 메인 루프 시작
    # ----------------------------------------
    while True:
        for symbol in config.TRADE_SYMBOLS:

            # 최근 OHLCV DataFrame 가져오기
            df = candle_stream.get_dataframe(symbol.lower())
            if df is None or len(df) < config.SEQ_LEN + 1:
                continue

            # DataFrame → Numpy 변환
            window = df.iloc[-config.SEQ_LEN:][["open","high","low","close","volume"]].values
            last_close = df.iloc[-1]["close"]

            # ----------------------------------------
            # (A) 모델 예측
            # ----------------------------------------
            predicted = predictor.predict(window)

            if config.DEBUG_MODE:
                print(f"[{symbol}] Predicted={predicted:.4f}  LastClose={last_close:.4f}")

            # ----------------------------------------
            # (B) Gemini(지능형 판단) 분석
            # ----------------------------------------
            decision = gemini.analyze(symbol, predicted, last_close)

            # ----------------------------------------
            # (C) 자동 매매 실행
            # ----------------------------------------
            if config.ENABLE_AUTO_TRADE:
                trader.execute(symbol, decision)

            # ----------------------------------------
            # (D) 온라인 학습 데이터 추가
            # ----------------------------------------
            ohlcv_row = df.iloc[-1][["open","high","low","close","volume"]].values
            trainer.add_data(ohlcv_row)

            # ----------------------------------------
            # (E) 온라인 학습 step 실행
            # ----------------------------------------
            loss = trainer.train_step()
            if loss is not None and config.DEBUG_MODE:
                print(f"[TRAIN] Loss={loss:.6f}")

        # 메인 루프 주기
        time.sleep(config.LOOP_INTERVAL)
