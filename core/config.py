"""
config.py
---------
AI 트레이딩 프로젝트 전체에서 공통으로 사용하는 설정(Config)을 관리하는 파일.

이 파일의 역할:
 - API KEY 저장 (환경변수 또는 별도 파일로 관리)
 - 시계열 모델 설정(seq_len, 학습률 등)
 - 트레이딩 관련 설정(최소 주문 수량, 슬리피지, 레버리지 등)
 - Gemini API 설정
 - Binance API 설정
"""

import os
from dotenv import load_dotenv

# .env 파일 로드 (API 키 등 민감정보를 코드 외부에 저장)
load_dotenv()


class Config:
    """프로젝트 전체에서 공유되는 설정 클래스"""

    # -----------------------------
    # 1) Binance API 설정
    # -----------------------------
    BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
    BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY")

    TRADE_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]

    # 자동매매 기본 옵션
    TRADE_QUANTITY = 0.01          # 기본 매수/매도 수량
    ENABLE_AUTO_TRADE = True       # True면 자동 매매 실행
    FUTURES = False                # 선물 트레이딩 여부

    # -----------------------------
    # 2) Gemini API 설정
    # -----------------------------
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL = "gemini-pro"

    # -----------------------------
    # 3) 시계열 모델 설정 (Transformer)
    # -----------------------------
    SEQ_LEN = 60          # 입력 시계열 길이 (예: 최근 60개 캔들 사용)
    LEARNING_RATE = 1e-4
    MODEL_EMBED = 64
    MODEL_LAYERS = 3
    MODEL_HEADS = 4

    # -----------------------------
    # 4) 시스템 실행 옵션
    # -----------------------------
    LOOP_INTERVAL = 1.0   # main_loop 실행 간격(초)
    DEBUG_MODE = True     # 디버그 로그 출력 여부


config = Config()
