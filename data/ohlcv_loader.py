"""
ohlcv_loader.py
----------------
여러 코인의 과거 OHLCV 데이터를 동시에 불러오기 쉽게 확장된 버전.
BTC, ETH, SOL, XRP 등 원하는 심볼을 리스트로 전달하면 자동으로 가져옵니다.
"""

import ccxt
import pandas as pd
from datetime import datetime


class OHLCVLoader:
    def __init__(self, exchange=None):
        self.exchange = exchange or ccxt.binance()

    def fetch_single(self, symbol="BTC/USDT", timeframe="1m", limit=1000):
        """단일 심볼 OHLCV 로드 (기본 단위 기능)"""
        raw = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

        df = pd.DataFrame(raw, columns=[
            "timestamp", "open", "high", "low", "close", "volume"
        ])
        df["datetime"] = df["timestamp"].apply(lambda x: datetime.fromtimestamp(x / 1000))
        df["symbol"] = symbol

        return df

    def fetch_multi(self, symbols=None, timeframe="1m", limit=1000):
        """
        여러 심볼의 OHLCV를 불러서 딕셔너리로 반환.
        예: {"BTC/USDT": df1, "ETH/USDT": df2, ...}
        """
        if symbols is None:
            symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"]

        result = {}
        for symbol in symbols:
            print(f"[LOAD] Fetching OHLCV → {symbol}")
            result[symbol] = self.fetch_single(symbol, timeframe, limit)

        return result
