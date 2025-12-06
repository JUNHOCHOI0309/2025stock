"""
live_stream.py
----------------
여러 심볼(BTC, ETH, SOL, XRP)의 실시간 가격/캔들 데이터를 동시에 수신하기 위한 모듈.

 - 실시간 가격 스트림: trade stream 사용
 - 실시간 캔들 스트림: kline stream 사용
 - 여러 코인을 동시에 구독 가능하도록 구조 확장
"""

import json
import websocket
import threading
import pandas as pd
from datetime import datetime


# -----------------------------
# 1) 실시간 가격 스트림 (다중 심볼)
# -----------------------------
class MultiSymbolPriceStream:
    """
    여러 코인의 실시간 가격을 WebSocket으로 수신하는 클래스.
    예) ["btcusdt", "ethusdt", "solusdt", "xrpusdt"]
    
    latest_prices = {
        "btcusdt": 43200.5,
        "ethusdt": 2240.3,
        "solusdt": 92.15,
        "xrpusdt": 0.61
    }
    """

    def __init__(self, symbols=None):
        if symbols is None:
            symbols = ["btcusdt", "ethusdt", "solusdt", "xrpusdt"]

        self.symbols = [s.lower() for s in symbols]
        self.latest_prices = {s: None for s in self.symbols}
        self.ws_list = []      # 여러 웹소켓 관리
        self.threads = []

    def create_stream(self, symbol):
        """특정 심볼에 대한 WebSocket 스트림 생성"""
        stream_url = f"wss://stream.binance.com:9443/ws/{symbol}@trade"

        def on_message(ws, message):
            data = json.loads(message)
            self.latest_prices[symbol] = float(data["p"])  # last price

        ws = websocket.WebSocketApp(stream_url, on_message=on_message)
        return ws

    def start(self):
        """여러 심볼의 가격 스트림을 각각 별도 쓰레드로 시작"""
        for symbol in self.symbols:
            ws = self.create_stream(symbol)
            self.ws_list.append(ws)

            t = threading.Thread(target=ws.run_forever)
            t.daemon = True
            t.start()
            self.threads.append(t)

            print(f"[STREAM STARTED] {symbol} price stream running...")

    def get_latest_prices(self):
        """모든 심볼의 최신 가격 반환"""
        return self.latest_prices


# -----------------------------
# 2) 실시간 OHLCV(Kline) 스트림 (다중 심볼)
# -----------------------------
class MultiSymbolCandleStream:
    """
    여러 심볼의 캔들 데이터를 동시에 WebSocket으로 받아서
    DataFrame 형식으로 저장하는 클래스.

    df_map = {
        "btcusdt": DataFrame(...),
        "ethusdt": DataFrame(...),
        "solusdt": DataFrame(...),
        "xrpusdt": DataFrame(...)
    }
    """

    def __init__(self, symbols=None, interval="1m"):
        if symbols is None:
            symbols = ["btcusdt", "ethusdt", "solusdt", "xrpusdt"]

        self.symbols = [s.lower() for s in symbols]
        self.interval = interval
        self.ws_list = []
        self.threads = []

        # DataFrame 저장소
        self.df_map = {s: pd.DataFrame() for s in self.symbols}

    def create_stream(self, symbol):
        """특정 심볼의 캔들 스트림 생성"""
        stream_url = f"wss://stream.binance.com:9443/ws/{symbol}@kline_{self.interval}"

        def on_message(ws, message):
            data = json.loads(message)
            k = data["k"]

            candle = {
                "open_time": datetime.fromtimestamp(k["t"] / 1000),
                "open": float(k["o"]),
                "high": float(k["h"]),
                "low": float(k["l"]),
                "close": float(k["c"]),
                "volume": float(k["v"]),
                "closed": k["x"]
            }

            # 해당 심볼의 DataFrame에 추가
            self.df_map[symbol] = pd.concat(
                [self.df_map[symbol], pd.DataFrame([candle])],
                ignore_index=True
            )

        ws = websocket.WebSocketApp(stream_url, on_message=on_message)
        return ws

    def start(self):
        """각 심볼마다 WebSocket 캔들 스트림 시작"""
        for symbol in self.symbols:
            ws = self.create_stream(symbol)
            self.ws_list.append(ws)

            t = threading.Thread(target=ws.run_forever)
            t.daemon = True
            t.start()
            self.threads.append(t)

            print(f"[STREAM STARTED] {symbol} {self.interval} candle stream running...")

    def get_latest_candle(self, symbol):
        """특정 심볼의 가장 최신 캔들 반환"""
        df = self.df_map.get(symbol.lower())
        if df is None or len(df) == 0:
            return None
        return df.iloc[-1]

    def get_dataframe(self, symbol):
        """특정 심볼의 전체 DataFrame 반환"""
        return self.df_map.get(symbol.lower(), None)
