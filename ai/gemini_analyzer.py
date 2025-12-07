"""
gemini_analyzer.py
------------------
Gemini API를 활용하여 실시간 트레이딩 의사결정을 돕는 모듈.

이 모듈의 역할:
 - 모델 예측값(predicted_price)과 현재 시세(last_close)를 기반으로
   Gemini에게 트레이딩 전략 분석 요청
 - BUY/SELL/HOLD 여부 판단
 - 리스크 설명 및 근거 제공
 - 포지션 크기 추천

출력 예시:
{
    "action": "BUY",
    "confidence": 0.78,
    "position_size": 0.32,
    "reason": "단기 상승 확률이 높으며 변동성이 안정적.",
    "risk": "BTC 변동성 증가 주의."
}
"""

import google.generativeai as genai
from core.config import config


class GeminiAnalyzer:
    """
    Gemini LLM을 사용해 트레이딩 전략을 분석하는 클래스.
    - 모델 예측값(predicted_price)
    - 현재가(last_close)
    - 기술적 지표(optional)
    를 입력으로 받아 트레이딩 결정을 생성함.
    """

    def __init__(self):
        # Gemini API 키 설정
        genai.configure(api_key=config.GEMINI_API_KEY)

        # 사용할 모델명 설정 (기본: gemini-pro)
        self.model = genai.GenerativeModel(config.GEMINI_MODEL)

    def build_prompt(self, symbol, predicted_price, last_close):
        """
        LLM에게 전달할 프롬프트를 생성하는 함수.
        이 프롬프트를 바꿔서 전략을 세밀하게 조절할 수도 있음.
        """

        price_diff = predicted_price - last_close
        pct_change = (price_diff / last_close) * 100

        prompt = f"""
당신은 전문 암호화폐 트레이딩 분석가입니다.

**현재 코인**: {symbol}
**모델 예측 미래 가격**: {predicted_price:.4f}
**현재 또는 최근 종가**: {last_close:.4f}
**예상 변동폭**: {pct_change:.4f} %

다음 조건에 따라 전략을 분석하세요:

1) BUY / SELL / HOLD 중 하나를 결정
2) 이유를 명확하게 설명
3) 예측 결과에 기반한 confidence(0~1)를 수치로 제공
4) 포지션 크기 추천 (0~1 사이값, 예: 0.25 = 자금의 25%)
5) 리스크 요인도 텍스트로 요약

응답은 반드시 JSON 형식으로만 출력하세요:

예시 형식:
{
  "action": "BUY",
  "confidence": 0.75,
  "position_size": 0.30,
  "reason": "단기 상승 흐름 지속 가능성이 높음.",
  "risk": "변동성 주의."
}
"""

        return prompt

    def analyze(self, symbol, predicted_price, last_close):
        """
        Gemini API를 호출하여 트레이딩 분석을 수행.

        return: dict
        {
            "action": "...",
            "confidence": 0.x,
            "position_size": 0.x,
            "reason": "...",
            "risk": "..."
        }
        """

        prompt = self.build_prompt(symbol, predicted_price, last_close)

        try:
            # Gemini에게 분석 요청
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.5,
                    "top_p": 0.9,
                }
            )

            # Gemini 응답 내용
            text = response.text.strip()

            # JSON 파싱
            import json
            result = json.loads(text)

            return result

        except Exception as e:
            print("[Gemini ERROR]", e)

            # 장애 발생 시 기본 fallback 전략
            fallback = {
                "action": "HOLD",
                "confidence": 0.0,
                "position_size": 0.0,
                "reason": "Gemini 분석 실패로 HOLD 추천.",
                "risk": "LLM 응답 오류."
            }
            return fallback
