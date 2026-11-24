# 제01장 핵심 실습 2: Ollama API로 로컬 LLM에 질문하고 생성 속도 측정
#
# 준비: ollama pull qwen2.5:0.5b  (Ollama 서버가 실행 중이어야 한다)
# 실행: python examples/ch01/ollama_generate.py "질문"

import json
import sys
import urllib.error
import urllib.request

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL = "qwen2.5:0.5b"


def generate(prompt: str, url: str = OLLAMA_URL, model: str = MODEL,
             timeout: float = 120.0) -> dict:
    """프롬프트 하나를 보내고 응답 JSON 전체를 돌려준다.

    temperature 0과 고정 seed로 같은 질문에 같은 답이 나오게 한다(재현성).
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0, "seed": 42},
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def tokens_per_second(result: dict) -> float:
    """생성 토큰 수(eval_count)를 생성 시간(eval_duration, 나노초)으로 나눈다."""
    duration_ns = result.get("eval_duration", 0)
    if duration_ns <= 0:
        return 0.0
    return result.get("eval_count", 0) / (duration_ns / 1e9)


def main() -> None:
    prompt = sys.argv[1] if len(sys.argv) > 1 else "LLM을 한 문장으로 설명해 주세요."
    try:
        result = generate(prompt)
    except urllib.error.URLError as error:
        print(f"Ollama 서버에 연결할 수 없습니다: {error.reason}")
        print("확인: 'ollama serve' 실행 여부, 'ollama pull qwen2.5:0.5b' 완료 여부")
        sys.exit(1)

    print("=== 응답 ===")
    print(result.get("response", "").strip())
    print("\n=== 생성 통계 ===")
    print(f"입력 토큰 수   : {result.get('prompt_eval_count', 0)}")
    print(f"생성 토큰 수   : {result.get('eval_count', 0)}")
    print(f"초당 생성 토큰 : {tokens_per_second(result):.1f} tokens/s")


if __name__ == "__main__":
    main()
