# 제01장 검증 테스트
#
# 실행: python -m pytest examples/ch01 -q
#
# 실제 GPU나 Ollama가 없어도 실행되도록, 장치 상태와 Ollama 서버를 모의(mock)로 흉내 낸다.

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.device import select_device  # noqa: E402
import check_env  # noqa: E402
import ollama_generate  # noqa: E402


# ── 1. 장치 선택 모의 실험 ──────────────────────────────────────────────

def fake_gpu(monkeypatch, available, capability=(6, 1), arch_list=("sm_61",)):
    """torch.cuda의 판단 함수를 바꿔 원하는 GPU 상황을 흉내 낸다."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda i=0: capability)
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: list(arch_list))


@pytest.mark.parametrize(
    "available, capability, arch_list, expected",
    [
        (False, (0, 0), (), ("cpu", torch.float32)),                     # GPU 없음
        (True, (6, 1), ("sm_75", "sm_80"), ("cpu", torch.float32)),      # GT 1030 + 미지원 빌드
        (True, (6, 1), ("sm_61", "sm_75"), ("cuda", torch.float32)),     # GT 1030 + cu126 빌드
        (True, (7, 5), ("sm_75",), ("cuda", torch.float16)),             # Turing
        (True, (8, 6), ("sm_86",), ("cuda", torch.bfloat16)),            # Ampere
    ],
)
def test_select_device(monkeypatch, available, capability, arch_list, expected):
    fake_gpu(monkeypatch, available, capability, arch_list)
    assert select_device(verbose=False) == expected


# ── 2. 환경 보고서 ──────────────────────────────────────────────────────

def test_report_has_required_fields():
    report = check_env.collect_report()
    for key in ("os", "python", "torch", "device", "dtype", "ram_gib"):
        assert key in report


def test_smoke_test_on_selected_device():
    device, dtype = select_device(verbose=False)
    error = check_env.smoke_test(device, dtype)
    # FP32는 거의 0, 16비트 자료형은 정밀도가 낮아 기준을 넓게 잡는다.
    limit = 1e-4 if dtype == torch.float32 else 0.5
    assert error <= limit


# ── 3. Ollama API 클라이언트(모의 서버) ─────────────────────────────────

FAKE_RESPONSE = {
    "model": "qwen2.5:0.5b",
    "response": "LLM은 다음 토큰을 예측하는 언어 모델입니다.",
    "done": True,
    "prompt_eval_count": 12,
    "eval_count": 20,
    "eval_duration": 500_000_000,   # 0.5초(나노초 단위)
}


class FakeOllama(BaseHTTPRequestHandler):
    received = {}

    def do_POST(self):
        length = int(self.headers["Content-Length"])
        FakeOllama.received = json.loads(self.rfile.read(length))
        body = json.dumps(FAKE_RESPONSE).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def test_generate_with_fake_server():
    server = HTTPServer(("127.0.0.1", 0), FakeOllama)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/api/generate"
        result = ollama_generate.generate("테스트", url=url)
    finally:
        server.shutdown()

    # 요청: 재현성 설정(temperature 0, seed 고정)이 실제로 전송되었는가
    assert FakeOllama.received["stream"] is False
    assert FakeOllama.received["options"] == {"temperature": 0, "seed": 42}
    # 응답: 20개 토큰 / 0.5초 = 40 tokens/s
    assert result["response"] == FAKE_RESPONSE["response"]
    assert ollama_generate.tokens_per_second(result) == pytest.approx(40.0)


def test_tokens_per_second_zero_duration():
    assert ollama_generate.tokens_per_second({"eval_count": 5, "eval_duration": 0}) == 0.0
