# 제57장 검증 테스트(모델 없이 EchoBackend로 정상·오류·Timeout·접근 제어를 확인한다)
#
# 실행: python -m pytest examples/ch57 -q

import socket
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pytest  # noqa: E402

from local_api import APIError, Client, Conversation, EchoBackend, make_server, reachable  # noqa: E402


@pytest.fixture
def server():
    s = make_server(EchoBackend())
    yield s
    s.shutdown()
    s.server_close()


def url(s):
    return f"http://127.0.0.1:{s.server_address[1]}"


def test_generate_and_stream(server):
    c = Client(url(server), model="echo")
    assert c.generate("가 나") == "답: 가 나 "
    assert c.generate("가", system="규칙") == "답: 가 "
    assert list(c.chat_stream([{"role": "user", "content": "가 나"}])) == ["답: ", "가 ", "나 "]


def test_conversation_keeps_and_trims_history(server):
    conv = Conversation(Client(url(server), model="echo"), "시스템", max_turns=2)
    for q in ("하나", "둘", "셋"):
        _, _, sent = conv.ask(q)
    assert len(conv.turns) == 6 and sent == 1 + 3                    # System + 최근 3개(사용자·모델·사용자)


def test_client_errors(server):
    t0 = time.time()
    with pytest.raises(APIError, match="404"):
        Client(url(server), model="없음", retries=2).generate("x")
    assert time.time() - t0 < 0.5                                      # 4xx는 재시도하지 않는다


def test_timeout():
    s = make_server(EchoBackend(delay=0.5))
    with pytest.raises((TimeoutError, socket.timeout)):
        list(Client(url(s), model="echo", timeout=0.2).chat_stream([{"role": "user", "content": "a b"}]))
    s.shutdown()


def test_retry_when_server_down():
    s = make_server(EchoBackend())
    u = url(s)
    s.shutdown()
    s.server_close()
    t0 = time.time()
    with pytest.raises(APIError, match="3회"):
        Client(u, model="echo", retries=2).generate("x")
    assert time.time() - t0 >= 0.6


def test_bound_to_loopback_only(server):
    assert server.server_address[0] == "127.0.0.1" and reachable("127.0.0.1", server.server_address[1])
