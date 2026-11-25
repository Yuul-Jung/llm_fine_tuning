# 제57장 핵심 실습: 학습한 모델을 로컬 API로 띄우고, 대화 이력·Streaming·오류·접근 제어를 확인한다
#
# 실행: python examples/ch57/api_demo.py   (제52장의 병합 모델 필요, CPU 약 1분)

import socket
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (HERE, HERE.parent / "ch44"):
    sys.path.insert(0, str(p))

from chat_template import SYSTEM  # noqa: E402                   (제44장, 학습 때의 System 문구)
from local_api import APIError, Client, Conversation, EchoBackend, HFBackend, make_server, reachable  # noqa: E402

MERGED = ROOT / "models" / "fpga_merged_v1"                      # 제52장


def lan_address():
    """이 컴퓨터의 외부용 주소(없으면 None). 패킷은 보내지 않는다."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
        return None if ip.startswith("127.") else ip
    except OSError:
        return None
    finally:
        s.close()


def main() -> None:
    print("── 1. 서버 시작(학습한 병합 모델, 127.0.0.1) ──")
    server = make_server(HFBackend(MERGED))
    host, port = server.server_address
    client = Client(f"http://{host}:{port}", model="fpga-qwen")
    print(f"  주소 http://{host}:<port>, 모델 이름 fpga-qwen")

    print("\n── 2. Generate API(한 번에 받기) ──")
    print(f"  {client.generate('FPGA의 LUT는 어떤 역할을 하나요?', system=SYSTEM)[:70]}…")

    print("\n── 3. Chat API + Streaming, 대화 이력 ──")
    conv = Conversation(client, SYSTEM)
    for q in ("블로킹 할당과 논블로킹 할당은 언제 쓰나요?", "그럼 조합 논리에서는 어느 쪽을 쓰나요?"):
        answer, n, sent = conv.ask(q)
        print(f"  질문: {q}\n  답({n}조각, 보낸 메시지 {sent}개): {answer[:70]}…")
    print(f"  이력에 쌓인 메시지 {len(conv.turns)}개(사용자 2 + 모델 2), 매 요청마다 System과 함께 다시 보냄")
    server.shutdown()

    print("\n── 4. 오류 처리 ──")
    echo = make_server(EchoBackend(delay=1.0))
    url = f"http://127.0.0.1:{echo.server_address[1]}"
    try:
        Client(url, model="없는-모델").generate("안녕")
    except APIError as e:
        print(f"  없는 모델: {e}(4xx는 다시 시도하지 않음)")
    try:
        list(Client(url, model="echo", timeout=0.3).chat_stream([{"role": "user", "content": "느린 응답"}]))
    except (TimeoutError, socket.timeout):
        print("  Timeout: 조각 사이 1초 > 기다림 0.3초 → 시간 초과로 중단")
    echo.shutdown()
    echo.server_close()
    t0 = time.time()
    try:
        Client(url, model="echo", retries=2).generate("안녕")
    except APIError as e:
        print(f"  서버 꺼짐: {str(e)[:22]}… {time.time() - t0:.1f}초(0.2 + 0.4초 기다리며 재시도)")

    print("\n── 5. 접근 제어 ──")
    srv = make_server(EchoBackend())
    port = srv.server_address[1]
    lan = lan_address()
    print(f"  127.0.0.1로 접속: {reachable('127.0.0.1', port)}")
    if lan:
        print(f"  이 컴퓨터의 외부용 주소로 접속: {reachable(lan, port)}(127.0.0.1에만 열었으므로 차단)")
    else:
        print("  외부용 주소가 없는 환경(확인 생략)")
    srv.shutdown()


if __name__ == "__main__":
    main()
