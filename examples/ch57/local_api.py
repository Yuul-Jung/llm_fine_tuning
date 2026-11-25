# 제57장: Ollama와 같은 형식의 로컬 API 서버(학습한 모델 제공)와, 대화 이력·Streaming·오류를 다루는 클라이언트
#
# 서버는 Ollama의 /api/generate, /api/chat 요청·응답 형식(JSON, 줄마다 JSON 하나인 Streaming)을 따른다.
# 같은 클라이언트를 실제 Ollama(기본 주소 http://127.0.0.1:11434)에 그대로 쓸 수 있다.

import json
import socket
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


# ── 서버 ──────────────────────────────────────────────────────────────

class EchoBackend:
    """테스트용: 마지막 사용자 말을 단어 단위로 되돌려준다(지연 시간 조절 가능)"""

    def __init__(self, name="echo", delay=0.0):
        self.name, self.delay = name, delay

    def stream(self, messages):
        for word in ("답: " + messages[-1]["content"]).split(" "):
            time.sleep(self.delay)
            yield word + " "


class HFBackend:
    """학습한 모델(제52장 병합 모델)을 Greedy로 생성하며 조각을 내보낸다"""

    def __init__(self, model_dir, name="fpga-qwen", max_new_tokens=120):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        self.torch, self.name, self.n = torch, name, max_new_tokens
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32).eval()

    def stream(self, messages):
        from transformers import TextIteratorStreamer
        text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = self.tok(text, return_tensors="pt", add_special_tokens=False).input_ids
        streamer = TextIteratorStreamer(self.tok, skip_prompt=True, skip_special_tokens=True)
        kw = dict(input_ids=ids, max_new_tokens=self.n, do_sample=False, streamer=streamer,
                  pad_token_id=self.tok.pad_token_id)
        threading.Thread(target=lambda: self.model.generate(**kw), daemon=True).start()
        yield from (piece for piece in streamer if piece)


def make_server(backend, host="127.0.0.1", port=0):
    """host 기본값 127.0.0.1: 이 컴퓨터 안에서만 접속할 수 있다(인증이 없으므로 외부에 열지 않는다)"""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):                       # 요청마다 찍히는 기록 끔
            pass

        def reply(self, code, obj):
            body = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            except json.JSONDecodeError:
                return self.reply(400, {"error": "invalid JSON"})
            if self.path not in ("/api/generate", "/api/chat"):
                return self.reply(404, {"error": "not found"})
            if req.get("model") != backend.name:
                return self.reply(404, {"error": f"model '{req.get('model')}' not found"})
            chat = self.path == "/api/chat"
            if chat:
                messages = req.get("messages", [])
            else:                                            # generate: system 항목이 있으면 앞에 붙인다(Ollama와 같음)
                messages = ([{"role": "system", "content": req["system"]}] if req.get("system") else []) \
                    + [{"role": "user", "content": req.get("prompt", "")}]

            def chunk(text, done):
                obj = {"model": backend.name, "done": done}
                if chat:
                    obj["message"] = {"role": "assistant", "content": text}
                else:
                    obj["response"] = text
                return obj

            if not req.get("stream", True):
                return self.reply(200, chunk("".join(backend.stream(messages)), True))
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.end_headers()
            for piece in backend.stream(messages):           # 한 줄에 JSON 하나(NDJSON)
                self.wfile.write((json.dumps(chunk(piece, False), ensure_ascii=False) + "\n").encode())
                self.wfile.flush()
            self.wfile.write((json.dumps(chunk("", True)) + "\n").encode())

    server = ThreadingHTTPServer((host, port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


# ── 클라이언트 ────────────────────────────────────────────────────────

class APIError(Exception):
    pass


class Client:
    def __init__(self, base_url="http://127.0.0.1:11434", model="fpga-qwen", timeout=60.0, retries=2):
        self.base, self.model, self.timeout, self.retries = base_url, model, timeout, retries

    def _open(self, path, payload):
        """연결 실패·5xx는 잠시 기다렸다 다시 시도, 4xx는 요청이 틀린 것이므로 바로 오류"""
        data = json.dumps(payload).encode()
        for attempt in range(self.retries + 1):
            req = urllib.request.Request(self.base + path, data=data, headers={"Content-Type": "application/json"})
            try:
                return urllib.request.urlopen(req, timeout=self.timeout)
            except urllib.error.HTTPError as e:
                msg = json.loads(e.read() or b"{}").get("error", e.reason)
                if e.code < 500 or attempt == self.retries:
                    raise APIError(f"HTTP {e.code}: {msg}") from None
            except (urllib.error.URLError, ConnectionError) as e:
                if attempt == self.retries:
                    raise APIError(f"연결 실패({self.retries + 1}회 시도): {e}") from None
            time.sleep(0.2 * 2 ** attempt)                          # 0.2초, 0.4초 … 지수적으로 늘림

    def generate(self, prompt, system=None):
        payload = {"model": self.model, "prompt": prompt, "stream": False}
        if system:
            payload["system"] = system
        with self._open("/api/generate", payload) as r:
            return json.loads(r.read())["response"]

    def chat_stream(self, messages):
        """조각을 차례로 돌려준다. 조각 사이가 timeout보다 길면 socket.timeout"""
        with self._open("/api/chat", {"model": self.model, "messages": messages, "stream": True}) as r:
            for line in r:
                obj = json.loads(line)
                if obj["done"]:
                    return
                yield obj["message"]["content"]


class Conversation:
    """대화 이력을 들고 있다가 매번 전체를 보낸다(서버는 이전 대화를 기억하지 않는다)"""

    def __init__(self, client, system, max_turns=4):
        self.client, self.system, self.max_turns, self.turns = client, system, max_turns, []

    def ask(self, text, on_piece=None):
        self.turns.append({"role": "user", "content": text})
        messages = [{"role": "system", "content": self.system}] + self.turns[-(2 * self.max_turns - 1):]
        pieces = []
        for p in self.client.chat_stream(messages):
            pieces.append(p)
            if on_piece:
                on_piece(p)
        answer = "".join(pieces)
        self.turns.append({"role": "assistant", "content": answer})
        return answer, len(pieces), len(messages)


def reachable(host, port, timeout=0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False
