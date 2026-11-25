# 제56장 핵심 실습: 학습·평가 때의 템플릿과 생성 설정을 그대로 옮긴 Ollama Modelfile을 만들고 대조한다
#
# 실행: python examples/ch56/modelfile.py   (제54·55장의 GGUF 필요, 수 초)
#
# 만드는 파일: models/gguf/Modelfile
# Ollama 등록(ollama create)과 실제 응답·GPU 적재 확인은 교재 검증 환경에서 하지 않았다(미검증).
# 이 실습은 Modelfile의 각 항목이 학습·평가 조건과 같은지를 파이썬으로 대조한다.

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (HERE.parent, HERE.parent / "ch44"):
    sys.path.insert(0, str(p))

from chat_template import SYSTEM, render  # noqa: E402            (제44장)

GGUF_DIR = ROOT / "models" / "gguf"
MERGED = ROOT / "models" / "fpga_merged_v1"

# Ollama의 TEMPLATE은 Go 템플릿 문법이다. Qwen2.5의 ChatML(제44장)과 같은 문자열을 만들도록 쓴다.
TEMPLATE = """{{- if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}
{{- range .Messages }}<|im_start|>{{ .Role }}
{{ .Content }}<|im_end|>
{{ end }}<|im_start|>assistant
"""


def modelfile(gguf_name: str, gen: dict, num_ctx: int = 2048) -> str:
    """FROM(모델) · SYSTEM(기본 System 문구) · TEMPLATE(대화 → 문자열) · PARAMETER(생성 설정)"""
    lines = [f"FROM ./{gguf_name}", "", f'SYSTEM """{SYSTEM}"""', "", f'TEMPLATE """{TEMPLATE}"""', "",
             "PARAMETER temperature 0",                                 # Greedy
             f"PARAMETER repeat_penalty {gen['repetition_penalty']}",  # 원본 생성 설정(제54장)
             "PARAMETER repeat_last_n -1",                              # 반복 억제를 문맥 전체에 적용(Transformers와 같게)
             f"PARAMETER num_ctx {num_ctx}",
             "PARAMETER num_predict 160",                               # 평가 때의 최대 생성 길이(제47장)
             'PARAMETER stop "<|im_end|>"',
             'PARAMETER stop "<|endoftext|>"']
    return "\n".join(lines) + "\n"


def render_go_like(system: str, messages: list) -> str:
    """위 TEMPLATE을 이 장에서 쓰는 문법(if, range, 값 치환)만 따라 펼친 결과(대조용 모사)"""
    out = f"<|im_start|>system\n{system}<|im_end|>\n" if system else ""
    for m in messages:
        out += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
    return out + "<|im_start|>assistant\n"


def parse_parameters(text: str) -> dict:
    params = {}
    for k, v in re.findall(r"^PARAMETER (\S+) (.+)$", text, re.M):
        params.setdefault(k, []).append(v.strip('"'))
    return params


def main() -> None:
    from transformers import AutoTokenizer
    gen = json.loads((MERGED / "generation_config.json").read_text())
    tok = AutoTokenizer.from_pretrained(MERGED)
    text = modelfile("fpga-q8_0.gguf", gen)
    (GGUF_DIR / "Modelfile").write_text(text, encoding="utf-8")
    print("── 1. Modelfile ──")
    print(text)

    print("── 2. 학습·평가 조건과 대조 ──")
    q = [{"role": "user", "content": "FPGA의 LUT는 어떤 역할을 하나요?"}]
    hf = render(tok, [{"role": "system", "content": SYSTEM}] + q, gen=True)
    print(f"  TEMPLATE을 펼친 문자열 = 학습 때의 추론 문자열(제44장): {render_go_like(SYSTEM, q) == hf}")
    two = q + [{"role": "assistant", "content": "LUT는 작은 메모리입니다."},
               {"role": "user", "content": "입력이 6개면요?"}]
    hf2 = render(tok, [{"role": "system", "content": SYSTEM}] + two, gen=True)
    print(f"  여러 차례 대화도 같음: {render_go_like(SYSTEM, two) == hf2}")
    p = parse_parameters(text)
    print(f"  반복 억제 {p['repeat_penalty'][0]} = 원본 생성 설정 {gen['repetition_penalty']}: "
          f"{float(p['repeat_penalty'][0]) == gen['repetition_penalty']}")
    eos = [tok.convert_ids_to_tokens(i) for i in gen["eos_token_id"]]
    print(f"  멈춤 토큰 {p['stop']} = 원본 종료 토큰 {eos}: {sorted(p['stop']) == sorted(eos)}")
    print(f"  SYSTEM = 학습 때의 System 문구: {SYSTEM in text}")

    print("\n── 3. 등록과 확인 명령(교재 검증 환경에서는 미실행) ──")
    for cmd in ("cd models/gguf", "ollama create fpga-qwen -f Modelfile", "ollama run fpga-qwen \"FPGA의 LUT는 어떤 역할을 하나요?\"",
                "ollama ps"):
        print(f"  {cmd}")


if __name__ == "__main__":
    main()
