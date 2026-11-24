# 제20장 핵심 실습 2: Sampling 설정이 확률 분포를 바꾸는 모습과 KV Cache의 크기·효과 측정
#
# 실행: python examples/ch20/sampling_kv.py

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from common.qwen import load_config, load_model, load_tokenizer  # noqa: E402

PROMPT = "대한민국의 수도는"


def kv_bytes_per_token(cfg, bytes_per: int) -> int:
    """K와 V(2) × 계층 × KV 헤드 × head_dim × 자료형 바이트"""
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    return 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * head_dim * bytes_per


def filter_top_k(probs, k):
    keep = torch.zeros_like(probs, dtype=torch.bool)
    keep[probs.topk(k).indices] = True
    return keep


def filter_top_p(probs, p):
    """누적 확률이 p에 처음 닿는 토큰까지 남긴다(1등은 항상 남는다)."""
    sorted_p, idx = probs.sort(descending=True)
    before = sorted_p.cumsum(0) - sorted_p                # 자기 앞까지의 누적
    keep = torch.zeros_like(probs, dtype=torch.bool)
    keep[idx[before < p]] = True
    return keep


@torch.no_grad()
def greedy(model, ids, n, use_cache):
    """KV Cache를 쓰면 두 번째 단계부터 새 토큰 하나만 넣는다."""
    out, cache, x = [], None, ids
    for _ in range(n):
        r = model(x, past_key_values=cache, use_cache=use_cache)
        nxt = r.logits[:, -1].argmax(-1, keepdim=True)
        out.append(nxt.item())
        if use_cache:
            cache, x = r.past_key_values, nxt
        else:
            x = torch.cat([x, nxt], dim=1)                # 매번 전체 문장을 다시 계산
    return out, cache


def main() -> None:
    cfg, tok, model = load_config(), load_tokenizer(), load_model()
    ids = tok(PROMPT, return_tensors="pt").input_ids
    with torch.no_grad():
        logits = model(ids).logits[0, -1]

    print(f"── 1. Temperature: '{PROMPT}' 다음 토큰 상위 3개 확률 ──")
    top = logits.topk(3).indices
    for t in (0.3, 0.7, 1.0, 1.5):
        p = torch.softmax(logits / t, dim=-1)
        row = ", ".join(f"{tok.decode(int(i))!r} {p[i].item():.3f}" for i in top)
        print(f"  T={t:<4}{row}")

    print("\n── 2. Top-k·Top-p로 남는 후보 수(T=0.7) ──")
    p = torch.softmax(logits / 0.7, dim=-1)
    k20, p08 = filter_top_k(p, 20), filter_top_p(p, 0.8)
    print(f"  Top-k 20: {int(k20.sum())}개 / Top-p 0.8: {int(p08.sum())}개 / "
          f"둘 다 적용: {int((k20 & p08).sum())}개")

    print("\n── 3. KV Cache 크기: 계산 ──")
    for label, b in (("BF16·FP16", 2), ("FP32", 4)):
        per = kv_bytes_per_token(cfg, b)
        print(f"  {label:<10}토큰당 {per:,}B = {per / 1024:.0f}KiB, "
              f"32,768토큰 {per * 32768 / 2**20:,.0f}MiB")

    print("\n── 4. KV Cache 크기: 측정(FP32, Greedy 20토큰) ──")
    t0 = time.perf_counter()
    with_cache, cache = greedy(model, ids, 20, use_cache=True)
    t1 = time.perf_counter()
    no_cache, _ = greedy(model, ids, 20, use_cache=False)
    t2 = time.perf_counter()
    stored = cache.get_seq_length()
    measured = sum(layer.keys.numel() * layer.keys.element_size()
                   + layer.values.numel() * layer.values.element_size() for layer in cache.layers)
    print(f"  캐시에 저장된 토큰 {stored}개, 실제 텐서 {measured:,}B "
          f"= 계산값 {kv_bytes_per_token(cfg, 4) * stored:,}B")
    print(f"  K 텐서 하나의 Shape {tuple(cache.layers[0].keys.shape)}  [배치, KV 헤드, 토큰, head_dim]")
    print(f"  생성 결과 같음: {with_cache == no_cache}")
    print(f"  → {tok.decode(with_cache)!r}")
    print(f"  시간: 캐시 사용 {t1 - t0:.2f}초 / 미사용 {t2 - t1:.2f}초(환경마다 다름)")


if __name__ == "__main__":
    main()
