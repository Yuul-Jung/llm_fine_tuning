# 제23장 실습 2: Qwen2-Tiny에서 역전파 한 번이 모든 가중치에 Gradient를 남기는 모습 관찰
#
# 실행: python examples/ch23/grad_flow.py

import re
import sys
from collections import OrderedDict
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE.parent, HERE.parent / "ch19"):
    sys.path.insert(0, str(p))

import torch  # noqa: E402

from common.device import select_device  # noqa: E402
from common.fixed_data import load_blocks  # noqa: E402
from common.qwen import load_tokenizer  # noqa: E402
from hf_compat import to_hf  # noqa: E402            (제19장)
from tiny_model import Qwen2Tiny, init_weights  # noqa: E402


def group_of(name: str) -> str:
    """파라미터 이름을 '어느 계층의 어느 부품'으로 묶는다."""
    m = re.match(r"model\.layers\.(\d+)\.(self_attn|mlp|input_layernorm|post_attention_layernorm)", name)
    if m:
        part = {"self_attn": "Attention", "mlp": "FFN"}.get(m.group(2), "RMSNorm")
        return f"Block {m.group(1)} {part}"
    return {"model.embed_tokens.weight": "Embedding(= LM Head)", "model.norm.weight": "Final Norm"}[name]


def grad_table(model) -> "OrderedDict[str, float]":
    sq = OrderedDict()
    for name, p in model.named_parameters():               # 공유 텐서는 한 번만 나온다
        g = group_of(name)
        sq[g] = sq.get(g, 0.0) + p.grad.float().pow(2).sum().item()
    return OrderedDict((k, v ** 0.5) for k, v in sq.items())


def main() -> None:
    device, _ = select_device()
    tok = load_tokenizer()
    block = load_blocks(tok, "train")[:1].to(device)
    model = to_hf(init_weights(Qwen2Tiny(), seed=0)).to(device).train()

    loss = model(input_ids=block, labels=block).loss
    print(f"── 1. 순전파 손실 {loss.item():.4f} → loss.backward() 한 번 ──")
    print(f"  역전파 전 .grad: {model.model.layers[0].mlp.up_proj.weight.grad}")
    loss.backward()

    print("\n── 2. 부품별 Gradient 크기(L2 노름) ──")
    table = grad_table(model)
    top = max(table.values())
    for g, v in table.items():
        print(f"  {g:<20}{v:9.4f}  {'█' * max(1, round(20 * v / top))}")

    print("\n── 3. Embedding(= LM Head)이 받은 Gradient ──")
    emb = model.model.embed_tokens.weight.grad
    rows = (emb.abs().sum(dim=1) > 0).sum().item()
    print(f"  Gradient가 0이 아닌 행 {rows:,} / {emb.shape[0]:,}  (입력 토큰 종류 {block.unique().numel()}개)")
    alone = torch.nn.Embedding(*emb.shape).to(device)        # 공유하지 않는 표라면?
    alone(block).sum().backward()
    rows_alone = (alone.weight.grad.abs().sum(dim=1) > 0).sum().item()
    print(f"  공유하지 않은 Embedding 표만 따로 역전파하면: {rows_alone}행")

    n = sum(p.numel() for p in model.parameters())
    print(f"\n── 4. Gradient 메모리: 파라미터 {n:,}개 × 4바이트 = {n * 4 / 2**20:.1f}MiB(가중치와 같은 크기) ──")


if __name__ == "__main__":
    main()
