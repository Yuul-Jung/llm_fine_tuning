# 제29장 핵심 실습: 사전학습 Qwen2-Tiny의 q_proj·v_proj에 직접 만든 LoRA를 붙여 학습하고 병합한다
#
# 실행: python examples/ch29/lora_tiny.py   (먼저 제25장 pretrain_tiny.py로 사전학습 모델을 만든다)

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent, HERE.parent / "ch24", HERE.parent / "ch25", HERE.parent / "ch26"):
    sys.path.insert(0, str(p))

import torch  # noqa: E402
from transformers import Qwen2ForCausalLM  # noqa: E402
from transformers.utils import logging  # noqa: E402

from common.device import select_device  # noqa: E402
from common.fixed_data import load_blocks  # noqa: E402
from common.qwen import load_tokenizer  # noqa: E402
from lora_linear import ALL_LINEAR, LoRALinear, apply_lora, merge_lora, trainable_count  # noqa: E402
from partial_ft import snapshot  # noqa: E402         (제26장)
from trainer import eval_loss, train  # noqa: E402     (제25장)

logging.disable_progress_bar()
PRETRAINED = HERE.parents[1] / "models" / "qwen2-tiny-pretrained"


def ppl(model, blocks, device) -> float:
    return math.exp(eval_loss(model, blocks, device))


def main() -> None:
    device, _ = select_device()
    tok = load_tokenizer()
    data = {s: load_blocks(tok, s) for s in ("eval", "domain_train", "domain_eval")}

    print("── 1. LoRALinear 붙이기(r = 8, α = 16) ──")
    model = Qwen2ForCausalLM.from_pretrained(PRETRAINED, dtype=torch.float32).to(device)
    probe = data["domain_eval"][:, :32].to(device)
    with torch.no_grad():
        before_logits = model(probe).logits
    replaced = apply_lora(model)
    with torch.no_grad():
        same = torch.equal(model(probe).logits, before_logits)
    total = sum(p.numel() for p in model.parameters())
    print(f"  바꾼 계층 {len(replaced)}개: {replaced[0]}, {replaced[1]}, …")
    print(f"  학습 파라미터 {trainable_count(model):,} / {total:,} ({trainable_count(model) / total:.3%})")
    print(f"  붙인 직후 Logits가 원본과 완전히 같음(B = 0): {same}")

    print("\n── 2. 역전파: Gradient는 A·B에만 ──")
    x = data["domain_train"][:1].to(device)
    model(input_ids=x, labels=x).loss.backward()
    lora = model.model.layers[0].self_attn.q_proj
    print(f"  원본 q_proj.weight.grad: {lora.base.weight.grad}")
    print(f"  lora_A.grad 0 아닌 원소 {torch.count_nonzero(lora.lora_A.grad).item()}개, "
          f"lora_B.grad 0 아닌 원소 {torch.count_nonzero(lora.lora_B.grad).item()}개")
    model.zero_grad(set_to_none=True)

    print("\n── 3. 대상과 학습률별 도메인 학습(60단계, 제26장과 같은 데이터·단계) ──")
    base_ppl = (ppl(model, data["domain_eval"], device), ppl(model, data["eval"], device))
    print(f"  학습 전{'':<18}도메인 PPL {base_ppl[0]:6.1f}, 일반 PPL {base_ppl[1]:6.1f}")
    results = {}
    for label, targets, lr in (("q·v, lr 3e-4", ("q_proj", "v_proj"), 3e-4),
                               ("q·v, lr 2e-3", ("q_proj", "v_proj"), 2e-3),
                               ("7종 전부, lr 2e-3", ALL_LINEAR, 2e-3)):
        m = Qwen2ForCausalLM.from_pretrained(PRETRAINED, dtype=torch.float32)
        apply_lora(m, targets)
        frozen = {n: p for n, p in snapshot(m).items() if "lora_" not in n}
        train(m, data["domain_train"], {}, steps=60, lr=lr, warmup=5, device=device, log_every=60)
        kept = all(torch.equal(p, snapshot(m)[n].cpu()) for n, p in frozen.items())
        results[label] = m
        print(f"  {label:<22}도메인 PPL {ppl(m, data['domain_eval'], device):6.1f}, "
              f"일반 PPL {ppl(m, data['eval'], device):6.1f}, 학습 {trainable_count(m):>7,}개, "
              f"원본 그대로 {kept}")

    print("\n── 4. 병합: LoRALinear → 보통의 Linear ──")
    m = results["7종 전부, lr 2e-3"].eval()
    with torch.no_grad():
        lora_logits = m(probe).logits
    n = merge_lora(m)
    with torch.no_grad():
        merged_logits = m(probe).logits
    left = sum(isinstance(mod, LoRALinear) for mod in m.modules())
    print(f"  병합한 계층 {n}개, 남은 LoRALinear {left}개, 파라미터 {sum(p.numel() for p in m.parameters()):,}개")
    print(f"  병합 전후 Logits 최대 차이 {(lora_logits - merged_logits).abs().max().item():.1e}")
    print(f"  병합 모델 도메인 PPL {ppl(m, data['domain_eval'], device):.1f}")


if __name__ == "__main__":
    main()
