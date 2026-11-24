# 제30장 핵심 실습: PEFT 라이브러리 LoRA를 직접 구현(제29장)과 같은 조건으로 학습해 비교하고,
#                   Adapter를 저장·재로딩·병합한다
#
# 실행: python examples/ch30/peft_lora.py   (먼저 제25장 pretrain_tiny.py로 사전학습 모델을 만든다)

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE.parent, HERE.parent / "ch24", HERE.parent / "ch25", HERE.parent / "ch29"):
    sys.path.insert(0, str(p))

import torch  # noqa: E402
from peft import LoraConfig, PeftModel, get_peft_model  # noqa: E402
from transformers import Qwen2ForCausalLM  # noqa: E402
from transformers.utils import logging  # noqa: E402

from common.device import select_device  # noqa: E402
from common.fixed_data import load_blocks  # noqa: E402
from common.qwen import load_tokenizer  # noqa: E402
from lora_linear import ALL_LINEAR, apply_lora, trainable_count  # noqa: E402   (제29장)
from trainer import eval_loss, train  # noqa: E402                             (제25장)

logging.disable_progress_bar()
ROOT = HERE.parents[1]
PRETRAINED = ROOT / "models" / "qwen2-tiny-pretrained"
ADAPTER = ROOT / "models" / "qwen2-tiny-bike-lora"
STEPS, LR = 60, 2e-3


def config(targets=("q_proj", "v_proj")) -> LoraConfig:
    """제29장 apply_lora(r=8, alpha=16)와 같은 설정"""
    return LoraConfig(r=8, lora_alpha=16, target_modules=list(targets), lora_dropout=0.0,
                      bias="none", task_type="CAUSAL_LM")


def load_pretrained():
    return Qwen2ForCausalLM.from_pretrained(PRETRAINED, dtype=torch.float32)


def copy_A_from(mine, peft_model) -> int:
    """직접 구현의 A 초기값을 PEFT 쪽에 복사해, 두 구현이 같은 출발점에서 학습하게 한다."""
    n = 0
    for name, module in mine.named_modules():
        if hasattr(module, "lora_A") and isinstance(module.lora_A, torch.nn.Parameter):
            target = peft_model.get_submodule("base_model.model." + name)
            with torch.no_grad():
                target.lora_A["default"].weight.copy_(module.lora_A)
            n += 1
    return n


def main() -> None:
    device, _ = select_device()
    tok = load_tokenizer()
    data = {s: load_blocks(tok, s) for s in ("eval", "domain_train", "domain_eval")}

    print("── 1. LoraConfig + get_peft_model ──")
    peft_model = get_peft_model(load_pretrained(), config())
    names = [(n, tuple(p.shape)) for n, p in peft_model.named_parameters() if p.requires_grad]
    print(f"  학습 파라미터 {trainable_count(peft_model):,}개, 텐서 {len(names)}개")
    for n, s in names[:2]:
        print(f"  {n}  {s}")
    mine = load_pretrained()
    apply_lora(mine, seed=0)
    print(f"  직접 구현(제29장)  {trainable_count(mine):,}개, "
          f"이름 예 model.layers.0.self_attn.q_proj.lora_A  {tuple(mine.model.layers[0].self_attn.q_proj.lora_A.shape)}")

    print("\n── 2. Target Modules별 학습 파라미터 ──")
    for label, targets in (("q·v", ("q_proj", "v_proj")), ("Attention 4종", ALL_LINEAR[:4]),
                           ("7종 전부", ALL_LINEAR)):
        m = get_peft_model(load_pretrained(), config(targets))
        print(f"  {label:<14}{trainable_count(m):>9,}개")

    print(f"\n── 3. 같은 출발점·같은 조건 학습(q·v, r 8, α 16, lr {LR:.0e}, {STEPS}단계) ──")
    print(f"  A 초기값 복사: {copy_A_from(mine, peft_model)}개 계층")
    for label, m in (("직접 구현", mine), ("PEFT", peft_model)):
        train(m, data["domain_train"], {}, steps=STEPS, lr=LR, warmup=5, device=device, log_every=STEPS)
        ppl_d = math.exp(eval_loss(m, data["domain_eval"], device))
        ppl_g = math.exp(eval_loss(m, data["eval"], device))
        print(f"  {label:<10}도메인 PPL {ppl_d:.2f}, 일반 PPL {ppl_g:.2f}")
    b_mine = mine.model.layers[2].self_attn.v_proj.lora_B.detach().cpu()
    b_peft = peft_model.get_submodule("base_model.model.model.layers.2.self_attn.v_proj").lora_B["default"].weight
    print(f"  학습된 B(2번 계층 v_proj) 최대 차이 {(b_mine - b_peft.detach().cpu()).abs().max().item():.1e}")

    print("\n── 4. Adapter 저장 → 재로딩 → 병합 ──")
    peft_model.save_pretrained(ADAPTER)
    files = {f.name: f.stat().st_size for f in ADAPTER.iterdir()}
    print(f"  저장 파일 {files}")
    print(f"  원본 model.safetensors {(PRETRAINED / 'model.safetensors').stat().st_size:,}바이트와 비교")
    probe = data["domain_eval"][:, :32].to(device)
    with torch.no_grad():
        trained = peft_model.eval()(probe).logits
    reloaded = PeftModel.from_pretrained(load_pretrained(), ADAPTER).to(device).eval()
    with torch.no_grad():
        again = reloaded(probe).logits
    merged = reloaded.merge_and_unload()
    with torch.no_grad():
        after_merge = merged(probe).logits
    print(f"  재로딩 후 Logits 완전히 같음: {torch.equal(trained, again)}")
    print(f"  merge_and_unload 후 최대 차이 {(after_merge - trained).abs().max().item():.1e}, "
          f"모델 종류 {type(merged).__name__}")


if __name__ == "__main__":
    main()
