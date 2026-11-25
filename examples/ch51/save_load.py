# 제51장 핵심 실습: Checkpoint·Safetensors·Adapter 구조를 열어 보고, Adapter를 저장·재로딩해 출력을 비교한다
#
# 실행: python examples/ch51/save_load.py   (제50장의 확정 Adapter 필요, CPU 약 1분)
#
# 만드는 파일: models/fpga_lora_v1/(배포용 Adapter), examples/ch51/manifest.json(대응 기록)

import hashlib
import json
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))

import torch  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

from common.device import select_device  # noqa: E402
from common.models import QWEN_ID, QWEN_REVISION  # noqa: E402

CKPT = ROOT / "models" / "ch50" / "원본" / "checkpoint-15"     # 제50장 확정 모델(원본-15)
FINAL = ROOT / "models" / "fpga_lora_v1"
MANIFEST = HERE / "manifest.json"
MiB = 2 ** 20


def read_header(path: Path) -> tuple:
    """Safetensors = [8바이트 길이(little-endian)][JSON 머리말][텐서 바이트를 이어 붙인 본문]"""
    raw = path.read_bytes()
    n = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8:8 + n])
    return n, header, raw[8 + n:]


def tensor_by_hand(header: dict, body: bytes, name: str) -> torch.Tensor:
    """머리말의 자료형·모양·위치만으로 텐서 하나를 직접 꺼낸다"""
    info = header[name]
    start, end = info["data_offsets"]
    assert info["dtype"] == "F32"
    return torch.frombuffer(bytearray(body[start:end]), dtype=torch.float32).reshape(info["shape"])


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_base(device: str):
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(QWEN_ID, revision=QWEN_REVISION,
                                                dtype=torch.float32).to(device).eval()


def logits_of(model, tok, device: str) -> torch.Tensor:
    ids = tok("FPGA의 LUT는 어떤 역할을 하나요?", return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        return model(ids).logits[0].float().cpu()


def main() -> None:
    from peft import PeftModel
    from transformers import AutoTokenizer
    device, _ = select_device()                    # 병합 전 0.5B FP32 Adapter 모델: GT 1030에서는 CPU 권장
    print("── 1. 학습 Checkpoint에 들어 있는 것(제50장 원본-15) ──")
    roles = {"adapter_model.safetensors": "LoRA 가중치(A, B)", "optimizer.pt": "AdamW 1·2차 모멘트(pickle)",
             "scheduler.pt": "학습률 Scheduler 상태(pickle)", "rng_state.pth": "난수 상태(pickle)",
             "trainer_state.json": "단계·기록(log_history)", "training_args.bin": "학습 설정(pickle)",
             "adapter_config.json": "LoRA 설정과 원본 모델 이름"}
    for name, role in roles.items():
        size = (CKPT / name).stat().st_size
        text = f"{size / MiB:.2f}MiB" if size >= MiB else f"{size / 1024:.1f}KiB"
        print(f"  {name:<27}{text:>10}  {role}")
    ratio = (CKPT / "optimizer.pt").stat().st_size / (CKPT / "adapter_model.safetensors").stat().st_size
    print(f"  Optimizer 상태 / LoRA 가중치 = {ratio:.2f}배(모멘트 2개)")

    print("\n── 2. Safetensors 구조 ──")
    n, header, body = read_header(CKPT / "adapter_model.safetensors")
    names = sorted(k for k in header if k != "__metadata__")
    print(f"  머리말 {n}바이트(JSON), 텐서 {len(names)}개, 본문 {len(body) / MiB:.2f}MiB")
    for k in names[:2]:
        print(f"  {k}\n    {header[k]}")
    lib = load_file(CKPT / "adapter_model.safetensors")
    same = all(torch.equal(tensor_by_hand(header, body, k), lib[k]) for k in names)
    print(f"  머리말만으로 직접 꺼낸 텐서 = 라이브러리로 읽은 텐서: {same}")

    print("\n── 3. Adapter 파일과 원본 모델의 연결 ──")
    cfg = json.loads((CKPT / "adapter_config.json").read_text())
    print(f"  base_model_name_or_path {cfg['base_model_name_or_path']}, revision {cfg['revision']}, "
          f"r {cfg['r']}, alpha {cfg['lora_alpha']}")
    print(f"  target_modules {sorted(cfg['target_modules'])}")
    tok = AutoTokenizer.from_pretrained(QWEN_ID, revision=QWEN_REVISION)
    base = load_base(device)
    modules = {n for n, _ in base.named_modules()}
    mapped = {k.removeprefix("base_model.model.").rsplit(".lora_", 1)[0] for k in names}
    print(f"  Adapter 텐서 {len(names)}개 → 원본 모듈 {len(mapped)}개(모두 원본에 있음 {mapped <= modules})")
    print(f"  예: {names[0]}\n    → 원본의 {sorted(mapped)[0]}에 A·B 한 쌍이 붙는다")
    ref = logits_of(base, tok, device)

    print("\n── 4. 저장·재로딩과 Revision 기록 ──")
    model = PeftModel.from_pretrained(base, CKPT).eval()
    out1 = logits_of(model, tok, device)
    model.peft_config["default"].revision = QWEN_REVISION         # 원본 Revision을 Adapter 설정에 남긴다
    model.save_pretrained(FINAL)
    del model, base
    base2 = load_base(device)
    model2 = PeftModel.from_pretrained(base2, FINAL).eval()
    out2 = logits_of(model2, tok, device)
    w1, w2 = load_file(CKPT / "adapter_model.safetensors"), load_file(FINAL / "adapter_model.safetensors")
    print(f"  저장 파일: {sorted(p.name for p in FINAL.iterdir())}")
    print(f"  가중치 완전 일치: {w1.keys() == w2.keys() and all(torch.equal(w1[k], w2[k]) for k in w1)}")
    print(f"  재로딩 전후 Logits 최대 차이 {(out1 - out2).abs().max().item():.2e}")
    print(f"  Adapter 적용 전후 Logits 최대 차이 {(ref - out1).abs().max().item():.2f}(Adapter가 실제로 작동)")
    saved = json.loads((FINAL / "adapter_config.json").read_text())
    manifest = {
        "adapter": "models/fpga_lora_v1",
        "base_model": QWEN_ID, "base_revision": QWEN_REVISION,
        "adapter_sha256": sha256(FINAL / "adapter_model.safetensors"),
        "source": "제50장 원본-15(train 38, 15단계, lr 2e-4, 모든 Linear r 8)",
        "data": {"dataset": "datasets/fpga_qa/fpga_qa.json", "split": "split_v1.json"},
        "criteria_version": 2, "final_test": "3/4 합격(핵심어 불합격), 배포 후보 불채택, 실습용",
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"  저장된 설정의 revision {saved['revision'][:12]}…, 대응 기록 {MANIFEST.relative_to(ROOT)}")
    print(f"  Adapter 해시 {manifest['adapter_sha256'][:16]}…")


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
