# 제60장 핵심 실습: 데이터부터 API까지 전 과정의 산출물을 한 장의 기록으로 묶고, 재현성을 다시 확인한다
#
# 실행: python examples/ch60/project.py   (앞 장의 산출물 필요, CPU 약 3분)
#
# 만드는 파일: examples/ch60/project.json(프로젝트 기록: 판 번호·해시·Seed·설정·결과·미검증 항목)

import hashlib
import json
import math
import platform
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (HERE.parent, HERE.parent / "ch43", HERE.parent / "ch44", HERE.parent / "ch46", HERE.parent / "ch57"):
    sys.path.insert(0, str(p))

import torch  # noqa: E402

from common.device import select_device  # noqa: E402
from common.models import QWEN_ID, QWEN_REVISION  # noqa: E402

OUT = HERE / "project.json"
FILES = {                                       # 저장소에 들어 있는 입력(데이터·기준)
    "데이터 원본": "datasets/fpga_qa/fpga_qa.json",
    "분할": "datasets/fpga_qa/split_v1.json",
    "보강": "datasets/fpga_qa/augment_v1.json",
    "평가 기준": "configs/eval_criteria.yaml",
}
ARTIFACTS = {                                   # 저장소 밖(models/)의 산출물
    "배포용 Adapter": "models/fpga_lora_v1/adapter_model.safetensors",
    "병합 모델": "models/fpga_merged_v1/model.safetensors",
    "배포 GGUF(Q8_0)": "models/gguf/fpga-q8_0.gguf",
    "Modelfile": "models/gguf/Modelfile",
}
STAGES = [                                      # (단계, 장, 상태)
    ("요구사항·데이터·기준", "제41~43장", "완료"),
    ("학습(0.5B FP32 LoRA, Tiny 비교)", "제46~48장", "완료"),
    ("학습(0.5B QLoRA, GT 1030 경로)", "제49장", "CPU 경로 완료, GT 1030 미검증"),
    ("오답 분석·최종 평가", "제50장", "완료(배포 후보 불채택)"),
    ("저장·병합", "제51·52장", "완료"),
    ("GGUF 변환·양자화", "제54·55장", "완료(Q8_0 선정), llama.cpp 실행·속도 미검증"),
    ("Ollama 배포", "제56장", "Modelfile 완료, Ollama 등록·응답·GPU 적재 미검증"),
    ("API", "제57장", "완료(교재 서버), 실제 Ollama 미검증"),
    ("장치별 실행", "제58장", "계산·CPU 실측 완료, GT 1030·Raspberry Pi 미검증"),
]


def sha(path: Path, n: int = 16) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:n]


def versions() -> dict:
    import datasets
    import peft
    import transformers
    import trl
    return {"python": platform.python_version(), "torch": torch.__version__,
            "transformers": transformers.__version__, "peft": peft.__version__, "trl": trl.__version__,
            "datasets": datasets.__version__}


def repeat_training_check() -> tuple:
    """같은 Seed로 Qwen2-Tiny SFT를 두 번 5단계씩 학습해 손실이 똑같은지 본다(제46장 루프)"""
    from sft_loop import setup, train_sft
    runs = []
    for _ in range(2):
        _, model, tr, va = setup()
        runs.append([round(h["train_loss"], 6) for h in train_sft(model, tr, va, steps=5, log_every=1)])
    return runs[0], runs[0] == runs[1]


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sft_loop import SFTDataset, split_rows, validate
    device, _ = select_device()
    final = json.loads((ROOT / "examples/ch50/final_test.json").read_text(encoding="utf-8"))
    merged_rec = json.loads((ROOT / "examples/ch52/merge.json").read_text())

    print("── 1. 입력과 산출물의 지문(SHA-256 앞 16자리) ──")
    hashes = {k: sha(ROOT / v) for k, v in {**FILES, **ARTIFACTS}.items()}
    for k, v in hashes.items():
        print(f"  {k}: {v}")
    manifest = json.loads((ROOT / "examples/ch51/manifest.json").read_text(encoding="utf-8"))
    same = manifest["adapter_sha256"].startswith(hashes["배포용 Adapter"])
    print(f"  배포용 Adapter 해시 = 제51장 대응 기록: {same}")

    print("\n── 2. 단계별 상태 ──")
    for stage, ch, state in STAGES:
        print(f"  {stage}({ch}): {state}")

    print("\n── 3. 재현 확인 ──")
    losses, same_loss = repeat_training_check()
    print(f"  Seed 0으로 두 번 학습(Tiny 5단계): 손실 {losses} → 두 번 같음 {same_loss}")
    tok = AutoTokenizer.from_pretrained(ROOT / "models" / "fpga_merged_v1")
    model = AutoModelForCausalLM.from_pretrained(ROOT / "models" / "fpga_merged_v1",
                                                 dtype=torch.float32).to(device).eval()
    ppl = validate(model, SFTDataset(tok, split_rows("validation")))["val_ppl"]
    print(f"  병합 모델의 도메인 PPL 다시 계산 {ppl:.4f} / 제52장 기록 {merged_rec['final']['ppl_after']:.4f}"
          f" → 같음 {math.isclose(ppl, merged_rec['final']['ppl_after'], rel_tol=1e-6)}")
    from local_api import Client, HFBackend, make_server
    from chat_template import SYSTEM
    backend = HFBackend.__new__(HFBackend)                        # 이미 불러온 모델을 그대로 쓴다
    backend.torch, backend.name, backend.n, backend.tok, backend.model = torch, "fpga-qwen", 40, tok, model
    srv = make_server(backend)
    ans = Client(f"http://127.0.0.1:{srv.server_address[1]}").generate("임계 경로란 무엇인가요?", system=SYSTEM)
    srv.shutdown()
    print(f"  API 연결 확인: {ans[:40]}…")

    print("\n── 4. 최종 결과(제50장, test 1회) ──")
    v = final["result"]
    hits = sum(v["keyword_hit"].values()) / len(v["keyword_hit"])
    fmts = sum(v["format_ok"].values()) / len(v["format_ok"])
    print(f"  확정 모델 {final['chosen']}: 핵심어 {hits:.2f}, 형식 {fmts:.2f}, 도메인 PPL {v['domain_ppl']:.2f}, "
          f"일반 PPL {v['general_ppl']:.2f} → 배포 후보 불채택, 실습 완료")

    record = {
        "project": "FPGA 질의응답 로컬 LLM", "base_model": QWEN_ID, "base_revision": QWEN_REVISION,
        "hashes": hashes, "versions": versions(), "seed": 0, "device": device,
        "training": {"method": "LoRA r 8, alpha 16, 모든 Linear", "lr": 2e-4, "steps": 15,
                     "batch": "4 × 누적 2", "max_length": 256, "loss": "Completion-only"},
        "final_test": {"chosen": final["chosen"], "keyword": hits, "format": fmts,
                       "domain_ppl": v["domain_ppl"], "general_ppl": v["general_ppl"], "adopted": False},
        "stages": [{"stage": s, "chapters": c, "state": t} for s, c, t in STAGES],
        "reproduced": {"training_loss_same": same_loss, "merged_ppl": ppl},
    }
    OUT.write_text(json.dumps(record, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"\n  기록: {OUT.relative_to(ROOT)}, 라이브러리 {record['versions']}")


if __name__ == "__main__":
    main()
