# 제46장 핵심 실습: Dataset·Collator·학습 단계·Validation·Checkpoint를 갖춘 SFT 루프를 직접 구현한다
#
# 실행: python examples/ch46/sft_loop.py   (제25장의 models/qwen2-tiny-pretrained 필요, CPU 약 5분)
#
# 만드는 파일: models/ch46_sft/(Checkpoint), examples/ch46/log.jsonl(학습 기록)

import json
import math
import random
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (HERE.parent, HERE.parent / "ch24", HERE.parent / "ch41", HERE.parent / "ch44", HERE.parent / "ch45"):
    sys.path.insert(0, str(p))

import torch  # noqa: E402
from torch.utils.data import DataLoader, Dataset, RandomSampler  # noqa: E402

from check_dataset import load  # noqa: E402                 (제41장)
from common.device import select_device  # noqa: E402
from label_mask import IGNORE, build_example, pad_batch  # noqa: E402   (제45장)
from optim_memory import warmup_cosine  # noqa: E402          (제24장)

TINY = ROOT / "models" / "qwen2-tiny-pretrained"
CKPT = ROOT / "models" / "ch46_sft"
LOG = HERE / "log.jsonl"


def split_rows(name: str) -> list:
    split = json.loads((ROOT / "datasets/fpga_qa/split_v1.json").read_text(encoding="utf-8"))["split"]
    return [r for r in load() if split[r["id"]] == name]


class SFTDataset(Dataset):
    """샘플마다 제45장의 input_ids·attention_mask·labels를 미리 만들어 둔다"""

    def __init__(self, tok, rows: list, max_len: int = 256):
        self.items = [build_example(tok, r, max_len) for r in rows]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


class Collator:
    """길이가 다른 샘플을 가장 긴 샘플에 맞춰 오른쪽 PAD로 묶는다"""

    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, examples: list) -> dict:
        return pad_batch(examples, self.pad_id)


def loader(ds, batch: int, seed: int, epoch: int) -> DataLoader:
    g = torch.Generator().manual_seed(seed * 1000 + epoch)      # 에폭마다 다르지만 재현되는 순서
    return DataLoader(ds, batch_size=batch, sampler=RandomSampler(ds, generator=g),
                      collate_fn=Collator(ds.pad_id), drop_last=True)


@torch.no_grad()
def validate(model, ds) -> dict:
    """학습과 같은 Label 정책. 토큰 수로 가중 평균해야 샘플 길이에 따라 치우치지 않는다."""
    was = model.training
    model.eval()
    total, n = 0.0, 0
    for e in ds.items:
        dev = next(model.parameters()).device
        ids, lab = torch.tensor([e["input_ids"]], device=dev), torch.tensor([e["labels"]], device=dev)
        k = int((lab[0, 1:] != IGNORE).sum())
        total += model(input_ids=ids, labels=lab).loss.item() * k
        n += k
    model.train(was)
    return {"val_loss": total / n, "val_ppl": math.exp(total / n), "val_tokens": n}


def save_checkpoint(path: Path, model, opt, sched, state: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    model.save_pretrained(tmp)
    torch.save({"optimizer": opt.state_dict(), "scheduler": sched.state_dict(), **state}, tmp / "trainer_state.pt")
    shutil.rmtree(path, ignore_errors=True)
    tmp.rename(path)                     # 저장 도중 멈춰도 이전 Checkpoint가 남도록 다 쓴 뒤에 바꾼다


def train_sft(model, train_ds, val_ds, *, steps: int, batch: int = 4, accum: int = 2, lr: float = 1e-3,
              warmup: int = 6, clip: float = 1.0, seed: int = 0, log_every: int = 5,
              ckpt: Path = None, save_at: tuple = (), resume: bool = False, stop_at: int = None,
              log_path: Path = None) -> list:
    torch.manual_seed(seed)
    random.seed(seed)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.1)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: warmup_cosine(s, warmup, steps))
    step, consumed = 0, 0                                         # consumed: 지금까지 쓴 묶음 수
    if resume:
        st = torch.load(ckpt / "trainer_state.pt", weights_only=False)
        opt.load_state_dict(st["optimizer"])
        sched.load_state_dict(st["scheduler"])
        step, consumed = st["step"], st["consumed"]
    per_epoch = len(train_ds) // batch
    history = []
    log = open(log_path, "a", encoding="utf-8") if log_path else None
    batches = None
    while step < (stop_at or steps):
        total, tokens = 0.0, 0
        for _ in range(accum):
            epoch, pos = divmod(consumed, per_epoch)
            if batches is None or pos == 0:
                batches = iter(loader(train_ds, batch, seed, epoch))
                for _ in range(pos):                              # 재개: 이번 에폭에서 쓴 묶음 건너뛰기
                    next(batches)
            b = {k: v.to(next(model.parameters()).device) for k, v in next(batches).items()}
            consumed += 1
            out = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"], labels=b["labels"])
            (out.loss / accum).backward()
            total += out.loss.item() / accum
            tokens += int((b["labels"][:, 1:] != IGNORE).sum())
        grad_norm = torch.nn.utils.clip_grad_norm_(params, clip).item()
        if not math.isfinite(grad_norm):
            raise RuntimeError(f"{step + 1}단계에서 Gradient가 NaN·Inf")
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        if step % log_every == 0 or step == steps:
            row = {"step": step, "epoch": round(consumed / per_epoch, 2), "lr": opt.param_groups[0]["lr"],
                   "train_loss": total, "grad_norm": grad_norm, "tokens": tokens, **validate(model, val_ds)}
            history.append(row)
            if log:
                log.write(json.dumps(row) + "\n")
        if ckpt and step in save_at:
            save_checkpoint(ckpt, model, opt, sched, {"step": step, "consumed": consumed, "seed": seed})
    if log:
        log.close()
    return history


def setup(model_dir: Path = TINY):
    """토크나이저는 언제나 원래 모델 폴더에서(Checkpoint에는 가중치와 학습 상태만 저장)"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TINY)
    torch.manual_seed(0)
    device, _ = select_device(verbose=False)                 # Qwen2-Tiny는 GT 1030에도 들어간다
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32).to(device)
    tr, va = SFTDataset(tok, split_rows("train")), SFTDataset(tok, split_rows("validation"))
    tr.pad_id = va.pad_id = tok.pad_token_id
    return tok, model, tr, va


def main() -> None:
    STEPS = 30
    tok, model, tr, va = setup()
    print("── 1. Dataset · DataLoader · Collator ──")
    b = next(iter(loader(tr, 4, 0, 0)))
    print(f"  train {len(tr)}개, validation {len(va)}개, 묶음 4 × 누적 2 = 단계당 8샘플, 에폭당 {len(tr) // 4}묶음")
    print(f"  첫 묶음 모양 {tuple(b['input_ids'].shape)}, PAD {int((b['attention_mask'] == 0).sum())}자리, "
          f"학습 대상 {int((b['labels'] != IGNORE).sum())}토큰")

    print(f"\n── 2·3. 학습 {STEPS}단계(Qwen2-Tiny, lr 1e-3, Warmup 6 + Cosine), 5단계마다 Validation ──")
    v0 = validate(model, va)
    print(f"  시작 전 val loss {v0['val_loss']:.3f}, val PPL {v0['val_ppl']:.1f}(학습 대상 {v0['val_tokens']}토큰)")
    LOG.unlink(missing_ok=True)
    full = train_sft(model, tr, va, steps=STEPS, ckpt=CKPT, save_at=(15,), log_path=LOG)
    print("  단계  에폭   lr        train   val     val PPL  grad")
    for h in full:
        print(f"  {h['step']:>4}  {h['epoch']:>4}  {h['lr']:.2e}  {h['train_loss']:.3f}  {h['val_loss']:.3f}"
              f"  {h['val_ppl']:>7.1f}  {h['grad_norm']:.2f}")

    print("\n── 4. Checkpoint 재개: 15단계 Checkpoint에서 다시 불러와 30단계까지 ──")
    _, model2, _, _ = setup(CKPT)
    resumed = train_sft(model2, tr, va, steps=STEPS, ckpt=CKPT, resume=True)
    for a, r in zip([h for h in full if h["step"] > 15], resumed):
        print(f"  {a['step']:>4}단계 train 연속 {a['train_loss']:.6f} / 재개 {r['train_loss']:.6f}   "
              f"val 연속 {a['val_loss']:.6f} / 재개 {r['val_loss']:.6f}")
    same = all(abs(a["train_loss"] - r["train_loss"]) < 1e-5 for a, r in zip(full[3:], resumed))
    print(f"  재개 결과가 연속 학습과 일치: {same}")
    _, model3, _, _ = setup(CKPT)
    weights_only = train_sft(model3, tr, va, steps=STEPS - 15, warmup=6)
    print("  비교: 가중치만 불러오고 Optimizer·Scheduler를 새로 시작하면(같은 15단계 더)")
    for a, w in zip([h for h in full if h["step"] > 15], weights_only):
        print(f"  {a['step']:>4}단계 train 연속 {a['train_loss']:.3f} / 가중치만 {w['train_loss']:.3f}   "
              f"val 연속 {a['val_loss']:.3f} / 가중치만 {w['val_loss']:.3f}   lr 연속 {a['lr']:.1e} / 가중치만 {w['lr']:.1e}")
    print(f"  기록 파일 {LOG.relative_to(ROOT)}: {len(LOG.read_text().splitlines())}줄")


if __name__ == "__main__":
    main()
