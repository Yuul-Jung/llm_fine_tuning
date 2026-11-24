# 제10장 핵심 실습: 실제 Qwen2.5 Embedding으로 문장을 벡터로 바꾸고 단어 유사도 비교
#
# 실행: python examples/ch10/qwen_embedding.py
# 준비: 모델 파일과 토크나이저를 내려받는다(제05·08장에서 받았다면 재사용).

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from safetensors import safe_open  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from common.models import QWEN_ID, QWEN_REVISION, qwen_file  # noqa: E402

TOTAL_PARAMS = 494_032_768          # 제02장에서 계산한 Qwen2.5-0.5B 전체 파라미터 수
PAIRS = [(" king", " queen"), (" Monday", " Tuesday"), (" man", " woman"),
         (" cat", " dog"), (" king", " banana"), (" apple", " Monday")]


def load_embedding(local_only: bool = False):
    """Embedding 표와 파일에 저장된 텐서 이름 목록을 돌려준다."""
    path = qwen_file("model.safetensors", local_only=local_only)
    with safe_open(path, framework="pt") as f:
        names = list(f.keys())
        table = f.get_tensor("model.embed_tokens.weight").float()   # BF16 → FP32
    return table, names


def load_tokenizer(local_only: bool = False):
    return AutoTokenizer.from_pretrained(QWEN_ID, revision=QWEN_REVISION,
                                         local_files_only=local_only)


def main() -> None:
    table, names = load_embedding()
    tok = load_tokenizer()

    print("── 1. Embedding 표 ──")
    print(f"모양 {tuple(table.shape)} = [어휘 행 수, hidden]")
    print(f"파라미터 {table.numel():,}개 = 전체의 {table.numel() / TOTAL_PARAMS:.1%}")
    print(f"파일 속 텐서 {len(names)}개, lm_head 저장 여부: {any('lm_head' in n for n in names)}")

    print("\n── 2. 문장 → Token ID → Embedding Vector ──")
    text = "언어 모델은 다음 토큰을 예측한다."
    ids = torch.tensor([tok.encode(text)])                 # [batch 1, seq]
    vectors = table[ids]                                    # Lookup
    print(f"문장 {len(text)}글자 → Token ID {tuple(ids.shape)} → 벡터 {tuple(vectors.shape)}")
    print(f"첫 토큰 '{tok.decode(ids[0, :1])}' 벡터의 앞 5칸: "
          f"{[round(v, 4) for v in vectors[0, 0, :5].tolist()]}")

    print("\n── 3. 단어 벡터의 코사인 유사도 ──")
    for a, b in PAIRS:
        ia, ib = tok.encode(a), tok.encode(b)
        assert len(ia) == len(ib) == 1, "한 토큰짜리 단어만 비교한다"
        sim = F.cosine_similarity(table[ia[0]], table[ib[0]], dim=0).item()
        print(f"  {a.strip():>8} · {b.strip():<8} {sim:+.3f}")

    print("\n── 4. 어휘 크기와 Logits 메모리(FP32, Batch 1) ──")
    for seq in (16, 256, 1024):
        mib = seq * table.shape[0] * 4 / 2**20
        print(f"  시퀀스 {seq:>4}: Logits [{seq}, {table.shape[0]}] = {mib:7.1f} MiB")


if __name__ == "__main__":
    main()
