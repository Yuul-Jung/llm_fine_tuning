# 제43장 핵심 실습: 중복·유사 검출 → 그룹 분할 → 누출 검사를 순서대로 실행한다
#
# 실행: python examples/ch43/clean_split.py
#
# 만드는 파일: datasets/fpga_qa/split_v1.json(분할 목록, 저장소에 포함)

import hashlib
import json
import random
import re
import sys
from itertools import combinations
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (HERE.parent, HERE.parent / "ch41"):
    sys.path.insert(0, str(p))

import torch  # noqa: E402

from check_dataset import load  # noqa: E402            (제41장)

SPLIT = ROOT / "datasets" / "fpga_qa" / "split_v1.json"
NGRAM_T, EMB_T, SEED = 0.5, 0.92, 0


def normalize(s: str) -> str:
    """공백·문장부호·대소문자 차이를 지운다."""
    return re.sub(r"[\s\.,!?·()\[\]\"'“”‘’:;]", "", s).lower()


def text_hash(s: str) -> str:
    return hashlib.sha256(normalize(s).encode()).hexdigest()[:12]


def ngrams(s: str, n: int = 3) -> set:
    s = normalize(s)
    return {s[i:i + n] for i in range(max(1, len(s) - n + 1))}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b)


def sentence_embeddings(texts: list) -> torch.Tensor:
    """Qwen2.5의 마지막 hidden state를 토큰 평균(mean pooling)해 문장 벡터로 쓴다."""
    from common.qwen import load_model, load_tokenizer
    tok, model = load_tokenizer(), load_model()
    vecs = []
    with torch.no_grad():
        for t in texts:
            ids = tok(t, return_tensors="pt").input_ids
            h = model(ids, output_hidden_states=True).hidden_states[-1][0]
            vecs.append(h.mean(dim=0))
    v = torch.stack(vecs)
    return torch.nn.functional.normalize(v - v.mean(dim=0), dim=-1)   # 평균을 빼 공통 성분 제거


def groups_from_pairs(ids: list, pairs: list) -> dict:
    """유사 쌍을 이어 같은 그룹으로 묶는다(Union-Find)."""
    parent = {i: i for i in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in pairs:
        parent[find(a)] = find(b)
    return {i: find(i) for i in ids}


def group_split(rows: list, group: dict, per_cat: int = 1, seed: int = SEED) -> dict:
    """분류마다 그룹 단위로 test 1개, validation 1개를 뽑고 나머지는 train(같은 그룹은 같은 분할)."""
    rng = random.Random(seed)
    split = {}
    for cat in sorted({r["category"] for r in rows}):
        gids = sorted({group[r["id"]] for r in rows if r["category"] == cat})
        rng.shuffle(gids)
        for g in gids:
            members = [r["id"] for r in rows if group[r["id"]] == g]
            name = "train"
            for s in ("test", "validation"):
                cat_n = sum(1 for r in rows if r["category"] == cat and split.get(r["id"]) == s)
                if cat_n < per_cat and len(members) <= per_cat:
                    name = s
                    break
            for m in members:
                split.setdefault(m, name)
    return split


def main() -> None:
    rows = load()
    ids = [r["id"] for r in rows]
    text = {r["id"]: r["instruction"] + " " + r["input"] for r in rows}

    print("── 1. 결측·오류·정확한 중복(정규화 후 해시) ──")
    empty = [r["id"] for r in rows if not r["output"].strip() or not r["instruction"].strip()]
    hashes = {}
    for r in rows:
        hashes.setdefault(text_hash(text[r["id"]]), []).append(r["id"])
    dups = [v for v in hashes.values() if len(v) > 1]
    print(f"  결측 {len(empty)}건, 정확한 중복 {len(dups)}건")
    probe = dict(rows[2], id="probe", instruction=rows[2]["instruction"].replace("?", " ?  "))
    print(f"  검출 시험: 띄어쓰기만 바꾼 사본의 해시 같음 {text_hash(probe['instruction']) == text_hash(rows[2]['instruction'])}")

    print(f"\n── 2. 유사 후보: 글자 3-gram Jaccard ≥ {NGRAM_T} ──")
    grams = {i: ngrams(text[i]) for i in ids}
    ng_pairs = [(a, b, jaccard(grams[a], grams[b])) for a, b in combinations(ids, 2)]
    ng_pairs = sorted([p for p in ng_pairs if p[2] >= NGRAM_T], key=lambda p: -p[2])
    for a, b, s in ng_pairs:
        print(f"  {s:.2f}  {a} / {b}  {text[a][:22]} | {text[b][:22]}")
    print(f"  후보 {len(ng_pairs)}쌍")
    copy = text[ids[2]] + " 예를 들어 설명해 주세요."
    print(f"  검출 시험: 요청 한 문장을 덧붙인 사본과의 유사도 {jaccard(grams[ids[2]], ngrams(copy)):.2f}")

    print(f"\n── 3. 유사 후보: 문장 임베딩 코사인 ≥ {EMB_T}(Qwen2.5 평균 풀링) ──")
    emb = sentence_embeddings([text[i] for i in ids])
    sim = emb @ emb.T
    em_pairs = sorted([(ids[a], ids[b], sim[a, b].item()) for a, b in combinations(range(len(ids)), 2)
                       if sim[a, b] >= EMB_T], key=lambda p: -p[2])
    for a, b, s in em_pairs:
        print(f"  {s:.3f}  {a} / {b}  {text[a][:22]} | {text[b][:22]}")
    print(f"  후보 {len(em_pairs)}쌍(사람이 검토해 같은 주제면 같은 그룹으로 묶는다)")

    print("\n── 4. 그룹 분할(분류마다 test 1, validation 1) ──")
    group = groups_from_pairs(ids, [(a, b) for a, b, _ in ng_pairs + em_pairs])
    sizes = sorted((sum(1 for v in group.values() if v == g) for g in set(group.values())), reverse=True)
    print(f"  그룹 {len(set(group.values()))}개, 2개 이상인 그룹 크기 {[s for s in sizes if s > 1]}")
    split = group_split(rows, group)
    for s in ("train", "validation", "test"):
        members = [i for i in ids if split[i] == s]
        print(f"  {s:<11}{len(members):>3}개  " + (str(members) if s != "train" else ""))
    SPLIT.write_text(json.dumps({"seed": SEED, "ngram_threshold": NGRAM_T, "embedding_threshold": EMB_T,
                                 "split": split}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    print("\n── 5. 누출 검사: test·validation 각 문항과 가장 비슷한 train 문항 ──")
    train_ids = [i for i in ids if split[i] == "train"]
    worst = 0.0
    for i in [i for i in ids if split[i] != "train"]:
        best = max(train_ids, key=lambda t: jaccard(grams[i], grams[t]))
        e = sim[ids.index(i), ids.index(best)].item()
        worst = max(worst, jaccard(grams[i], grams[best]))
        print(f"  {i}({split[i][:4]}) ↔ {best}: 3-gram {jaccard(grams[i], grams[best]):.2f}, 임베딩 {e:.3f}")
    print(f"  최대 3-gram 유사도 {worst:.2f} < 기준 {NGRAM_T}: {worst < NGRAM_T}")


if __name__ == "__main__":
    main()
