# 제3·4부의 모든 학습 비교 실험이 함께 쓰는 고정 토큰 데이터
#
# 문장 파일(datasets/fixed/*.txt) → Qwen2.5 토크나이저 → 문장마다 끝에 <|endoftext|> →
# 하나로 이어 붙임 → 길이 SEQ_LEN의 블록으로 자름(남는 꼬리는 버림)
# 같은 파일·같은 토크나이저면 언제나 같은 토큰이 나오며, 지문(fingerprint)으로 확인한다.

import hashlib
from pathlib import Path

import torch

DATA_DIR = Path(__file__).resolve().parents[2] / "datasets" / "fixed"
SEQ_LEN = 128                                   # GT 1030 기준 학습 시퀀스 길이(제25장)


def read_lines(split: str) -> list:
    """split: "train" 또는 "eval". 빈 줄은 건너뛴다."""
    text = (DATA_DIR / f"{split}.txt").read_text(encoding="utf-8")
    return [line.strip() for line in text.splitlines() if line.strip()]


def token_stream(tok, split: str) -> list:
    ids = []
    for line in read_lines(split):
        ids += tok(line).input_ids + [tok.convert_tokens_to_ids("<|endoftext|>")]
    return ids


def make_blocks(ids: list, seq_len: int = SEQ_LEN) -> torch.Tensor:
    n = len(ids) // seq_len
    return torch.tensor(ids[: n * seq_len], dtype=torch.long).view(n, seq_len)


def load_blocks(tok, split: str, seq_len: int = SEQ_LEN) -> torch.Tensor:
    """[블록 수, seq_len] 정수 텐서"""
    return make_blocks(token_stream(tok, split), seq_len)


def fingerprint(blocks: torch.Tensor) -> str:
    """토큰 값과 모양이 하나라도 다르면 달라지는 16자리 지문"""
    data = str(tuple(blocks.shape)).encode() + blocks.numpy().astype("<i8").tobytes()
    return hashlib.sha256(data).hexdigest()[:16]
