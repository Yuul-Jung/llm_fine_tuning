# 제10장 검증 테스트
#
# 실행: python -m pytest examples/ch10 -q

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from embedding_basics import cosine, lookup, one_hot_matmul  # noqa: E402


def test_one_hot_equals_lookup():
    torch.manual_seed(0)
    table = torch.randn(50, 8)
    ids = torch.randint(0, 50, (3, 7))
    torch.testing.assert_close(one_hot_matmul(ids, table), lookup(ids, table))


def test_nn_embedding_is_lookup():
    emb = nn.Embedding(10, 4)
    ids = torch.tensor([[1, 5, 7], [2, 2, 9]])
    assert emb(ids).shape == (2, 3, 4)
    assert torch.equal(emb(ids), emb.weight[ids])


def test_cosine_values():
    u = torch.tensor([1.0, 0.0])
    assert cosine(u, torch.tensor([3.0, 0.0])) == pytest.approx(1.0)     # 같은 방향
    assert cosine(u, torch.tensor([0.0, 2.0])) == pytest.approx(0.0)     # 직각
    assert cosine(u, torch.tensor([-1.0, 0.0])) == pytest.approx(-1.0)   # 반대 방향
    v, w = torch.randn(16), torch.randn(16)
    assert cosine(v, w) == pytest.approx(F.cosine_similarity(v, w, dim=0).item(), abs=1e-6)


@pytest.fixture(scope="module")
def qwen():
    mod = pytest.importorskip("qwen_embedding")
    try:
        table, names = mod.load_embedding(local_only=True)
        tok = mod.load_tokenizer(local_only=True)
    except Exception:
        pytest.skip("Qwen 모델 파일이 아직 내려받아지지 않음(qwen_embedding.py를 먼저 실행)")
    return mod, table, names, tok


def test_qwen_embedding_shape_and_share(qwen):
    mod, table, names, _ = qwen
    assert table.shape == (151_936, 896)
    assert round(table.numel() / mod.TOTAL_PARAMS, 3) == 0.276


def test_qwen_lm_head_is_tied_not_stored(qwen):
    _, _, names, _ = qwen
    assert "model.embed_tokens.weight" in names
    assert not any("lm_head" in n for n in names)
    assert len(names) == 290          # 24계층 × 12 + Embedding + Final Norm


def test_related_words_more_similar(qwen):
    _, table, _, tok = qwen

    def sim(a, b):
        return F.cosine_similarity(table[tok.encode(a)[0]], table[tok.encode(b)[0]], dim=0)

    assert sim(" king", " queen") > sim(" king", " banana")
    assert sim(" Monday", " Tuesday") > sim(" apple", " Monday")
