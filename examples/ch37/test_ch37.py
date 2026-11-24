# 제37장 검증 테스트
#
# 실행: python -m pytest examples/ch37 -q

import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent, HERE.parent / "ch22", HERE.parent / "ch36"):
    sys.path.insert(0, str(p))

pytest.importorskip("bitsandbytes")

from load_4bit import hardware, memory_parts, nf4_config  # noqa: E402


def test_hardware_rules():
    hw = hardware()
    if hw["device"] == "cuda":
        assert hw["nf4"] == (hw["cc"] >= 6.0) and hw["int8"] == (hw["cc"] >= 7.5)
    else:
        assert hw["nf4"]


def test_config_matches_ch36():
    c = nf4_config()
    assert c.load_in_4bit and c.bnb_4bit_quant_type == "nf4" and c.bnb_4bit_use_double_quant
    assert c.bnb_4bit_compute_dtype == torch.float32


def test_linear4bit_matches_own_nf4():
    import bitsandbytes as bnb
    from nf4 import nf4_fake
    torch.manual_seed(0)
    lin = torch.nn.Linear(128, 64, bias=False)
    q = bnb.nn.Linear4bit(128, 64, bias=False, compute_dtype=torch.float32, quant_type="nf4")
    q.weight = bnb.nn.Params4bit(lin.weight.data.clone(), requires_grad=False, quant_type="nf4",
                                compress_statistics=False)                # 이중 양자화 없이 비교
    q = q.to("cpu") if not torch.cuda.is_available() else q.cuda()
    x = torch.randn(3, 128, device=q.weight.device)
    expect = x.cpu() @ nf4_fake(lin.weight.data).T                  # 제36장 직접 구현(이중 양자화 없음)
    torch.testing.assert_close(q(x).cpu(), expect, atol=1e-4, rtol=1e-4)


def test_qwen_4bit_memory():
    try:
        from load_4bit import load_nf4
        model = load_nf4(hardware()["device"])
    except Exception:
        pytest.skip("Qwen 모델이 아직 내려받아지지 않음")
    parts = memory_parts(model)
    assert parts["Embedding(= LM Head)"] == 151_936 * 896 * 4          # 양자화하지 않음(약 519MiB)
    assert parts["4비트 가중치(번호)"] == 24 * (896 * 896 * 2 + 128 * 896 * 2 + 4864 * 896 * 3) // 2
    assert sum(parts.values()) < 0.4 * 494_032_768 * 4
