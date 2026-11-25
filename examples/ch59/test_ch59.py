# 제59장 검증 테스트(선택 장)
#
# 실행: python -m pytest examples/ch59 -q

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

from golden import BRAM_KIB, COLS, ROWS, acc_bits, quant_rows, to_hex  # noqa: E402
from rtl_model import RTL, read_hex, simulate  # noqa: E402


def test_accumulator_and_bram():
    assert acc_bits(896) == 25 and 896 * 128 * 128 < 2 ** 24
    assert BRAM_KIB == 225 and 128 * 896 / 1024 <= BRAM_KIB < 896 * 896 / 1024


def test_hex_twos_complement_roundtrip(tmp_path):
    vals = [-128, -1, 0, 1, 127]
    (tmp_path / "a.hex").write_text("\n".join(to_hex(vals, 8)))
    assert to_hex(vals, 8) == ["80", "ff", "00", "01", "7f"] and read_hex(tmp_path / "a.hex", 8) == vals


def test_quant_range():
    q, s = quant_rows(np.array([[0.5, -1.0, 0.25]], dtype=np.float32))
    assert q.tolist() == [[64, -127, 32]] and abs(s[0, 0] - 1 / 127) < 1e-7


def test_rtl_model_matches_golden_files():
    w, x = read_hex(RTL / "w.hex", 8), read_hex(RTL / "x.hex", 8)
    exp = read_hex(RTL / "expected.hex", 32)
    got, cycles = simulate(w, x, ROWS, COLS)
    assert got == exp == (np.array(w).reshape(ROWS, COLS) @ np.array(x)).tolist()
    assert cycles == ROWS * COLS + 2                                    # 시작 1 + MAC ROWS×COLS + 마지막 누산 1
