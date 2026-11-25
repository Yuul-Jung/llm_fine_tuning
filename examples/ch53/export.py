# 제53장 핵심 실습: 한 계층의 가중치를 이름으로 찾아 NumPy·CSV·Binary·HEX로 내보내고, 다시 읽어 행렬곱을 재현한다
#
# 실행: python examples/ch53/export.py   (제52장의 병합 모델 필요, CPU 약 30초)
#
# 만드는 파일: models/ch53/(내보낸 파일 네 가지, 저장소에는 넣지 않음)

import csv
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from common.device import select_device  # noqa: E402

MERGED = ROOT / "models" / "fpga_merged_v1"                      # 제52장
OUT = ROOT / "models" / "ch53"
NAME = "model.layers.0.self_attn.k_proj"                          # 896 → 128, 작은 계층


def save_all(w: np.ndarray, b: np.ndarray, out: Path) -> dict:
    """같은 가중치를 네 형식으로 저장한다. 모두 행 우선(row-major), FP32."""
    out.mkdir(parents=True, exist_ok=True)
    paths = {k: out / f"k_proj.{k}" for k in ("npy", "csv", "bin", "hex")}
    np.save(paths["npy"], w)
    with open(paths["csv"], "w", newline="") as f:
        csv.writer(f).writerows(w.tolist())                       # 파이썬 float → 최단 10진 표현(왕복 손실 없음)
    w.astype("<f4").tofile(paths["bin"])                          # little-endian 4바이트씩
    with open(paths["hex"], "w") as f:                            # 한 줄에 32비트 단어 하나($readmemh 형식)
        f.writelines(f"{x:08x}\n" for x in w.astype("<f4").view("<u4").ravel())
    np.save(out / "k_proj_bias.npy", b)
    return paths


def load_all(paths: dict, shape: tuple) -> dict:
    return {
        "npy": np.load(paths["npy"]),
        "csv": np.loadtxt(paths["csv"], delimiter=",", dtype=np.float32).reshape(shape),
        "bin": np.fromfile(paths["bin"], dtype="<f4").reshape(shape),
        "hex": np.array([int(x, 16) for x in paths["hex"].read_text().split()], dtype="<u4")
                 .view("<f4").reshape(shape),
    }


def main() -> None:
    from transformers import AutoModelForCausalLM
    device, _ = select_device()
    model = AutoModelForCausalLM.from_pretrained(MERGED, dtype=torch.float32).to(device).eval()

    print("── 1. Named Parameters: 이름으로 찾기 ──")
    params = dict(model.named_parameters())
    print(f"  파라미터 텐서 {len(params)}개. 앞의 셋:")
    for k in list(params)[:3]:
        print(f"    {k:<44}{tuple(params[k].shape)}")
    k_all = [k for k in params if k.endswith("self_attn.k_proj.weight")]
    print(f"  'self_attn.k_proj.weight'로 끝나는 이름 {len(k_all)}개(층마다 하나)")
    layer = model.get_submodule(NAME)
    print(f"  {NAME}: weight {tuple(layer.weight.shape)}, bias {tuple(layer.bias.shape)}")

    print("\n── 2. Weight Tying ──")
    emb, head = model.model.embed_tokens.weight, model.lm_head.weight
    print(f"  설정 tie_word_embeddings {model.config.tie_word_embeddings}")
    print(f"  Embedding과 LM Head의 메모리 주소가 같음: {emb.data_ptr() == head.data_ptr()}"
          f"(주소 값 자체는 실행마다 다르다)")
    print(f"  named_parameters에 LM Head가 따로 있음: {'lm_head.weight' in params}"
          f"(같은 텐서는 한 번만 나옴), 크기 {emb.numel() * 4 / 2**20:.0f}MiB")

    print("\n── 3. 네 형식으로 내보내고 다시 읽기 ──")
    w = layer.weight.detach().cpu().numpy()
    b = layer.bias.detach().cpu().numpy()
    paths = save_all(w, b, OUT)
    back = load_all(paths, w.shape)
    for k, arr in back.items():
        print(f"  {k:<4}{paths[k].stat().st_size / 1024:>8.0f}KiB  원본과 비트 단위로 같음 {np.array_equal(arr, w)}")
    print(f"  HEX 첫 줄 {paths['hex'].read_text().split()[0]} = {w[0, 0]!r}")

    print("\n── 4. 읽어 들인 HEX로 행렬곱 재현 ──")
    torch.manual_seed(0)
    x = torch.randn(4, w.shape[1])
    with torch.no_grad():
        y_layer = layer(x.to(device)).cpu()
    w_hex, b_t = torch.from_numpy(back["hex"]), torch.from_numpy(b)
    y_linear = torch.nn.functional.linear(x, w_hex, b_t)           # 계층과 같은 연산
    y_manual = x @ w_hex.T + b_t                                   # 곱한 뒤 따로 더함
    print(f"  같은 연산(linear)으로 재현: 최대 차이 {(y_layer - y_linear).abs().max().item():.2e}")
    print(f"  곱셈과 덧셈을 나눠 재현:    최대 차이 {(y_layer - y_manual).abs().max().item():.2e}(계산 순서 차이)")

    print("\n── 5. 자료형과 Byte Order ──")
    v = np.float32(1.0)
    little, big = struct.pack("<f", v), struct.pack(">f", v)
    print(f"  1.0의 바이트: little-endian {little.hex(' ')}, big-endian {big.hex(' ')}")
    wrong = np.fromfile(paths["bin"], dtype=">f4", count=4)
    print(f"  little로 쓴 파일을 big으로 읽으면 {wrong} (원래 {w.ravel()[:4]})")
    print(f"  이 컴퓨터의 기본 순서: {sys.byteorder}")


if __name__ == "__main__":
    main()
