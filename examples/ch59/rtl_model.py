# 제59장(선택): mac_matmul.v의 동작을 사이클 단위로 흉내 내는 파이썬 모델
#
# RTL 시뮬레이터가 없는 환경에서 설계의 순서(2단 파이프라인, 행 끝 처리)가 맞는지 먼저 확인하는 용도다.
# 레지스터는 사이클마다 "현재 값으로 다음 값을 계산한 뒤 한꺼번에 바꾼다"(Verilog의 <=와 같음).
# 이 모델이 맞아도 RTL 자체의 검증을 대신하지는 않는다.
#
# 실행: python examples/ch59/rtl_model.py   (golden.py가 만든 HEX 파일 필요)

from pathlib import Path

RTL = Path(__file__).resolve().parent / "rtl"


def read_hex(path: Path, bits: int) -> list:
    """2의 보수 HEX를 부호 있는 정수로(제31장)"""
    out = []
    for line in path.read_text().split():
        v = int(line, 16)
        out.append(v - (1 << bits) if v >> (bits - 1) else v)
    return out


def simulate(w: list, x: list, rows: int, cols: int, max_cycles: int = 10 ** 7) -> tuple:
    s = {"running": 0, "row": 0, "col": 0, "w_q": 0, "x_q": 0, "rd_valid": 0, "rd_last": 0, "rd_row": 0,
         "acc": 0}
    results, cycle, start = {}, 0, 1
    while cycle < max_cycles:
        n = dict(s)                                                  # 다음 상태(모두 현재 값으로 계산)
        n["rd_valid"], n["rd_last"] = 0, 0
        if start and not s["running"]:                               # 1단
            n.update(running=1, row=0, col=0)
        elif s["running"]:
            r, c = s["row"], s["col"]
            n.update(w_q=w[r * cols + c], x_q=x[c], rd_valid=1, rd_row=r, rd_last=int(c == cols - 1))
            if c == cols - 1:
                n["col"] = 0
                if r == rows - 1:
                    n["running"] = 0
                else:
                    n["row"] = r + 1
            else:
                n["col"] = c + 1
        done = False
        if s["rd_valid"]:                                            # 2단
            total = s["acc"] + s["w_q"] * s["x_q"]
            if s["rd_last"]:
                results[s["rd_row"]] = total
                n["acc"] = 0
                done = s["rd_row"] == rows - 1
            else:
                n["acc"] = total
        s, start, cycle = n, 0, cycle + 1
        if done:
            return [results[r] for r in range(rows)], cycle
    raise RuntimeError("끝나지 않음")


if __name__ == "__main__":
    w, x, exp = read_hex(RTL / "w.hex", 8), read_hex(RTL / "x.hex", 8), read_hex(RTL / "expected.hex", 32)
    rows, cols = len(exp), len(x)
    got, cycles = simulate(w, x, rows, cols)
    for r, (g, e) in enumerate(zip(got, exp)):
        print(f"  행 {r}: 모델 {g:>8,}  기대값 {e:>8,}  {'일치' if g == e else '불일치'}")
    print(f"  {cycles:,}사이클(= {rows} × {cols} + 2), 100MHz라면 {cycles / 100:.1f}마이크로초")
