# 제01장 핵심 실습 1: 로컬 실습 환경 점검 보고서
#
# 실행: python examples/ch01/check_env.py
# 결과: 화면 보고서 + examples/ch01/env_report.json

import json
import os
import platform
import sys
from pathlib import Path

# examples/common 모듈을 불러오기 위해 상위 폴더(examples)를 검색 경로에 추가한다.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from common.device import select_device  # noqa: E402


def read_total_ram_gib() -> float:
    """Linux의 /proc/meminfo에서 전체 메모리를 GiB 단위로 읽는다."""
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kib = int(line.split()[1])
                    return round(kib / 1024 / 1024, 1)
    except OSError:
        pass
    return -1.0


def collect_report() -> dict:
    """환경 정보를 사전(dict) 하나로 모은다."""
    device, dtype = select_device(verbose=False)

    report = {
        "os": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,      # CPU 전용 빌드면 None
        "cpu_cores": os.cpu_count(),
        "ram_gib": read_total_ram_gib(),
        "device": device,
        "dtype": str(dtype).replace("torch.", ""),
        "gpu": None,
    }

    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        free_b, total_b = torch.cuda.mem_get_info(0)
        report["gpu"] = {
            "name": props.name,
            "compute_capability": f"{props.major}.{props.minor}",
            "vram_total_mib": total_b // 2**20,
            "vram_free_mib": free_b // 2**20,
            "bf16_fast": props.major >= 8,
        }
    return report


def smoke_test(device: str, dtype: torch.dtype) -> float:
    """선택한 장치에서 작은 행렬곱을 실행해 결과를 CPU FP32 계산과 비교한다."""
    torch.manual_seed(0)
    a = torch.randn(64, 64)
    b = torch.randn(64, 64)
    expected = a @ b
    actual = (a.to(device, dtype) @ b.to(device, dtype)).float().cpu()
    return (actual - expected).abs().max().item()


def main() -> None:
    report = collect_report()
    dtype = getattr(torch, report["dtype"])
    report["smoke_test_max_abs_error"] = smoke_test(report["device"], dtype)

    print("=== 로컬 실습 환경 점검 ===")
    for key, value in report.items():
        print(f"{key:>24} : {value}")

    out = Path(__file__).with_name("env_report.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n보고서 저장: {out.name}")


if __name__ == "__main__":
    main()
