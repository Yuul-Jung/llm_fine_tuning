# 모든 장이 공유하는 장치·연산 자료형 선택 모듈
#
# 판단 순서
#   1. GPU(CUDA)를 쓸 수 있는가
#   2. 설치된 PyTorch가 이 GPU 세대(Compute Capability)를 지원하는가
#   3. 이 GPU에서 빠르고 안전한 연산 자료형은 무엇인가

import torch


def select_device(verbose: bool = True):
    """사용할 장치 이름과 연산 자료형을 함께 돌려준다.

    반환값: (device, dtype)
        device: "cuda" 또는 "cpu"
        dtype : torch.float32 / torch.float16 / torch.bfloat16
    """
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        arch = f"sm_{major}{minor}"

        # PyTorch 빌드에 이 세대의 GPU 코드가 들어 있어야 실제 연산이 가능하다.
        if arch in torch.cuda.get_arch_list():
            if major >= 8:
                # Ampere 이후: BF16 고속 연산 지원
                return "cuda", torch.bfloat16
            if major == 7:
                # Volta·Turing: FP16 고속 연산 지원, BF16 미지원
                return "cuda", torch.float16
            # Pascal 이전(GT 1030 = sm_61): FP16이 FP32의 1/64 속도이므로 FP32 사용
            return "cuda", torch.float32

        if verbose:
            print(f"[device] {arch} GPU를 설치된 PyTorch가 지원하지 않아 CPU를 사용합니다.")

    return "cpu", torch.float32
