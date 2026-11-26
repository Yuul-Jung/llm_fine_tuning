# GPU 확인 보고서

- 운영체제: Linux-7.0.0-43-generic-x86_64-with-glibc2.39
- Python: 3.12.3
- PyTorch: 2.15.0+cu132
- CUDA 사용 가능: True
- GPU: NVIDIA GeForce RTX 3080 Ti Laptop GPU
- Compute Capability: 8.6
- VRAM MiB: 15984
- CUDA(PyTorch 빌드): 13.2
- 지원 세대: sm_75 sm_80 sm_86 sm_90 sm_100 sm_120
- 드라이버: 595.99.02
- transformers: 5.24.0
- peft: 0.22.0
- trl: 1.18.2
- bitsandbytes: 0.50.2
- gguf: 0.19.0
- Ollama: 없음
- llama-bench: 없음

## 2. 모델 다시 만들기: 실패 (586.7초)
- 오류: RuntimeError: ch51/save_load.py 실패: Traceback (most recent call last):
  File "/home/jung/llm_fine_tuning/examples/ch51/save_load.py", line 130, in <module>
    main()
  File "/home/jung/llm_fine_tuning/examples/ch51/save_load.py", line 71, in main
    size = (CKPT / name).stat().st_size
           ^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/pathlib.py", line 842, in stat
    return os.stat(self, follow_symlinks=follow_symlinks)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: '/home/jung/llm_fine_tuning/models/ch50/원본/checkpoint-15/adapter_model.safetensors'


## 3. QLoRA 학습 메모리(2GB 제한 모사): 성공 (14.3초)
- GPU 전체 MiB: 15984
- 제한 MiB: 2048
- 최대 할당 MiB: 1086
- 교재 계산값 MiB: 895.1
- 손실: [2.3936, 3.3486, 3.4834, 2.95, 2.7733]
- NaN·Inf 없음: True

## 4. Q8_0 GGUF 품질과 속도(GPU): 실패 (0.0초)
- 오류: OSError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '/home/jung/llm_fine_tuning/models/fpga_merged_v1'. Use `repo_type` argument if needed.

## 5. Ollama 등록과 응답 비교: 성공 (0.0초)
- 결과: Ollama 없음(건너뜀)

## 6. llama.cpp 속도: 성공 (0.0초)
- 결과: llama-bench 없음(건너뜀)
