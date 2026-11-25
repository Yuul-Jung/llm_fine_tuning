# GPU 확인 안내(저자 PC: RTX 3080 Ti)

교재 검증 환경에는 GPU가 없어서 GPU 관련 항목이 "미검증"으로 남아 있습니다. 학생 장비는 아직 정해지지 않았으므로, 교재는 다음 기준을 그대로 둡니다.

- 기본 경로: CPU(모든 실습이 CPU에서 동작)
- GPU 최소 기준: GT 1030 2GB(메모리가 가장 작은 경우를 가정)
- 저자 확인: RTX 3080 Ti에서 실행하되, GPU 메모리를 2GB로 제한해 최소 기준의 **용량**을 모사

2GB 제한은 메모리 용량만 흉내 냅니다. GT 1030의 속도나 세대별 특성(FP16이 매우 느림 등)은 모사하지 못하므로, 학생 장비가 정해지면 그 장비에서 한 번 더 확인합니다.

## 한 번에 확인하기

아래 순서대로 명령 창(Windows는 PowerShell)에 한 줄씩 입력합니다.

1. 드라이버 확인: `nvidia-smi` → 오른쪽 위 Driver Version이 **580 이상**이어야 합니다.
2. Python 3.13 설치 후, 저장소 폴더에서 가상 환경 만들기
   - Windows: `python -m venv .venv` → `.venv\Scripts\activate`
   - Linux: `python3 -m venv .venv` → `source .venv/bin/activate`
3. 패키지 설치: `python -m pip install -r requirements/gpu_rtx.txt`
4. 확인 실행: `python tools/gpu_verify.py`
   - 처음 실행하면 Qwen2.5 모델을 내려받고, 없는 학습 결과를 GPU로 다시 만듭니다(수십 분 걸릴 수 있음).
   - 중간에 실패한 단계가 있어도 끝까지 진행하고, 실패 내용을 보고서에 적습니다.
   - 모델을 다시 만드는 과정에서 저장소의 기록 파일(예: examples/ch49/qlora.json)이 GPU 결과로 바뀔 수 있습니다. 바뀐 파일도 함께 올려 주시면 CPU 결과와 비교하는 데 씁니다.
5. 결과 보내기: `reports/gpu_report.md`(사람이 읽는 요약)와 `reports/gpu_report.json`(자세한 기록) 두 파일을 보내 주시거나 저장소에 올려 주세요.

## 선택: Ollama와 llama.cpp까지 확인하기

- Ollama를 설치해 두면(https://ollama.com) 4번 실행 때 자동으로 모델을 등록하고, validation 질문 6개의 답을 교재의 Q8_0 답과 비교하며, `ollama ps`로 GPU 적재 여부를 기록합니다.
- llama.cpp의 `llama-bench`가 PATH에 있으면 Q8_0·Q4_0의 초당 토큰을 잽니다.
- 둘 다 없으면 해당 단계는 "건너뜀"으로 기록됩니다.

## 보고서로 채우는 교재 항목

| 보고서 단계 | 교재 위치 |
|---|---|
| 3. QLoRA 학습 메모리(2GB 제한) | 제49장 메모리 예산(계산값 895MiB)의 실측 |
| 4. Q8_0 GGUF 품질과 속도 | 제55장 품질의 장치 간 일치, 속도 |
| 5. Ollama 응답 비교 | 제56장 응답 일치, GPU 적재 |
| 6. llama.cpp 속도 | 제55·58장 초당 토큰 |
