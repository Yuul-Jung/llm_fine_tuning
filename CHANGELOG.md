# 변경 이력

## 2026-11-25 (제6부)

### 추가
- 제6부 제51~53장, 제57~60장 원고·실습 코드·검증 테스트(전체 테스트 319개)
- 배포용 Adapter(원본 Revision 기록)와 대응 기록 `examples/ch51/manifest.json`, 병합 모델 생성(`examples/ch52/merge.py`)
- Ollama 형식의 로컬 API 서버·클라이언트(`examples/ch57/local_api.py`), 장치별 메모리 판정(`examples/ch58`)
- 제59장(선택) INT8 MAC RTL과 Testbench(`examples/ch59/rtl/`), 사이클 단위 파이썬 모델
- 프로젝트 기록 `examples/ch60/project.json`

### 추가(사용자 허가 후)
- 제54장 GGUF 변환기(`examples/ch54/to_gguf.py`), 제55장 Q8_0·Q4_0 양자화 비교, 제56장 Ollama Modelfile
- requirements에 gguf 0.19.0 고정

### 변경(제6부 적대적 검증 반영, `design/검증보고서_제6부.md`)
- 제46장 v02, 제47·48장 v04, 제50장 v03: 실습 장치를 `select_device()`로 결정(제47장 `train_device()`: VRAM이 부족하면 CPU)
- 제47장 v05: Greedy 평가에 생성 설정의 반복 억제 1.1이 함께 적용됨을 명시
- 제54장 v02(양자화 경로), 제58장 v02(실제 GGUF 크기), 제60장 v02(상태표에 GGUF·Modelfile)
- 제47장 v06, 제49장 v03: 평가·메모리 측정 입력을 모델의 장치로(GPU 경로 오류 수정)
- 제53장 v02: 하드웨어 계산 참조를 제59장으로 정정(전체 장 연결 점검)
- 제49장 v04: 저자 GPU(RTX 3080 Ti) 결과 반영(2GB 제한 실측 1,086MiB, CPU와 곡선·판정 일치)
- 제50장 v04: 기록이 있어도 Adapter가 없으면 다시 학습(새 환경에서 모델 재생성 오류 수정)
- tools/gpu_verify.py: 재생성 중 바뀐 교재 기록은 reports/gpu_records/로 옮기고 원래대로 되돌림
- 저자 GPU 확인 도구 추가: `tools/gpu_verify.py`, `design/GPU_확인_안내.md`, `requirements/gpu_rtx.txt`(PyTorch 2.15.0 + CUDA 13.2)

## 2026-11-25 (제5부)

### 추가
- 제5부 제41~50장 원고·실습 코드·검증 테스트(전체 테스트 293개)
- FPGA 질의응답 데이터셋 `datasets/fpga_qa/`(원본 50개, JSONL, 그룹 분할 `split_v1.json`, 질문 변형 보강 `augment_v1.json`, 데이터 카드)
- 평가 기준 파일 `configs/eval_criteria.yaml`(실험 전 확정, version 2)
- 직접 구현 SFT 루프(Checkpoint 재개), SFTTrainer 대응, LoRA·QLoRA 도메인 학습, 오답 분석과 최종 test 평가 기록(`examples/ch47~50/*.json`)
- requirements에 Datasets 5.2.0, TRL 1.18.2 고정

### 변경(제5부 적대적 검증 반영, `design/검증보고서_제5부.md`)
- 평가 기준 version 2: 학습 중 핵심어·형식 판정을 test에서 validation으로(합격선·test 핵심어는 그대로), 제43·47·48장 v02에서 다시 측정
- 제44장 v02: 본문 속 특수 토큰 글자가 기본 설정에서 특수 토큰으로 바뀌는 사실을 바로잡고 실습·테스트 추가
- 제47장 v03, 제48장 v03, 제49장 v02: Trainer 작업 폴더를 운영체제 임시 폴더로(Windows 호환)
- 제42장 v02: 제45장 예고 정정
- 제50장 v02: 배포 후보 불채택과 실습 완료 구분, validation 기반 데이터 추가의 한계 명시
- 제41장 v02: 데이터 카드에 분할·보강·기준 파일 추가

## 2026-11-24 (제4부)

### 추가
- 제4부 제31~40장 원고·실습 코드·검증 테스트(전체 테스트 254개)
- 직접 구현: 2의 보수·고정소수점 변환기, IEEE 754 비트 분해, INT8·INT4 양자화와 Bit Packing, 가짜 양자화 오차 분석, NF4·이중 양자화
- bitsandbytes 4비트 로딩, QLoRA 메모리 측정, Qwen2.5-0.5B QLoRA 학습(CPU 경로), 학습 방식 비교 결과 `examples/ch40/results.json`
- `requirements/cpu.txt` 라이브러리 버전 고정(검증 조합), `requirements/gt1030.txt` 후보 조합(미검증)

### 변경(제4부 적대적 검증 반영, `design/검증보고서_제4부.md`)
- 제07장 v02: 표 안 조건부 확률 기호 수정(제1부에서 놓친 표시 오류)
- 제25장 v03, 제37·38·39·40장 v02: 내부 설계서 절 번호 인용을 교재 기준 표현으로 교체(본문·코드 주석)
- 제36장 v02: INT4·NF4 비교에 MSE·최대 절대오차 추가, NF4의 최악 오차 증가를 설명
- 제39장 v02: 준비 순서의 근거(kbit 학습 준비 함수의 전체 동결) 명시

## 2026-11-24 (제3부)

### 추가
- 제3부 제21~30장 원고·실습 코드·검증 테스트(전체 테스트 205개)
- 고정 데이터셋(`datasets/fixed/train.txt`, `eval.txt`)과 도메인 데이터셋(`domain_*.txt`), 출처 기록 `datasets/LICENSES.md`
- 공통 모듈 `examples/common/fixed_data.py`, 공용 학습 루프 `examples/ch25/trainer.py`, 직접 구현 LoRA `examples/ch29/lora_linear.py`
- 기준 PPL 기록 `examples/ch22/baseline_ppl.json`, 사전학습 기록 `examples/ch25/history.json`
- requirements에 PEFT 추가(교재 검증 0.22.0)

### 변경(제3부 적대적 검증 반영, `design/검증보고서_제3부.md`)
- 제21장 v02: 파국적 망각을 처음 측정하는 장을 제25장으로 정정
- 제25장 v02: 도메인 평가 데이터(블록 1개)의 해석 한계를 명시
- 제28장 v02: 제29·30장 예고를 실제 구성에 맞게 정정, B = 0 예제의 난수 고정(재현성)
- 제29장 v02: 작은 PPL 차이를 학습 효과로 해석하지 않도록 명시

## 2026-11-24 (제2부)

### 추가
- 제2부 제11~20장 원고·실습 코드·검증 테스트(전체 테스트 153개)
- 공통 모듈: Qwen 모델·토크나이저 불러오기(`examples/common/qwen.py`)
- 직접 구현 Qwen2-Tiny(`examples/ch19/tiny_model.py`)와 Hugging Face 형식 호환 검증(`examples/ch19/hf_compat.py`)

### 변경(제2부 적대적 검증 반영, `design/검증보고서_제2부.md`)
- 제04장 v03, 제10장 v02: 학습자용 본문에서 내부 설계서 절 번호 참조를 제거(제04장은 허용 오차 안내, 제10장은 근거 장으로 대체)
- 제11장 v02: 미래 토큰을 가리는 이유를 "생성할 때"에서 "다음 토큰을 맞히도록 학습할 때"로 정정
- 제18장 v02: Decoder Layer 재현 상대 오차를 실제 출력(2×10⁻⁶ 이하)과 일치하게 수정, 절대오차 설명을 측정값 근거로 교체
- 제20장 v02: Forward Hook을 처음 쓴 장(제12장)과 어긋난 서술 수정
- 목차 설계서: 제20장 01절 제목을 원고와 일치
- 제19·20장: 실행 결과 표기("실제 실행 결과(교재 검증 환경)")를 다른 장과 통일

## 2026-11-24

### 추가
- 저장소 초기 구성: rule_md 교육용 HTML 표준 템플릿, 교육과정 설계서 v1.3, 60장 목차·절 설계서
- 제1부 제01~10장 원고·실습 코드·검증 테스트(테스트 86개)
- 공통 모듈: 장치 선택(`examples/common/device.py`), 모델 Revision 고정(`examples/common/models.py`)
- 본문 코드 박스·실습 소스 일치 검사 도구(`tools/check_code_boxes.py`)

### 변경(제1부 적대적 검증 반영, `design/검증보고서_제1부.md`)
- 제01장 v02: GT 1030의 FP16 속도 서술을 제02장 실측 기준과 일치하도록 수정
- 제04장 v02: Softmax 첫 등장 위치에 정의 추가(선수학습 순서)
- 제08장 v02: 어휘 구성에 관한 근거 없는 단정을 확인된 사실로 수정
- 제09장 v02: "학습 데이터"를 "토크나이저를 학습한 데이터"로 정정
