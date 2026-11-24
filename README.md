# 📘 LLM 파인튜닝 및 로컬 AI 구현

LLM의 수학적 원리와 내부 연산을 직접 구현하고, 내 컴퓨터에서 파인튜닝·병합·변환·배포까지 완성하는 **6부 60장** 실습 교재입니다.

- **실습 환경:** Ubuntu 24.04 · VS Code · Python · PyTorch
- **하드웨어 경로:** CPU 또는 NVIDIA GT 1030 2GB(로컬 전용)
- **기본 모델:** Qwen2.5-0.5B-Instruct, Qwen2 구조 호환 초소형 모델 Qwen2-Tiny
- **배포:** GGUF · llama.cpp · Ollama · Python API

## 교과 구성

| 구분 | 교육과정 | 범위 |
|---|---|---|
| 제1부 | LLM 기초 및 수치 연산 | 01~10장 |
| 제2부 | Transformer 구조 및 LLM 내부 분석 | 11~20장 |
| 제3부 | 전이학습 및 LoRA 파인튜닝 | 21~30장 |
| 제4부 | 양자화 및 QLoRA 파인튜닝 | 31~40장 |
| 제5부 | Hugging Face 실전 지도 미세조정 | 41~50장 |
| 제6부 | 모델 배포 및 온디바이스 AI | 51~60장 |

전체 장·절 구성은 [`design/목차_설계서_v01.md`](design/목차_설계서_v01.md), 교육과정 기준은 [`design/교육과정_설계서_v1.3.md`](design/교육과정_설계서_v1.3.md)에 있습니다.

## 저장소 구조

```text
llm_fine_tuning/
├── content/          # 장별 원고(Markdown)
├── examples/         # 장별 실습 코드와 검증 테스트
│   ├── common/       # 전 장 공통 모듈(장치 자동 선택)
│   └── ch01/ ...
├── design/           # 교육과정 설계서, 목차 설계서
├── requirements/     # 경로별 패키지 목록(cpu.txt, gt1030.txt)
├── docs/             # GitHub Pages 게시본(생성 결과)
├── assets/           # 공통 디자인(rule_md 표준)
├── course.json       # 강좌·장 목록
├── build.py          # rule_md 공통 HTML 생성기
└── publish.sh        # 생성 → docs/ 게시
```

## 시작하기

```bash
# 저장소 내려받기 → VS Code로 루트 폴더 열기
git clone https://github.com/Yuul-Jung/llm_fine_tuning.git
cd llm_fine_tuning
code .

# 가상환경 생성·활성화 → 경로에 맞는 패키지 설치
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/cpu.txt      # GT 1030: requirements/gt1030.txt

# 제01장 실습과 검증
python examples/ch01/check_env.py
python -m pytest examples/ch01 -q
```

## 교재 HTML 생성

```bash
python -m pip install mistune
bash publish.sh        # content/*.md → dist/ → docs/
```

## 📘 교육용 HTML 공통 제작 규칙

이 저장소의 교육자료(웹페이지·강의 HTML)는 [Yuul-Jung 교육용 HTML 공통 표준](https://github.com/Yuul-Jung/rule_md/tree/main/web/education-html)을 적용합니다.

- [제작 규칙](https://github.com/Yuul-Jung/rule_md/blob/main/web/education-html/STANDARD.md)
- [다른 저장소 적용 안내](https://github.com/Yuul-Jung/rule_md/blob/main/web/education-html/INTEGRATION.md)
- [기준 디자인 사례](https://github.com/Yuul-Jung/ai-agent-engineering)

공통 책장 이미지, 컴팩트한 상단 배너, 좌측 전체 장/절 탐색, 모바일 메뉴, 파일 버전 및 검증 기준을 따릅니다.
적용한 rule_md 기준 커밋: `028248a19afc66e9aafdab393fbe0de10a3930ca`
