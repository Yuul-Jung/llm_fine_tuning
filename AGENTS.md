# GitHub 교육용 HTML 작성 규칙

이 저장소에서 새 HTML 교재를 만들거나 기존 교재를 수정할 때는 먼저 다음 문서를 확인한다.
- https://github.com/Yuul-Jung/rule_md/blob/main/AGENTS.md
- https://github.com/Yuul-Jung/rule_md/blob/main/web/education-html/STANDARD.md
- https://github.com/Yuul-Jung/rule_md/blob/main/web/education-html/INTEGRATION.md
- https://github.com/Yuul-Jung/rule_md/blob/main/common/code-style-rules.md

적용 사항:
1. 책장 배너 이미지는 중앙 표준의 hero-bookshelf.jpg와 동일하게 유지한다.
2. 좌측 전체 장 목차와 현재 페이지 절 목차 및 모바일 메뉴를 구현한다.
3. 기존 학습 내용·코드·이미지는 보존하고, 차이가 있는 부분만 수정한다.
4. 수정 시 장별 파일명을 _v01 → _v02로 올리고 이전 생성본은 결과 폴더에서 정리한다.
5. 코드 문법, 내부 링크, 개인정보, 접근성과 브라우저 표시를 검증한다.
6. 확인하지 못한 실행 결과는 미검증으로 표기한다.
7. 원격 문서를 읽을 수 없으면 추정 적용하지 않고 로컬 규칙 사본을 요구한다.

이 저장소 고유 규칙:
1. 장·절 구성은 `design/목차_설계서_v01.md`, 기술 기준은 `design/교육과정_설계서_v1.3.md`를 따른다.
2. 본문 코드 박스와 `examples/chNN/` 실습 소스는 같은 내용을 유지한다.
3. 모든 실습은 `examples/common/device.py`의 `select_device()`로 장치와 자료형을 정한다.
4. 각 장은 `examples/chNN/test_chNN.py` 검증 테스트를 둔다.
5. 실습 환경은 로컬(CPU 또는 GT 1030 2GB)이며 GT 1030에서는 FP32 연산을 사용한다.
