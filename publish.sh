#!/usr/bin/env bash
# 교재 HTML 생성 후 GitHub Pages 게시 폴더(docs/)로 복사
#
# 실행: bash publish.sh
# 준비: python -m pip install mistune

set -euo pipefail

# 1) content/*.md + course.json → dist/*.html (rule_md 공통 생성기)
python build.py

# 2) docs/의 이전 HTML을 지우고 최신 생성본만 복사
mkdir -p docs
rm -f docs/*.html
cp dist/*.html docs/

# 3) GitHub Pages가 Jekyll 처리 없이 그대로 게시하도록 표시
touch docs/.nojekyll

echo "게시 폴더 갱신 완료: docs/"
ls docs
