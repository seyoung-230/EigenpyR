#!/usr/bin/env bash
set -euo pipefail

# === 기본 경로(프로젝트 루트 기준) ===
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
CORE_INC="$ROOT_DIR/core/include"
CORE_SRC="$ROOT_DIR/core/src"
PKG_ROOT="$ROOT_DIR/bindings/r/EigenRBinding"
PKG_INC="$PKG_ROOT/inst/include"
PKG_SRC="$PKG_ROOT/src"

echo "[EigenPyR] Core -> R package sync 시작"
echo "  ROOT        : $ROOT_DIR"
echo "  CORE_INC    : $CORE_INC"
echo "  CORE_SRC    : $CORE_SRC"
echo "  PKG_INC     : $PKG_INC"
echo "  PKG_SRC     : $PKG_SRC"

# 폴더 생성
mkdir -p "$PKG_INC" "$PKG_SRC"

# include 전체 동기화(병합). 기존 파일 유지, 새/수정본만 갱신
# 주의: 모듈/라퍼 폴더는 inst/include에 없으니 안전함
rsync -av --delete-excluded --exclude=".DS_Store" \
  "$CORE_INC/" "$PKG_INC/"

# src 동기화. core/src의 모든 .cpp/.h 등을 복사하되 module.cpp는 건드리지 않음
# (module.cpp는 PKG_SRC에 있으므로 안전하게 제외)
rsync -av --exclude="module.cpp" --exclude=".DS_Store" \
  "$CORE_SRC/" "$PKG_SRC/"

echo "[EigenPyR] 복사 완료 "
echo "이제 bindings/r/EigenRBinding/src/module.cpp, R/api.R만 수정하면 됩니다."
