#!/usr/bin/env bash
# pitch-core-onnx — model fetcher.
#
# Each ONNX backend takes a path to a user-supplied .onnx file. This
# script downloads or builds the canonical model for each backend.
#
# What it handles:
#   models/swift_f0.onnx        — auto, ~389 KB, MIT (lars76/swift-f0 PyPI wheel)
#   models/crepe-{tiny,...,full}.onnx
#                                — auto, ~165 MB total, MIT (yqzhishen/onnxcrepe v1.1.0)
#   models/rmvpe.onnx           — auto, ~362 MB, Apache-2.0 weights, MIT mirror
#                                  (HuggingFace lj1995/VoiceConversionWebUI)
#   models/fcpe.onnx            — auto from GitHub Release if available;
#                                  otherwise built locally via tools/fcpe_export.py
#                                  (~42 MB, MIT throughout)
#   models/pesto_mir1k_g7_48k_960.onnx
#                                — built locally via tools/pesto_export.py
#                                  (~17 MB; **LGPL-3.0** weights — see PESTO note)
#
# Flags:
#   --skip-{swift,crepe,rmvpe,fcpe,pesto}      — opt out of one
#   --{swift,crepe,rmvpe,fcpe,pesto}-only      — only fetch one
#
# Env vars:
#   MODELS_DIR   — destination dir (default: ./models)
#   PROJECT_VENV — venv path used for FCPE/PESTO export steps
#                  (default: ./.venv; created on demand)

set -euo pipefail
cd "$(dirname "$0")/.."

step() { printf '\n\033[1;36m== %s ==\033[0m\n' "$*"; }
ok()   { printf '\033[1;32m  ok\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m  !!\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31merror:\033[0m %s\n' "$*"; exit 1; }

DO_SWIFT=1; DO_CREPE=1; DO_RMVPE=1; DO_FCPE=1; DO_PESTO=0
# ↑ PESTO is opt-in by default because its weights inherit upstream
# LGPL-3.0. Pass --pesto to fetch it.

PESTO_REQUESTED=0
for arg in "$@"; do
  case "$arg" in
    -h|--help)        sed -n '2,30p' "$0"; exit 0 ;;
    --skip-swift)     DO_SWIFT=0 ;;
    --skip-crepe)     DO_CREPE=0 ;;
    --skip-rmvpe)     DO_RMVPE=0 ;;
    --skip-fcpe)      DO_FCPE=0 ;;
    --skip-pesto)     DO_PESTO=0 ;;
    --swift-only)     DO_SWIFT=1; DO_CREPE=0; DO_RMVPE=0; DO_FCPE=0; DO_PESTO=0 ;;
    --crepe-only)     DO_SWIFT=0; DO_CREPE=1; DO_RMVPE=0; DO_FCPE=0; DO_PESTO=0 ;;
    --rmvpe-only)     DO_SWIFT=0; DO_CREPE=0; DO_RMVPE=1; DO_FCPE=0; DO_PESTO=0 ;;
    --fcpe-only)      DO_SWIFT=0; DO_CREPE=0; DO_RMVPE=0; DO_FCPE=1; DO_PESTO=0 ;;
    --pesto-only)     DO_SWIFT=0; DO_CREPE=0; DO_RMVPE=0; DO_FCPE=0; DO_PESTO=1; PESTO_REQUESTED=1 ;;
    --pesto)          DO_PESTO=1; PESTO_REQUESTED=1 ;;
    *) die "unknown flag: $arg (try --help)" ;;
  esac
done

REPO_ROOT="$(pwd)"
MODELS_DIR="${MODELS_DIR:-$REPO_ROOT/models}"
mkdir -p "$MODELS_DIR"

# Pre-built FCPE ONNX hosted on HuggingFace (MIT, mirrors upstream
# torchfcpe checkpoint). Set FCPE_URL to empty string to force local
# build via tools/fcpe_export.py instead.
FCPE_URL="${FCPE_URL:-https://huggingface.co/gzivdo/fcpe-onnx/resolve/main/fcpe.onnx}"

curl_fetch() {
  local url="$1" dst="$2"
  if [[ -s "$dst" ]]; then
    ok "$(basename "$dst") already at $dst ($(du -h "$dst" | cut -f1))"
    return 0
  fi
  step "  $(basename "$dst") ← $url"
  curl -L --fail --progress-bar -o "$dst.partial" "$url"
  mv "$dst.partial" "$dst"
  ok "downloaded ($(du -h "$dst" | cut -f1))"
}

ensure_venv() {
  # Create or locate a Python venv for export-script steps.
  VENV="${PROJECT_VENV:-$REPO_ROOT/.venv}"
  PY="$VENV/bin/python"
  if [[ ! -x "$PY" ]]; then
    step "  creating venv at $VENV"
    command -v python3 >/dev/null || die "python3 not found in PATH"
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install --quiet --upgrade pip
  fi
  echo "$PY"
}

run_export() {
  # $1 = script name (in tools/), $2 = pip-installable pkg names, $3 = output path
  local script="$1" pkgs="$2" out="$3"
  if [[ -s "$out" ]]; then
    ok "$(basename "$out") already at $out"
    return 0
  fi
  PY="$(ensure_venv)"
  if ! $PY -c "import torch" 2>/dev/null; then
    step "  installing torch (one-time, ~200 MB)"
    $PY -m pip install --quiet torch
  fi
  for pkg in $pkgs; do
    if ! $PY -c "import ${pkg//-/_}" 2>/dev/null; then
      step "  installing $pkg"
      $PY -m pip install --quiet "$pkg"
    fi
  done
  step "  running $script → $out"
  $PY "$REPO_ROOT/tools/$script" --out "$out"
  [[ -s "$out" ]] || die "$script reported success but $out is missing"
  ok "$(basename "$out") built ($(du -h "$out" | cut -f1))"
}

# ──────────────────────────────────────────────────────────────────────
if [[ "$DO_SWIFT" == "1" ]]; then
  step "SwiftF0 (~389 KB, MIT)"
  SWIFT_DST="$MODELS_DIR/swift_f0.onnx"
  if [[ -s "$SWIFT_DST" ]]; then
    ok "swift_f0.onnx already at $SWIFT_DST"
  else
    SWIFT_PYPI="$(curl -fsSL https://pypi.org/pypi/swift-f0/json \
                 | python3 -c 'import json,sys;d=json.load(sys.stdin)
[print(f["url"]) for f in d["urls"] if f["packagetype"]=="bdist_wheel"][:1]')"
    [[ -n "$SWIFT_PYPI" ]] || die "could not locate swift-f0 wheel on PyPI"
    TMP_WHL="$(mktemp --suffix=.whl)"
    trap 'rm -f "$TMP_WHL"' RETURN
    curl -L --fail --progress-bar -o "$TMP_WHL" "$SWIFT_PYPI"
    if command -v unzip >/dev/null; then
      unzip -p "$TMP_WHL" swift_f0/model.onnx > "$SWIFT_DST"
    else
      python3 -c "import zipfile,sys
with zipfile.ZipFile(sys.argv[1]) as z, open(sys.argv[2], 'wb') as o:
    o.write(z.read('swift_f0/model.onnx'))" "$TMP_WHL" "$SWIFT_DST"
    fi
    rm -f "$TMP_WHL"
    [[ -s "$SWIFT_DST" ]] || die "swift_f0/model.onnx not found inside wheel"
    ok "swift_f0.onnx extracted ($(du -h "$SWIFT_DST" | cut -f1))"
  fi
fi

# ──────────────────────────────────────────────────────────────────────
if [[ "$DO_CREPE" == "1" ]]; then
  step "CREPE-ONNX (~165 MB total, MIT — yqzhishen/onnxcrepe v1.1.0)"
  ONNXCREPE_TAG="v1.1.0"
  ONNXCREPE_BASE="https://github.com/yqzhishen/onnxcrepe/releases/download/$ONNXCREPE_TAG"
  for size in tiny small medium large full; do
    curl_fetch "$ONNXCREPE_BASE/$size.onnx" "$MODELS_DIR/crepe-$size.onnx"
  done
fi

# ──────────────────────────────────────────────────────────────────────
if [[ "$DO_RMVPE" == "1" ]]; then
  step "RMVPE (~362 MB, Apache-2.0 weights, MIT mirror)"
  curl_fetch \
    "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.onnx" \
    "$MODELS_DIR/rmvpe.onnx"
fi

# ──────────────────────────────────────────────────────────────────────
if [[ "$DO_FCPE" == "1" ]]; then
  step "FCPE (~42 MB, MIT)"
  FCPE_DST="$MODELS_DIR/fcpe.onnx"
  if [[ -s "$FCPE_DST" ]]; then
    ok "fcpe.onnx already at $FCPE_DST"
  else
    # Try pre-built first; fall back to local export.
    if curl -L --fail --silent --show-error -o "$FCPE_DST.partial" "$FCPE_URL" 2>/dev/null \
       && [[ -s "$FCPE_DST.partial" ]]; then
      mv "$FCPE_DST.partial" "$FCPE_DST"
      ok "fcpe.onnx downloaded from $FCPE_URL ($(du -h "$FCPE_DST" | cut -f1))"
    else
      rm -f "$FCPE_DST.partial"
      warn "pre-built FCPE not yet hosted at $FCPE_URL — building locally"
      run_export fcpe_export.py torchfcpe "$FCPE_DST"
    fi
  fi
fi

# ──────────────────────────────────────────────────────────────────────
if [[ "$DO_PESTO" == "1" ]]; then
  step "PESTO (~17 MB) — ⚠ LGPL-3.0 weights"
  if [[ "$PESTO_REQUESTED" != "1" ]]; then
    warn "PESTO is opt-in (LGPL-3.0); pass --pesto explicitly to fetch"
  else
    cat <<'EOF'
  ┌─────────────────────────────────────────────────────────────────┐
  │ PESTO upstream (SonyCSLParis/pesto, pesto-pitch on PyPI) is     │
  │ LGPL-3.0. The exported ONNX inherits LGPL-3.0 by default.       │
  │ Redistribution requires LGPL-3.0 compliance. If you only need   │
  │ permissive backends, skip this and use SwiftF0/CREPE/RMVPE/FCPE.│
  └─────────────────────────────────────────────────────────────────┘
EOF
    run_export pesto_export.py pesto-pitch "$MODELS_DIR/pesto_mir1k_g7_48k_960.onnx"
  fi
fi

# ──────────────────────────────────────────────────────────────────────
printf '\n\033[1;32m== models ready ==\033[0m\n'
printf 'Models directory: %s\n' "$MODELS_DIR"
printf 'Pass any model_path to the corresponding pitch-core-onnx constructor.\n\n'
