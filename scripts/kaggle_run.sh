#!/usr/bin/env bash
# Run one repo script on a private Kaggle T4 kernel and download what it wrote.
#
# Usage:
#   scripts/kaggle_run.sh <run_name> <script> [script args...]
#   scripts/kaggle_run.sh env-check scripts/env_check.py
#
# Env vars (optional):
#   BRANCH   branch to run (default: current branch)
#   SHA      commit to run (default: HEAD). Must already be pushed.
#   KAGGLE   path to the kaggle CLI (default: kaggle on PATH)
#   TIMEOUT  max run time in seconds (default: 2400 = 40 min)
#   DATASETS space-separated private Kaggle datasets to attach, e.g. "desi-tts-own-voice"
#            (mounted read-only at /kaggle/input/<name>/)
#
# Output lands in outputs/kaggle/<run_name>/ (gitignored), including
# out/run_info.json with the exact SHA that ran.
set -euo pipefail

if [[ $# -lt 2 || "$1" == "-h" || "$1" == "--help" ]]; then
  sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'
  exit 0
fi

RUN_NAME="$1"; SCRIPT="$2"; shift 2
# Quote each arg so sentences with spaces survive the trip (POSIX quoting for Kaggle's /bin/sh).
ARGS="$(python3 -c 'import shlex, sys; print(shlex.join(sys.argv[1:]))' "$@")"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
KAGGLE="${KAGGLE:-kaggle}"
BRANCH="${BRANCH:-$(git -C "$ROOT" rev-parse --abbrev-ref HEAD)}"
SHA="${SHA:-$(git -C "$ROOT" rev-parse HEAD)}"
TIMEOUT="${TIMEOUT:-2400}"
DATASETS="${DATASETS:-}"

# The kernel clones from GitHub, so the commit has to be there already.
if ! git -C "$ROOT" branch -r --contains "$SHA" | grep -q "origin/$BRANCH"; then
  echo "Commit $SHA is not pushed to origin/$BRANCH yet. Push first." >&2
  exit 1
fi

USER_NAME="$("$KAGGLE" config view | awk -F': ' '/username/ {print $2}')"
SLUG="desi-tts-$(echo "$RUN_NAME" | tr '[:upper:]_' '[:lower:]-')"
SOURCES="$(for d in $DATASETS; do printf '"%s/%s",' "$USER_NAME" "$d"; done | sed 's/,$//')"
WORK="$(mktemp -d)"
OUT="$ROOT/outputs/kaggle/$RUN_NAME"

# Fill in the template. ARGS goes through Python so quotes can't break the file.
python3 - "$ROOT/kaggle/kernel_template.py" "$WORK/kernel.py" "$BRANCH" "$SHA" "$SCRIPT" "$ARGS" <<'EOF'
import sys
src, dst, branch, sha, script, args = sys.argv[1:]
text = open(src).read()
for key, value in {"__BRANCH__": branch, "__SHA__": sha, "__SCRIPT__": script, "__ARGS__": args}.items():
    text = text.replace(f'"{key}"', repr(value))
open(dst, "w").write(text)
EOF

cat > "$WORK/kernel-metadata.json" <<EOF
{
  "id": "$USER_NAME/$SLUG",
  "title": "$SLUG",
  "code_file": "kernel.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "dataset_sources": [$SOURCES],
  "competition_sources": [],
  "kernel_sources": []
}
EOF

echo "Pushing $SLUG: $SCRIPT $ARGS @ ${SHA:0:7} ($BRANCH)"
"$KAGGLE" kernels push -p "$WORK" --accelerator NvidiaTeslaT4 -t "$TIMEOUT"

# Poll until the run finishes.
START=$(date +%s)
while true; do
  STATUS="$("$KAGGLE" kernels status "$USER_NAME/$SLUG" 2>&1 || true)"
  ELAPSED=$(( $(date +%s) - START ))
  echo "[$((ELAPSED / 60))m$((ELAPSED % 60))s] $STATUS"
  case "$STATUS" in
    *COMPLETE*|*ERROR*|*CANCEL*) break ;;
  esac
  sleep 30
done

mkdir -p "$OUT"
"$KAGGLE" kernels output "$USER_NAME/$SLUG" -p "$OUT" -o >/dev/null
echo "Output in $OUT"
[[ -f "$OUT/out/run_info.json" ]] && cat "$OUT/out/run_info.json"
case "$STATUS" in *COMPLETE*) exit 0 ;; *) exit 1 ;; esac
