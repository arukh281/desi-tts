#!/usr/bin/env bash
# Upload a local folder to Kaggle as a PRIVATE dataset (new version if it exists).
# My voice is never public: this never passes --public.
#
# Usage:
#   scripts/upload_dataset.sh <folder> <dataset-slug> ["version message"]
#   scripts/upload_dataset.sh data/processed desi-tts-own-voice "session 1"
#   scripts/upload_dataset.sh data/raw/reference desi-tts-reference
#
# Env: KAGGLE = path to the kaggle CLI (default: kaggle on PATH)
set -euo pipefail

if [[ $# -lt 2 || "$1" == "-h" || "$1" == "--help" ]]; then
  sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'
  exit 0
fi

DIR="$1"; SLUG="$2"; MSG="${3:-update}"
KAGGLE="${KAGGLE:-kaggle}"
USER_NAME="$("$KAGGLE" config view | awk -F': ' '/username/ {print $2}')"
[[ -d "$DIR" ]] || { echo "No such folder: $DIR" >&2; exit 1; }

cat > "$DIR/dataset-metadata.json" <<EOF
{
  "title": "$SLUG",
  "id": "$USER_NAME/$SLUG",
  "licenses": [{"name": "other"}]
}
EOF

if "$KAGGLE" datasets status "$USER_NAME/$SLUG" >/dev/null 2>&1; then
  "$KAGGLE" datasets version -p "$DIR" -m "$MSG" -r zip
else
  "$KAGGLE" datasets create -p "$DIR" -r zip   # private unless --public is given
fi
rm -f "$DIR/dataset-metadata.json"
echo "Kaggle dataset $USER_NAME/$SLUG (private). Mount it in a kernel at /kaggle/input/$SLUG"
