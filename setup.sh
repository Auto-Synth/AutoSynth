#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
role="${1:-train}"
case "$role" in
  train) requirements=requirements.txt ;;
  text) requirements=requirements-text.txt ;;
  languagebind) requirements=requirements-languagebind.txt ;;
  distribution) requirements=requirements-distribution.txt ;;
  *) echo 'Usage: bash setup.sh {train|text|languagebind|distribution}'; exit 2 ;;
esac
python_bin="${PYTHON_BIN:-python3.10}"
"$python_bin" -m venv ".venv/$role"
".venv/$role/bin/python" -m pip install -r "$requirements"
if [[ "$role" == languagebind ]]; then
  mkdir -p third_party
  if [[ ! -d third_party/LanguageBind ]]; then
    git clone https://github.com/PKU-YuanGroup/LanguageBind.git third_party/LanguageBind
  fi
  git -C third_party/LanguageBind checkout --detach 7070c53375661cdb235801176b564b45f96f0648
fi
