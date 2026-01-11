#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="${VENV_PATH:-$ROOT_DIR/venv-browse-dim}"
PYTHON_BIN="$VENV_PATH/bin/python"

if [[ ! -x "$PYTHON_BIN" ]]; then
	echo "Missing venv at $VENV_PATH. Create it with:" >&2
	echo "  python3 -m venv venv-browse-dim" >&2
	echo "  ./venv-browse-dim/bin/pip install -e ." >&2
	echo "  ./venv-browse-dim/bin/playwright install" >&2
	exit 1
fi

export BROWSER_USE_LLM_MODEL="${BROWSER_USE_LLM_MODEL:-qwen3-vl:8b}"
export TEXT_LLM_MODEL="${TEXT_LLM_MODEL:-$BROWSER_USE_LLM_MODEL}"
export OLLAMA_ENDPOINT="${OLLAMA_ENDPOINT:-http://localhost:11434}"
export TASK="${TASK:-XAUUSD price today}"
export LLM_TIMEOUT="${LLM_TIMEOUT:-300}"
export JUDGE_TIMEOUT="${JUDGE_TIMEOUT:-60}"
export TRAVEL_USE_AGENT="${TRAVEL_USE_AGENT:-0}"

exec "$PYTHON_BIN" "$ROOT_DIR/main/run_browser_with_ollama.py"
