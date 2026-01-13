#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="${VENV_PATH:-$ROOT_DIR/venv-browse-dim}"
PYTHON_BIN="$VENV_PATH/bin/python"

if [[ ! -x "$PYTHON_BIN" ]]; then
	echo "Missing venv at $VENV_PATH. Create it with:" >&2
	echo "  uv venv --python 3.11 venv-browse-dim" >&2
	echo "  source ./venv-browse-dim/bin/activate" >&2
	echo "  uv sync" >&2
	echo "  playwright install" >&2
	exit 1
fi

export BROWSER_USE_LLM_MODEL="${BROWSER_USE_LLM_MODEL:-qwen3-vl:8b}"
export TEXT_LLM_MODEL="${TEXT_LLM_MODEL:-$BROWSER_USE_LLM_MODEL}"
export OLLAMA_ENDPOINT="${OLLAMA_ENDPOINT:-http://localhost:11434}"
export TASK="${TASK:-XAUUSD price today}"
export TASK_MODE="${TASK_MODE:-google_search}"
export TASK_URL="${TASK_URL:-}"
export SEARCH_QUERY="${SEARCH_QUERY:-}"
export OUTPUT_PATH="${OUTPUT_PATH:-$ROOT_DIR/output-results/output.md}"
export RUN_USECASES="${RUN_USECASES:-0}"
export LLM_TIMEOUT="${LLM_TIMEOUT:-240}"
export AGENT_MAX_STEPS="${AGENT_MAX_STEPS:-120}"
export AGENT_STEP_TIMEOUT="${AGENT_STEP_TIMEOUT:-180}"
export AGENT_USE_JUDGE="${AGENT_USE_JUDGE:-1}"

exec "$PYTHON_BIN" "$ROOT_DIR/main/run_browser_tool.py"
