#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="${VENV_PATH:-$ROOT_DIR/venv-browse-dim}"
PYTHON_BIN="$VENV_PATH/bin/python"

print_usage() {
	cat <<'EOF'
Usage:
  ./run_browser_with_ollama.sh "task text"
  ./run_browser_with_ollama.sh --url https://example.com "task text"
  ./run_browser_with_ollama.sh --usecases

Options:
  -t, --task     Task text (can also be provided as a positional argument)
  -u, --url      Direct URL to open (sets direct_url mode)
  -o, --output   Output markdown path (default: output-results/output.md)
  -m, --model    Ollama model name (TEXT_LLM_MODEL/BROWSER_USE_LLM_MODEL)
      --ollama   Ollama endpoint (default: http://localhost:11434)
      --google   Force google_search mode
      --usecases Run built-in usecases
  -h, --help     Show this help
EOF
}

TASK_ARG=""
TASK_MODE_ARG=""
TASK_URL_ARG=""
OUTPUT_PATH_ARG=""
MODEL_ARG=""
OLLAMA_ARG=""
USECASES="0"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
	case "$1" in
		--usecases)
			USECASES="1"
			shift
			;;
		--google)
			TASK_MODE_ARG="google_search"
			shift
			;;
		--ollama)
			OLLAMA_ARG="${2:-}"
			shift 2
			;;
		-t|--task)
			TASK_ARG="${2:-}"
			shift 2
			;;
		-u|--url)
			TASK_URL_ARG="${2:-}"
			TASK_MODE_ARG="direct_url"
			shift 2
			;;
		-o|--output)
			OUTPUT_PATH_ARG="${2:-}"
			shift 2
			;;
		-m|--model)
			MODEL_ARG="${2:-}"
			shift 2
			;;
		-h|--help)
			print_usage
			exit 0
			;;
		--)
			shift
			EXTRA_ARGS+=("$@")
			break
			;;
		-*)
			echo "Unknown option: $1" >&2
			print_usage >&2
			exit 2
			;;
		*)
			EXTRA_ARGS+=("$1")
			shift
			;;
	esac
done

if [[ -z "$TASK_ARG" && ${#EXTRA_ARGS[@]} -gt 0 ]]; then
	TASK_ARG="${EXTRA_ARGS[*]}"
fi

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
export OUTPUT_PATH="${OUTPUT_PATH:-$ROOT_DIR/output-results/output.md}"
export RUN_USECASES="0"
export RUN_USECASES_FORCE="0"
export LLM_TIMEOUT="${LLM_TIMEOUT:-240}"
export AGENT_MAX_STEPS="${AGENT_MAX_STEPS:-120}"
export AGENT_STEP_TIMEOUT="${AGENT_STEP_TIMEOUT:-180}"
export AGENT_USE_JUDGE="${AGENT_USE_JUDGE:-1}"

if [[ -n "$MODEL_ARG" ]]; then
	export BROWSER_USE_LLM_MODEL="$MODEL_ARG"
	export TEXT_LLM_MODEL="$MODEL_ARG"
fi
if [[ -n "$OLLAMA_ARG" ]]; then
	export OLLAMA_ENDPOINT="$OLLAMA_ARG"
fi
if [[ -n "$OUTPUT_PATH_ARG" ]]; then
	export OUTPUT_PATH="$OUTPUT_PATH_ARG"
fi
IGNORE_ENV_OVERRIDES="0"
if [[ -n "$TASK_ARG" || "$USECASES" == "1" ]]; then
	IGNORE_ENV_OVERRIDES="1"
fi
if [[ -n "$TASK_MODE_ARG" ]]; then
	export TASK_MODE="$TASK_MODE_ARG"
elif [[ "$IGNORE_ENV_OVERRIDES" == "1" ]]; then
	export TASK_MODE="google_search"
else
	export TASK_MODE="${TASK_MODE:-google_search}"
fi

if [[ -n "$TASK_URL_ARG" ]]; then
	export TASK_URL="$TASK_URL_ARG"
elif [[ "$IGNORE_ENV_OVERRIDES" == "1" ]]; then
	export TASK_URL=""
else
	export TASK_URL="${TASK_URL:-}"
fi

if [[ "$USECASES" == "1" ]]; then
	export RUN_USECASES="1"
	export RUN_USECASES_FORCE="1"
else
	if [[ -n "$TASK_ARG" ]]; then
		export TASK="$TASK_ARG"
		export TASK_EXPLICIT="1"
	elif [[ -n "${TASK:-}" ]]; then
		export TASK_EXPLICIT="1"
	else
		print_usage >&2
		exit 2
	fi
fi

exec "$PYTHON_BIN" "$ROOT_DIR/main/run_browser_tool.py"
