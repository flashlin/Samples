#!/bin/bash
#
# Example runner for Confucius4-R2T2 streaming / one-shot inference.
#
# Usage:
#   ./run_example.sh <path/to/audio.wav> --model_path <path/to/Confucius4-R2T2> [options]
#   MODEL_PATH=/path/to/Confucius4-R2T2 ./run_example.sh <path/to/audio.wav>
#
# Required (either via CLI flag or environment variable):
#   --model_path <path>   Path or HF repo id of the Confucius4-R2T2 checkpoint.
#                         (Or set MODEL_PATH=<path> in the environment.)
#   <audio>               First positional argument, path to the input audio.
#                         (Or pass --audio <path>, or set AUDIO=<path>.)
#
# Optional CLI flags (each also has a matching environment variable):
#   --audio <path>              AUDIO
#   --infer_mode <mode>         INFER_MODE          stream_vllm | onetime_vllm  (default: stream_vllm)
#   --language <lang>           LANGUAGE            Chinese | English | Japanese | Korean | ... (default: Chinese)
#   --chunk_size_ms <int>       CHUNK_SIZE_MS       Streaming chunk size in ms (default: 160)
#   --lookahead_ms <int>        LOOKAHEAD_MS        Streaming lookahead in ms  (default: 160)
#   --unfixed_token_num <int>   UNFIXED_TOKEN_NUM   Number of unfixed trailing tokens (default: 1)
#   --context <str>             CONTEXT             Optional hotword / context hint
#   --log_file <path>           LOG_FILE            Where to write logs (default: run_example.log)
#   --gpu <id>                  CUDA_VISIBLE_DEVICES  GPU id to use (default: 0)
#
# Both `--key value` and `--key=value` forms are accepted.

set -euo pipefail

# Resolve the script directory so this script works from anywhere.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Make the local package importable without installation.
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

# ========== Defaults (env vars take precedence over hard-coded defaults) ==========

MODEL_PATH="${MODEL_PATH:-}"
AUDIO="${AUDIO:-}"
INFER_MODE="${INFER_MODE:-stream_vllm}"
LANGUAGE="${LANGUAGE:-Chinese}"
CHUNK_SIZE_MS="${CHUNK_SIZE_MS:-160}"
LOOKAHEAD_MS="${LOOKAHEAD_MS:-160}"
UNFIXED_TOKEN_NUM="${UNFIXED_TOKEN_NUM:-1}"
CONTEXT="${CONTEXT:-}"
LOG_FILE="${LOG_FILE:-run_example.log}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"

usage() {
    cat >&2 <<EOF
Usage:
  $0 <path/to/audio.wav> --model_path <path/to/Confucius4-R2T2> [options]
  MODEL_PATH=/path/to/model $0 <path/to/audio.wav>

Options (all accept both --key value and --key=value):
  --model_path <path>
  --audio <path>
  --infer_mode <stream_vllm|onetime_vllm>
  --language <str>
  --chunk_size_ms <int>
  --lookahead_ms <int>
  --unfixed_token_num <int>
  --context <str>
  --log_file <path>
  --gpu <cuda_visible_devices>
  -h, --help
EOF
}

# ========== Argument parsing ==========

# Helper: given "--key=value", set VALUE to "value"; otherwise consume the next arg.
# Sets globals PARSED_VALUE and SHIFT_COUNT.
parse_value() {
    local arg="$1"
    local next="${2-}"
    if [[ "${arg}" == *=* ]]; then
        PARSED_VALUE="${arg#*=}"
        SHIFT_COUNT=1
    else
        if [[ $# -lt 2 ]]; then
            echo "Error: ${arg%%=*} requires a value." >&2
            exit 1
        fi
        PARSED_VALUE="${next}"
        SHIFT_COUNT=2
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_path|--model_path=*)
            parse_value "$1" "${2-}"; MODEL_PATH="${PARSED_VALUE}"; shift "${SHIFT_COUNT}" ;;
        --audio|--audio=*)
            parse_value "$1" "${2-}"; AUDIO="${PARSED_VALUE}"; shift "${SHIFT_COUNT}" ;;
        --infer_mode|--infer_mode=*)
            parse_value "$1" "${2-}"; INFER_MODE="${PARSED_VALUE}"; shift "${SHIFT_COUNT}" ;;
        --language|--language=*)
            parse_value "$1" "${2-}"; LANGUAGE="${PARSED_VALUE}"; shift "${SHIFT_COUNT}" ;;
        --chunk_size_ms|--chunk_size_ms=*)
            parse_value "$1" "${2-}"; CHUNK_SIZE_MS="${PARSED_VALUE}"; shift "${SHIFT_COUNT}" ;;
        --lookahead_ms|--lookahead_ms=*)
            parse_value "$1" "${2-}"; LOOKAHEAD_MS="${PARSED_VALUE}"; shift "${SHIFT_COUNT}" ;;
        --unfixed_token_num|--unfixed_token_num=*)
            parse_value "$1" "${2-}"; UNFIXED_TOKEN_NUM="${PARSED_VALUE}"; shift "${SHIFT_COUNT}" ;;
        --context|--context=*)
            parse_value "$1" "${2-}"; CONTEXT="${PARSED_VALUE}"; shift "${SHIFT_COUNT}" ;;
        --log_file|--log_file=*)
            parse_value "$1" "${2-}"; LOG_FILE="${PARSED_VALUE}"; shift "${SHIFT_COUNT}" ;;
        --gpu|--gpu=*)
            parse_value "$1" "${2-}"; GPU="${PARSED_VALUE}"; shift "${SHIFT_COUNT}" ;;
        -h|--help)
            usage; exit 0 ;;
        --*)
            echo "Error: unknown option: $1" >&2
            usage
            exit 1 ;;
        *)
            # First bare positional becomes AUDIO if not already set.
            if [[ -z "${AUDIO}" ]]; then
                AUDIO="$1"
            else
                echo "Error: unexpected positional argument: $1" >&2
                usage
                exit 1
            fi
            shift ;;
    esac
done

export CUDA_VISIBLE_DEVICES="${GPU}"

# ========== Validation ==========

if [[ -z "${MODEL_PATH}" ]]; then
    echo "Error: MODEL_PATH is not set." >&2
    echo "Pass --model_path <path>, or set MODEL_PATH=<path> in the environment." >&2
    exit 1
fi

if [[ -z "${AUDIO}" ]]; then
    echo "Error: no audio file provided." >&2
    usage
    exit 1
fi

if [[ ! -f "${AUDIO}" ]]; then
    echo "Error: audio file not found: ${AUDIO}" >&2
    exit 1
fi

# ========== Run ==========

echo "Model:     ${MODEL_PATH}"
echo "Audio:     ${AUDIO}"
echo "Mode:      ${INFER_MODE}"
echo "Language:  ${LANGUAGE:-<Default>}"
echo "GPU(s):    ${CUDA_VISIBLE_DEVICES}"

python -u "${SCRIPT_DIR}/example.py" \
    --audio "${AUDIO}" \
    --model_path "${MODEL_PATH}" \
    --infer_mode "${INFER_MODE}" \
    --language "${LANGUAGE}" \
    --chunk_size_ms "${CHUNK_SIZE_MS}" \
    --lookahead_ms "${LOOKAHEAD_MS}" \
    --unfixed_token_num "${UNFIXED_TOKEN_NUM}" \
    --context "${CONTEXT}" \
    2>&1 | tee "${LOG_FILE}"
