#!/bin/bash
# Usage:
#   ./run_start_server.sh <action> [options]
#
# Actions:
#   start      Start the ASR websocket server
#   kill       Stop the running server
#   restart    Stop and then start the server
#
# Options (flags override environment variables; both forms `--key=value` and
# `--key value` are accepted):
#   -m, --model_path <path>   Path or HF repo id of the Confucius4-R2T2 checkpoint
#                             (env: ASR_MODEL_PATH). Required for start/restart.
#   -v, --vad_model_path <path>
#                             Path to the FireRedVAD Stream-VAD model
#                             (env: VAD_MODEL_PATH, default: checkpoints/vad/Stream-VAD).
#   -h, --host <name>         Host tag used only in the log file name
#                             (env: HOST_TAG, default: localhost).
#   -p, --port <port>         Websocket port to bind (env: PORT, default: 8272).
#   -g, --gpu <ids>           GPUs to expose via CUDA_VISIBLE_DEVICES
#                             (env: CUDA_VISIBLE_DEVICES, default: 0).
#       --help                Show this help message and exit.
#
# Examples:
#   ./run_start_server.sh start --model_path /path/to/Confucius4-R2T2
#   ./run_start_server.sh start --model_path=/path/to/Confucius4-R2T2 --vad_model_path /path/to/Stream-VAD --port 8272 --gpu 0
#   ASR_MODEL_PATH=/path/to/Confucius4-R2T2 VAD_MODEL_PATH=/path/to/Stream-VAD ./run_start_server.sh restart

set -u

# Resolve the script directory so this script works from anywhere.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Make the local package importable without installation.
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

# ----- defaults (can be overridden by env vars, then by CLI flags) -----
MODEL_PATH="${ASR_MODEL_PATH:-}"
VAD_MODEL_PATH="${VAD_MODEL_PATH:-${SCRIPT_DIR}/checkpoints/vad/Stream-VAD}"
HOST_TAG="${HOST_TAG:-localhost}"
PORT="${PORT:-8272}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"

print_usage() {
    sed -n '2,26p' "$0"
}

# ----- argument parsing -----
if [ $# -lt 1 ]; then
    print_usage
    exit 1
fi

ACTION="$1"
shift

while [ $# -gt 0 ]; do
    case "$1" in
        -m|--model_path)
            MODEL_PATH="$2"; shift 2 ;;
        --model_path=*)
            MODEL_PATH="${1#*=}"; shift ;;
        --vad_model_path)
            VAD_MODEL_PATH="$2"; shift 2 ;;
        --vad_model_path=*)
            VAD_MODEL_PATH="${1#*=}"; shift ;;
        -h|--host)
            HOST_TAG="$2"; shift 2 ;;
        --host=*)
            HOST_TAG="${1#*=}"; shift ;;
        -p|--port)
            PORT="$2"; shift 2 ;;
        --port=*)
            PORT="${1#*=}"; shift ;;
        -g|--gpu)
            GPU="$2"; shift 2 ;;
        --gpu=*)
            GPU="${1#*=}"; shift ;;
        --help)
            print_usage; exit 0 ;;
        *)
            echo "Unknown option: $1" >&2
            print_usage
            exit 1 ;;
    esac
done

export CUDA_VISIBLE_DEVICES=$GPU

kill_processes() {
    echo "正在停止进程..."

    # 找到主进程ID
    MAIN_PID=$(ps -ef | grep '[p]ython.*ws_server.py' | head -1 | awk '{print $2}')

    if [ -z "$MAIN_PID" ]; then
        echo "没有找到运行中的进程"
        return 0
    fi

    echo "找到主进程PID: $MAIN_PID"

    # 获取进程组ID
    PGID=$(ps -o pgid= -p "$MAIN_PID" | tr -d ' ')

    if [ -z "$PGID" ]; then
        echo "无法获取进程组ID，直接杀死主进程"
        kill -9 "$MAIN_PID" 2>/dev/null
        return $?
    fi

    echo "进程组ID: $PGID"
    echo "杀掉整个进程组..."

    # 杀掉进程组
    pkill -9 -g "$PGID" 2>/dev/null

    # 检查是否成功
    sleep 1
    if ps -p "$MAIN_PID" > /dev/null 2>&1; then
        echo "警告：进程可能仍在运行"
        return 1
    else
        echo "进程已成功停止"
        return 0
    fi
}

start_process() {
    echo "启动进程..."

    if [ -z "$MODEL_PATH" ]; then
        echo "错误: 未指定模型路径。请通过 --model_path <path> 或环境变量 ASR_MODEL_PATH 提供。" >&2
        return 1
    fi

    # 检查是否已有进程在运行
    if ps -ef | grep -q '[p]ython.*ws_server.py'; then
        echo "已有进程在运行，请先停止"
        return 1
    fi

    local log_file="nohup_service_ws_${HOST_TAG}_${PORT}.log"

    echo "  MODEL_PATH     = $MODEL_PATH"
    echo "  VAD_MODEL_PATH = $VAD_MODEL_PATH"
    echo "  HOST_TAG       = $HOST_TAG"
    echo "  PORT       = $PORT"
    echo "  GPU        = $GPU"
    echo "  LOG_FILE   = $log_file"

    # 启动新进程
    setsid nohup python -u ./ws_server.py \
        --port "$PORT" \
        --asr_model_path "$MODEL_PATH" \
        --vad_model_path "$VAD_MODEL_PATH" \
        > "$log_file" 2>&1 &

    NEW_PID=$!
    echo "进程已启动，PID: $NEW_PID"
    return 0
}

restart_process() {
    kill_processes
    sleep 2
    start_process
}

# Main logic based on the provided action
case "$ACTION" in
    kill)
        kill_processes
        ;;
    start)
        start_process
        ;;
    restart)
        restart_process
        ;;
    *)
        print_usage
        exit 1
        ;;
esac
