#!/usr/bin/env bash

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
cd "$PROJECT_ROOT"

GPUS=()
AUTO_RESUME=false
CONFIG=""
PORT=${PORT:-4321}

while [[ $# -gt 0 ]]; do
    case $1 in
        --resume) AUTO_RESUME=true; shift ;;
        --gpus) shift; IFS=',' read -ra GPUS <<< "$1"; shift ;;
        --port) shift; PORT=$1; shift ;;
        *.yml) CONFIG=$1; shift ;;
        *) shift ;;
    esac
done

if [ -z "$CONFIG" ]; then
    echo "Usage: bash scripts/train.sh <config.yml> [--gpus 0,1,...] [--port PORT] [--resume]"
    exit 1
fi

if [ ${#GPUS[@]} -gt 0 ]; then
    export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]}")
    NUM_GPUS=${#GPUS[@]}
else
    NUM_GPUS=1
fi

LOG_NAME=$(basename "$CONFIG" .yml)
LOG_FILE="logs/${LOG_NAME}.log"
mkdir -p logs

TRAIN_CMD="basicsr/train.py -opt $CONFIG"
[ "$AUTO_RESUME" = true ] && TRAIN_CMD="$TRAIN_CMD --auto_resume"

if [ "$NUM_GPUS" -gt 1 ]; then
    nohup torchrun --nproc_per_node=$NUM_GPUS --master_port=$PORT \
        $TRAIN_CMD --launcher pytorch > "$LOG_FILE" 2>&1 &
else
    nohup python3 $TRAIN_CMD > "$LOG_FILE" 2>&1 &
fi

echo "Logging to $LOG_FILE (PID: $!)"
