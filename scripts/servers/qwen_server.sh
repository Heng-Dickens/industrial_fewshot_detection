#!/bin/bash
LOG_DIR="/mnt/jixie16t/zj/zj/VLLM_SERVER/logs"
mkdir -p "$LOG_DIR"

MODEL_PATH="/mnt/jixie16t/dataset/ModelScopeModels/Qwen/Qwen3.8-27B-NVFP4"
# MODEL_PATH="/mnt/second8t/datasets/llm_models/ornith/ornith-ai/Ornith-1.5-9B"
PORT=2334
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/sglang_qwen3.8_${TIMESTAMP}.log"
PID_FILE="${LOG_DIR}/sglang.pid"

export SGLANG_USE_MODELSCOPE=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "[WARN] 服务已在运行 (PID $(cat "$PID_FILE"))。先停止: kill $(cat "$PID_FILE")"
    exit 1
fi

CUDA_VISIBLE_DEVICES=4,5,6,7 nohup sglang serve \
  --model-path "$MODEL_PATH" \
  --served-model-name Qwen3.8-27B-NVFP4 \
  --host 0.0.0.0 \
  --port "$PORT" \
  --tp 4 \
  --trust-remote-code \
  --context-length 262144 \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.55 \
  --max-running-requests 32 \
  --chunked-prefill-size 50000 \
  --max-prefill-tokens 50000 \
  --cuda-graph-max-bs-decode 64 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --mamba-ssm-dtype bfloat16 \
  --schedule-policy lpm \
  --linear-replayssm-cache-len 32 \
  --speculative-algo NEXTN \
  --speculative-eagle-topk 1 \
  --speculative-num-steps 3 \
  --speculative-num-draft-tokens 4 \
  > "$LOG_FILE" 2>&1 &

PID=$!
echo "$PID" > "$PID_FILE"
echo "[INFO] 服务已启动，PID: $PID"
echo "[INFO] 日志: $LOG_FILE"
echo "[INFO] 测试: curl http://127.0.0.1:$PORT/v1/models"


