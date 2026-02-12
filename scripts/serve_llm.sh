#!/bin/bash

# serve_llm.sh: สคริปต์สำหรับ Host Qwen3-8B
# ตำแหน่งจัดเก็บ: walkie-master/scripts/

MODE=${1:-vllm}

echo "🧠 กำลังเริ่มต้นการ Host โมเดล: Qwen/Qwen3-8B"

if [ "$MODE" == "vllm" ]; then
    # ใช้ vLLM เพื่อประสิทธิภาพสูงสุดบน RTX 5090
    python3 -m vllm.entrypoints.openai.api_server \
        --model "Qwen/Qwen3-8B" \
        --served-model-name "qwen3-8b" \
        --port 8000 \
        --gpu-memory-utilization 0.8 \
        --dtype float16 \
        --trust-remote-code

elif [ "$MODE" == "ollama" ]; then
    # ถ้าใช้ Ollama ให้มั่นใจว่าได้ pull model นี้มาแล้ว
    ollama run qwen3:8b
fi
