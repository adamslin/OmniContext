#!/bin/bash

# RAG 智能问答系统 - Docker 启动脚本
# 同时启动 FastAPI 后端和 Streamlit 前端

set -e

echo "============================================"
echo "🚀 RAG 智能问答系统 - Docker 启动"
echo "============================================"

# 检查 .env 文件
if [ ! -f /app/.env ]; then
    echo "⚠️ 警告: .env 文件不存在"
    echo "   请确保已挂载 .env 文件或设置 GOOGLE_API_KEY 环境变量"
fi

# 检查 GOOGLE_API_KEY
if [ -z "$GOOGLE_API_KEY" ]; then
    if [ -f /app/.env ]; then
        export $(grep -v '^#' /app/.env | xargs)
    fi
fi

if [ -z "$GOOGLE_API_KEY" ] || [ "$GOOGLE_API_KEY" = "your_api_key_here" ]; then
    echo "❌ 错误: GOOGLE_API_KEY 未配置"
    echo "   请设置环境变量或在 .env 文件中配置"
    exit 1
fi

echo "✅ API Key 已配置"

# 启动 FastAPI 后端 (后台运行)
echo ""
echo "🔧 启动 FastAPI 后端 (端口 8000)..."
uvicorn main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
echo "   后端 PID: $BACKEND_PID"

# 等待后端启动
sleep 3

# 检查后端是否启动成功
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo "❌ 后端启动失败"
    exit 1
fi
echo "✅ 后端启动成功"

# 启动 Streamlit 前端 (前台运行)
echo ""
echo "🎨 启动 Streamlit 前端 (端口 8501)..."
echo ""
echo "============================================"
echo "🎉 服务启动成功!"
echo "============================================"
echo ""
echo "📡 后端 API:  http://localhost:8000"
echo "🖥️  前端界面: http://localhost:8501"
echo "📚 API 文档:  http://localhost:8000/docs"
echo ""
echo "============================================"

# Streamlit 在前台运行，保持容器运行
exec streamlit run ui.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false
