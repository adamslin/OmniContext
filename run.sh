#!/bin/bash

# RAG 智能问答系统 - 一键启动脚本
# 同时启动 FastAPI 后端和 Streamlit 前端

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 项目目录
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}🚀 RAG 智能问答系统 - 启动中${NC}"
echo -e "${BLUE}============================================${NC}"

# 检查虚拟环境
if [ -d "venv" ]; then
    echo -e "${GREEN}✅ 检测到虚拟环境${NC}"
    source venv/bin/activate
else
    echo -e "${YELLOW}⚠️ 未检测到虚拟环境，使用系统 Python${NC}"
fi

# 检查依赖
echo -e "${BLUE}📦 检查依赖...${NC}"
python -c "import fastapi, streamlit, langchain" 2>/dev/null || {
    echo -e "${RED}❌ 缺少依赖，请先运行: pip install -r requirements.txt${NC}"
    exit 1
}
echo -e "${GREEN}✅ 依赖检查通过${NC}"

# 检查 .env 文件
if [ ! -f ".env" ]; then
    echo -e "${RED}❌ .env 文件不存在${NC}"
    exit 1
fi

if grep -q "your_api_key_here" .env; then
    echo -e "${RED}❌ 请在 .env 文件中配置 GOOGLE_API_KEY${NC}"
    exit 1
fi
echo -e "${GREEN}✅ 环境配置检查通过${NC}"

# 清理旧进程
cleanup() {
    echo -e "\n${YELLOW}🛑 正在停止服务...${NC}"
    
    # 终止后台进程
    if [ ! -z "$BACKEND_PID" ]; then
        kill $BACKEND_PID 2>/dev/null || true
    fi
    if [ ! -z "$FRONTEND_PID" ]; then
        kill $FRONTEND_PID 2>/dev/null || true
    fi
    
    # 清理端口 (使用 fuser 或跳过)
    fuser -k 8000/tcp 2>/dev/null || true
    fuser -k 8501/tcp 2>/dev/null || true
    
    echo -e "${GREEN}👋 服务已停止${NC}"
    exit 0
}

trap cleanup SIGINT SIGTERM

# 启动后端
echo -e "\n${BLUE}🔧 启动 FastAPI 后端 (端口 8000)...${NC}"
python main.py &
BACKEND_PID=$!
sleep 3

# 检查后端是否启动成功
if curl -s http://localhost:8000/ > /dev/null; then
    echo -e "${GREEN}✅ 后端启动成功${NC}"
else
    echo -e "${RED}❌ 后端启动失败${NC}"
    cleanup
fi

# 启动前端
echo -e "\n${BLUE}🎨 启动 Streamlit 前端 (端口 8501)...${NC}"
streamlit run ui.py --server.port 8501 --server.headless true &
FRONTEND_PID=$!
sleep 3

# 输出访问信息
echo -e "\n${GREEN}============================================${NC}"
echo -e "${GREEN}🎉 服务启动成功!${NC}"
echo -e "${GREEN}============================================${NC}"
echo -e ""
echo -e "📡 后端 API:  ${BLUE}http://localhost:8000${NC}"
echo -e "🖥️  前端界面: ${BLUE}http://localhost:8501${NC}"
echo -e "📚 API 文档:  ${BLUE}http://localhost:8000/docs${NC}"
echo -e ""
echo -e "${YELLOW}按 Ctrl+C 停止所有服务${NC}"
echo -e "${GREEN}============================================${NC}"

# 等待进程
wait
