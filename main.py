"""
RAG Web 应用 - FastAPI 后端
提供文件上传和流式聊天接口
"""

import os
import uuid
import shutil
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# 导入自定义模块
from ingest import (
    load_api_key,
    get_pdf_files,
    load_documents,
    split_documents,
    create_vector_store,
    DATA_DIR,
    DB_DIR
)
from rag_with_history import RAGWithHistory, load_vector_store as load_existing_vector_store


# 全局变量
sessions: dict[str, RAGWithHistory] = {}  # 会话存储
api_key: Optional[str] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理
    """
    global api_key
    
    # 启动时加载 API Key
    load_dotenv()
    try:
        api_key = load_api_key()
        print("✅ API Key 加载成功")
    except ValueError as e:
        print(f"⚠️ {e}")
        print("   请在 .env 文件中配置 GOOGLE_API_KEY")
    
    # 确保目录存在
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DB_DIR.mkdir(parents=True, exist_ok=True)
    
    yield
    
    # 关闭时清理
    sessions.clear()
    print("👋 服务已关闭")


# 创建 FastAPI 应用
app = FastAPI(
    title="RAG 智能问答系统",
    description="基于 Google Gemini 的 RAG 多轮对话系统",
    version="1.0.0",
    lifespan=lifespan
)

# 添加 CORS 中间件（允许 Streamlit 跨域请求）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== 数据模型 ====================

class ChatRequest(BaseModel):
    """聊天请求"""
    session_id: str
    message: str


class ChatResponse(BaseModel):
    """聊天响应"""
    answer: str
    sources: list[dict]


class SessionResponse(BaseModel):
    """会话响应"""
    session_id: str
    message: str


class UploadResponse(BaseModel):
    """上传响应"""
    success: bool
    message: str
    filename: str
    chunks_count: int


# ==================== 工具函数 ====================

def get_or_create_session(session_id: str) -> RAGWithHistory:
    """
    获取或创建会话
    """
    global api_key
    
    if not api_key:
        raise HTTPException(status_code=500, detail="API Key 未配置")
    
    if session_id not in sessions:
        try:
            sessions[session_id] = RAGWithHistory(
                api_key=api_key,
                k=3,
                show_rewritten_query=True
            )
        except FileNotFoundError:
            raise HTTPException(
                status_code=400,
                detail="向量数据库不存在，请先上传 PDF 文件"
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    return sessions[session_id]


async def process_pdf_to_vectors(pdf_path: Path) -> int:
    """
    处理单个 PDF 文件并更新向量库
    
    Returns:
        生成的文本块数量
    """
    global api_key
    
    if not api_key:
        raise HTTPException(status_code=500, detail="API Key 未配置")
    
    # 加载文档
    documents = load_documents([pdf_path])
    
    if not documents:
        raise HTTPException(status_code=400, detail="PDF 解析失败")
    
    # 切分文档
    chunks = split_documents(documents)
    
    if not chunks:
        raise HTTPException(status_code=400, detail="文档切分失败")
    
    # 向量化并存储
    create_vector_store(chunks, api_key)
    
    # 清除所有会话的 RAG 实例，强制重新加载向量库
    sessions.clear()
    
    return len(chunks)


# ==================== API 接口 ====================

@app.get("/")
async def root():
    """
    健康检查
    """
    return {
        "status": "running",
        "service": "RAG 智能问答系统",
        "version": "1.0.0"
    }


@app.post("/session/create", response_model=SessionResponse)
async def create_session():
    """
    创建新会话
    """
    session_id = str(uuid.uuid4())
    
    return SessionResponse(
        session_id=session_id,
        message="会话创建成功"
    )


@app.post("/session/{session_id}/clear")
async def clear_session(session_id: str):
    """
    清空会话历史
    """
    if session_id in sessions:
        sessions[session_id].clear_history()
    
    return {"message": "会话历史已清空"}


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """
    删除会话
    """
    if session_id in sessions:
        del sessions[session_id]
    
    return {"message": "会话已删除"}


@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """
    上传 PDF 文件并更新向量库
    """
    # 验证文件类型
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="只支持 PDF 文件")
    
    # 保存文件
    file_path = DATA_DIR / file.filename
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件保存失败: {e}")
    
    # 处理向量化
    try:
        chunks_count = await process_pdf_to_vectors(file_path)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"向量化处理失败: {e}")
    
    return UploadResponse(
        success=True,
        message="文件上传并处理成功",
        filename=file.filename,
        chunks_count=chunks_count
    )


@app.post("/chat")
async def chat(request: ChatRequest):
    """
    聊天接口（非流式）
    """
    rag = get_or_create_session(request.session_id)
    
    try:
        answer, docs = rag.ask(request.message)
        
        sources = [
            {
                "file": doc.metadata.get("source_file", doc.metadata.get("source", "未知")),
                "page": doc.metadata.get("page", "N/A"),
                "preview": doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
            }
            for doc in docs
        ]
        
        return ChatResponse(answer=answer, sources=sources)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream(session_id: str = Form(...), message: str = Form(...)):
    """
    流式聊天接口
    """
    rag = get_or_create_session(session_id)
    
    async def generate():
        """
        异步生成器，产出流式响应
        """
        try:
            for chunk in rag.ask_stream(message):
                yield chunk
        except Exception as e:
            yield f"\n\n❌ 错误: {e}"
    
    return StreamingResponse(
        generate(),
        media_type="text/plain; charset=utf-8"
    )


@app.get("/chat/{session_id}/sources")
async def get_last_sources(session_id: str):
    """
    获取上次检索的来源
    """
    if session_id not in sessions:
        return {"sources": []}
    
    rag = sessions[session_id]
    docs = getattr(rag, '_last_docs', [])
    
    sources = [
        {
            "file": doc.metadata.get("source_file", doc.metadata.get("source", "未知")),
            "page": doc.metadata.get("page", "N/A"),
            "content": doc.page_content
        }
        for doc in docs
    ]
    
    return {"sources": sources}


@app.get("/documents")
async def list_documents():
    """
    列出已上传的文档
    """
    pdf_files = get_pdf_files()
    
    return {
        "count": len(pdf_files),
        "files": [f.name for f in pdf_files]
    }


# ==================== 启动入口 ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
