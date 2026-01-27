"""
RAG Web 应用 - Streamlit 前端
提供类似 ChatGPT 的聊天界面
"""

import uuid
import requests
import streamlit as st

# ==================== 配置 ====================

API_BASE_URL = "http://localhost:8000"

# 页面配置
st.set_page_config(
    page_title="RAG 智能问答系统",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== 样式 ====================

st.markdown("""
<style>
    /* 聊天消息样式 */
    .user-message {
        background-color: #e3f2fd;
        padding: 12px 16px;
        border-radius: 16px;
        margin: 8px 0;
        max-width: 80%;
        margin-left: auto;
    }
    
    .ai-message {
        background-color: #f5f5f5;
        padding: 12px 16px;
        border-radius: 16px;
        margin: 8px 0;
        max-width: 80%;
    }
    
    /* 来源卡片样式 */
    .source-card {
        background-color: #fff3e0;
        padding: 8px 12px;
        border-radius: 8px;
        margin: 4px 0;
        font-size: 0.85em;
        border-left: 3px solid #ff9800;
    }
    
    /* 隐藏 Streamlit 默认的页脚 */
    footer {visibility: hidden;}
    
    /* 聊天容器 */
    .chat-container {
        max-height: 500px;
        overflow-y: auto;
        padding: 16px;
    }
</style>
""", unsafe_allow_html=True)


# ==================== 会话状态初始化 ====================

def init_session_state():
    """初始化会话状态"""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "sources" not in st.session_state:
        st.session_state.sources = []


init_session_state()


# ==================== API 调用函数 ====================

def check_backend_health() -> bool:
    """检查后端是否运行"""
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=5)
        return response.status_code == 200
    except:
        return False


def upload_file(file) -> dict:
    """上传文件到后端"""
    try:
        files = {"file": (file.name, file.getvalue(), "application/pdf")}
        response = requests.post(f"{API_BASE_URL}/upload", files=files, timeout=120)
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"success": False, "message": f"上传失败: {e}"}


def get_documents() -> list:
    """获取已上传的文档列表"""
    try:
        response = requests.get(f"{API_BASE_URL}/documents", timeout=10)
        data = response.json()
        return data.get("files", [])
    except:
        return []


def chat_stream(session_id: str, message: str):
    """流式聊天"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/chat/stream",
            data={"session_id": session_id, "message": message},
            stream=True,
            timeout=120
        )
        
        for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
            if chunk:
                yield chunk
                
    except requests.exceptions.RequestException as e:
        yield f"❌ 请求失败: {e}"


def get_sources(session_id: str) -> list:
    """获取参考来源"""
    try:
        response = requests.get(
            f"{API_BASE_URL}/chat/{session_id}/sources",
            timeout=10
        )
        data = response.json()
        return data.get("sources", [])
    except:
        return []


def clear_history(session_id: str):
    """清空对话历史"""
    try:
        requests.post(f"{API_BASE_URL}/session/{session_id}/clear", timeout=10)
    except:
        pass


# ==================== 侧边栏 ====================

with st.sidebar:
    st.title("📚 知识库管理")
    
    # 后端状态检查
    backend_healthy = check_backend_health()
    
    if backend_healthy:
        st.success("✅ 后端服务正常运行")
    else:
        st.error("❌ 后端服务未启动")
        st.info("请先运行: `python main.py`")
    
    st.divider()
    
    # 文件上传
    st.subheader("📤 上传文档")
    
    uploaded_file = st.file_uploader(
        "选择 PDF 文件",
        type=["pdf"],
        help="上传 PDF 文件到知识库"
    )
    
    if uploaded_file is not None:
        if st.button("📥 上传并处理", type="primary", use_container_width=True):
            with st.spinner("正在处理文档，请稍候..."):
                result = upload_file(uploaded_file)
                
                if result.get("success"):
                    st.success(f"✅ 知识库已更新!")
                    st.info(f"📄 文件: {result.get('filename')}")
                    st.info(f"📊 生成 {result.get('chunks_count')} 个文本块")
                else:
                    st.error(f"❌ {result.get('message', '上传失败')}")
    
    st.divider()
    
    # 已上传文档列表
    st.subheader("📋 已有文档")
    
    if backend_healthy:
        documents = get_documents()
        
        if documents:
            for doc in documents:
                st.markdown(f"📄 {doc}")
        else:
            st.info("暂无文档，请上传 PDF 文件")
    
    st.divider()
    
    # 会话管理
    st.subheader("⚙️ 会话管理")
    
    if st.button("🗑️ 清空对话", use_container_width=True):
        clear_history(st.session_state.session_id)
        st.session_state.messages = []
        st.session_state.sources = []
        st.success("对话已清空")
        st.rerun()
    
    if st.button("🔄 新建会话", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.sources = []
        st.success("新会话已创建")
        st.rerun()
    
    st.divider()
    
    # 参考来源
    if st.session_state.sources:
        st.subheader("📚 参考来源")
        
        for i, source in enumerate(st.session_state.sources, 1):
            with st.expander(f"[{i}] {source.get('file', '未知')} (页码: {source.get('page', 'N/A')})"):
                st.markdown(source.get("content", "无内容")[:500] + "...")


# ==================== 主界面 ====================

st.title("🤖 RAG 智能问答系统")
st.caption("基于 Google Gemini 的多轮对话 RAG 系统")

# 显示历史消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 聊天输入
if prompt := st.chat_input("输入你的问题...", disabled=not backend_healthy):
    # 添加用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # AI 回复
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        # 流式获取回复
        for chunk in chat_stream(st.session_state.session_id, prompt):
            full_response += chunk
            response_placeholder.markdown(full_response + "▌")
        
        response_placeholder.markdown(full_response)
    
    # 保存 AI 回复
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    
    # 获取参考来源
    st.session_state.sources = get_sources(st.session_state.session_id)
    
    # 刷新侧边栏显示来源
    st.rerun()

# 空状态提示
if not st.session_state.messages:
    st.info("👋 欢迎使用 RAG 智能问答系统！\n\n"
            "1. 在左侧上传 PDF 文档构建知识库\n"
            "2. 在下方输入框中提问\n"
            "3. AI 将基于文档内容回答您的问题")

# 底部提示
if st.session_state.sources:
    st.caption(f"📚 本次回答参考了 {len(st.session_state.sources)} 个文档片段，详情请查看左侧边栏")
