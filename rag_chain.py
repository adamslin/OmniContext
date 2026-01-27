"""
RAG 在线问答脚本 - 检索增强生成
基于向量库检索相关文档片段，结合 Gemini 模型生成回答
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel


# 项目路径配置
BASE_DIR = Path(__file__).parent
DB_DIR = BASE_DIR / "db"


def load_api_key() -> str:
    """
    从 .env 文件读取 Google API Key
    """
    load_dotenv()
    
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key or api_key == "your_api_key_here":
        raise ValueError(
            "❌ 请在 .env 文件中设置有效的 GOOGLE_API_KEY\n"
            "   获取地址: https://aistudio.google.com/app/apikey"
        )
    
    return api_key


def load_vector_store(api_key: str) -> Chroma:
    """
    加载已存在的 Chroma 向量数据库
    
    Args:
        api_key: Google API Key
    
    Returns:
        Chroma 向量数据库实例
    """
    if not DB_DIR.exists() or not any(DB_DIR.iterdir()):
        raise FileNotFoundError(
            f"❌ 向量数据库不存在或为空: {DB_DIR}\n"
            "   请先运行 python ingest.py 构建向量库"
        )
    
    # 使用与 ingest.py 相同的向量化模型
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=api_key,
    )
    
    # 加载已存在的向量库
    vector_store = Chroma(
        persist_directory=str(DB_DIR),
        embedding_function=embeddings,
        collection_name="rag_documents"
    )
    
    return vector_store


def create_retriever(vector_store: Chroma, k: int = 3):
    """
    创建检索器
    
    Args:
        vector_store: Chroma 向量数据库
        k: 返回最相关的 k 个文档片段
    
    Returns:
        检索器实例
    """
    retriever = vector_store.as_retriever(
        search_type="similarity",  # 相似度搜索
        search_kwargs={
            "k": k  # 返回 top-k 个最相关的文档
        }
    )
    
    return retriever


def get_rag_prompt() -> ChatPromptTemplate:
    """
    创建 RAG 提示词模板
    
    Returns:
        ChatPromptTemplate 实例
    """
    template = """你是一个专业的助手。请根据以下提供的上下文内容回答用户问题。

## 重要规则：
1. 只能根据提供的上下文回答问题
2. 如果上下文中找不到答案，请诚实地回答："抱歉，根据您提供的文档，我无法回答这个问题。"
3. 不要胡编乱造或添加上下文中没有的信息
4. 回答要简洁、准确、有条理
5. 如果需要，可以引用上下文中的具体内容

## 上下文内容：
{context}

## 用户问题：
{question}

## 你的回答："""

    prompt = ChatPromptTemplate.from_template(template)
    
    return prompt


def format_docs(docs) -> str:
    """
    格式化检索到的文档片段
    
    Args:
        docs: 文档列表
    
    Returns:
        格式化后的文本
    """
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source_file", doc.metadata.get("source", "未知来源"))
        page = doc.metadata.get("page", "N/A")
        formatted.append(f"[片段 {i}] 来源: {source}, 页码: {page}\n{doc.page_content}")
    
    return "\n\n---\n\n".join(formatted)


def create_rag_chain(retriever, api_key: str):
    """
    使用 LCEL 创建 RAG 问答链
    
    流程: 用户问题 -> 检索相关片段 -> 格式化提示词 -> Gemini 生成回答
    
    Args:
        retriever: 检索器
        api_key: Google API Key
    
    Returns:
        RAG Chain 和 检索 Chain (用于获取来源)
    """
    # 初始化 Gemini 模型
    llm = ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview",
        google_api_key=api_key,
        temperature=0.3,  # 较低的温度，使回答更加准确
        streaming=True,
    )
    
    # 获取提示词模板
    prompt = get_rag_prompt()
    
    # 创建 RAG Chain (使用 LCEL)
    # RunnableParallel 允许并行执行多个操作
    rag_chain = (
        RunnableParallel(
            context=retriever | format_docs,  # 检索并格式化文档
            question=RunnablePassthrough()     # 直接传递问题
        )
        | prompt      # 填充提示词模板
        | llm         # 调用 LLM 生成回答
        | StrOutputParser()  # 解析输出为字符串
    )
    
    return rag_chain, retriever


def print_sources(docs):
    """
    打印检索到的文档来源
    
    Args:
        docs: 检索到的文档列表
    """
    print("\n📚 参考来源:")
    print("-" * 50)
    
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source_file", doc.metadata.get("source", "未知"))
        page = doc.metadata.get("page", "N/A")
        content_preview = doc.page_content[:100].replace("\n", " ")
        if len(doc.page_content) > 100:
            content_preview += "..."
        
        print(f"\n[{i}] 📄 文件: {source}")
        print(f"    📖 页码: {page}")
        print(f"    📝 内容预览: {content_preview}")
    
    print("\n" + "-" * 50)


def interactive_qa(rag_chain, retriever):
    """
    交互式问答循环
    
    Args:
        rag_chain: RAG 问答链
        retriever: 检索器 (用于获取来源)
    """
    print("\n" + "=" * 60)
    print("💬 RAG 问答系统已启动")
    print("=" * 60)
    print("输入你的问题，AI 将基于文档内容回答")
    print("输入 'quit' 或 'exit' 退出")
    print("输入 'sources' 查看上次检索的来源详情")
    print("-" * 60)
    
    last_docs = []  # 保存上次检索的文档
    
    while True:
        try:
            # 获取用户输入
            print("\n")
            question = input("🙋 你的问题: ").strip()
            
            # 检查退出命令
            if question.lower() in ["quit", "exit", "q"]:
                print("\n👋 感谢使用，再见!")
                break
            
            # 检查是否要查看来源
            if question.lower() == "sources":
                if last_docs:
                    print_sources(last_docs)
                else:
                    print("⚠️ 还没有进行过检索")
                continue
            
            # 检查空输入
            if not question:
                print("⚠️ 请输入有效的问题")
                continue
            
            # 1. 先执行检索，获取相关文档
            print("\n🔍 正在检索相关内容...")
            last_docs = retriever.invoke(question)
            
            if not last_docs:
                print("⚠️ 未找到相关文档")
                continue
            
            print(f"   找到 {len(last_docs)} 个相关片段")
            
            # 2. 打印来源摘要
            print("\n📚 参考文档:")
            for i, doc in enumerate(last_docs, 1):
                source = doc.metadata.get("source_file", doc.metadata.get("source", "未知"))
                page = doc.metadata.get("page", "N/A")
                print(f"   [{i}] {source} (页码: {page})")
            
            # 3. 流式输出回答
            print("\n🤖 AI 回答:")
            print("-" * 40)
            
            for chunk in rag_chain.stream(question):
                print(chunk, end="", flush=True)
            
            print("\n" + "-" * 40)
            print("\n💡 提示: 输入 'sources' 可查看详细参考内容")
            
        except KeyboardInterrupt:
            print("\n\n👋 检测到中断，退出程序")
            break
        except Exception as e:
            print(f"\n❌ 发生错误: {e}")
            continue


def main():
    """
    主函数
    """
    print("=" * 60)
    print("🚀 RAG 问答系统 - 初始化")
    print("=" * 60)
    
    # 1. 加载 API Key
    try:
        api_key = load_api_key()
        print("✅ API Key 加载成功")
    except ValueError as e:
        print(e)
        sys.exit(1)
    
    # 2. 加载向量库
    try:
        print("📦 正在加载向量数据库...")
        vector_store = load_vector_store(api_key)
        print(f"✅ 向量库加载成功: {DB_DIR}")
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)
    except Exception as e:
        print(f"❌ 加载向量库失败: {e}")
        sys.exit(1)
    
    # 3. 创建检索器
    print("🔧 正在创建检索器 (k=3)...")
    retriever = create_retriever(vector_store, k=3)
    print("✅ 检索器创建成功")
    
    # 4. 创建 RAG Chain
    print("⛓️ 正在构建 RAG Chain...")
    rag_chain, retriever = create_rag_chain(retriever, api_key)
    print("✅ RAG Chain 构建成功")
    
    # 5. 启动交互式问答
    interactive_qa(rag_chain, retriever)


if __name__ == "__main__":
    main()
