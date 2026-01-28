"""
RAG 多轮对话系统 - 支持对话记忆
具备上下文重构能力，能够理解追问并保持对话连贯性
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


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
    加载已存在的 Chroma 向量数据库（复用任务 3 配置）
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
    
    vector_store = Chroma(
        persist_directory=str(DB_DIR),
        embedding_function=embeddings,
        collection_name="rag_documents"
    )
    
    return vector_store


def get_contextualize_prompt() -> ChatPromptTemplate:
    """
    创建上下文重构提示词模板
    用于将追问重写为独立完整的问题
    """
    contextualize_system_prompt = """你是一个问题重构助手。你的任务是根据聊天历史，
将用户的最新问题重写为一个独立、完整、语义明确的问题。

规则：
1. 如果最新问题本身已经很完整，直接返回原问题
2. 如果最新问题是追问（如"那后来呢？"、"还有吗？"、"为什么？"），结合历史对话补全语义
3. 不要回答问题，只需要重写问题
4. 重写后的问题应该可以独立理解，不需要依赖聊天历史

示例：
- 历史：用户问"Python 是什么？"，AI 回答了
- 追问："它有什么优点？"
- 重写为："Python 编程语言有什么优点？"

只输出重写后的问题，不要有任何解释。"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    
    return prompt


def get_qa_prompt() -> ChatPromptTemplate:
    """
    创建 RAG 问答提示词模板
    同时参考检索到的文档和聊天历史
    """
    qa_system_prompt = """你是一个专业的助手，能够根据提供的文档内容和聊天历史来回答问题。

## 重要规则：
1. 优先根据【检索到的文档】回答问题
2. 参考【聊天历史】保持对话连贯性
3. 如果文档中找不到答案，诚实回答："抱歉，根据提供的文档，我无法回答这个问题。"
4. 不要胡编乱造或添加文档中没有的信息
5. 回答要简洁、准确、有条理

## 检索到的文档内容：
{context}

请基于以上信息回答用户的问题。"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    
    return prompt


class RAGWithHistory:
    """
    带有对话记忆的 RAG 系统
    """
    
    def __init__(self, api_key: str, k: int = 3, show_rewritten_query: bool = True):
        """
        初始化 RAG 系统
        
        Args:
            api_key: Google API Key
            k: 检索返回的文档数量
            show_rewritten_query: 是否显示重写后的查询
        """
        self.api_key = api_key
        self.k = k
        self.show_rewritten_query = show_rewritten_query
        
        # 聊天历史（内存存储）
        self.chat_history: list = []
        
        # 初始化组件
        self._init_components()
    
    def _init_components(self):
        """
        初始化所有组件
        """
        # 1. 加载向量库
        self.vector_store = load_vector_store(self.api_key)
        
        # 2. 创建检索器
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.k}
        )
        
        # 3. 初始化 LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-3-flash-preview",
            google_api_key=self.api_key,
            temperature=0.3,
            streaming=True,
        )
        
        # 4. 创建上下文感知检索器（用于重写追问）
        contextualize_prompt = get_contextualize_prompt()
        self.history_aware_retriever = create_history_aware_retriever(
            self.llm,
            self.retriever,
            contextualize_prompt
        )
        
        # 5. 创建问答链
        qa_prompt = get_qa_prompt()
        self.question_answer_chain = create_stuff_documents_chain(
            self.llm,
            qa_prompt
        )
        
        # 6. 组合成完整的 RAG 链
        self.rag_chain = create_retrieval_chain(
            self.history_aware_retriever,
            self.question_answer_chain
        )
        
        # 7. 单独的问题重写链（用于显示重写后的问题）
        self.rewrite_chain = (
            contextualize_prompt 
            | self.llm 
            | StrOutputParser()
        )
    
    def rewrite_question(self, question: str) -> str:
        """
        根据聊天历史重写问题
        
        Args:
            question: 用户原始问题
        
        Returns:
            重写后的完整问题
        """
        if not self.chat_history:
            return question
        
        try:
            rewritten = self.rewrite_chain.invoke({
                "chat_history": self.chat_history,
                "input": question
            })
            return rewritten.strip()
        except Exception:
            return question
    
    def get_relevant_docs(self, question: str) -> list:
        """
        检索相关文档
        
        Args:
            question: 用户问题（已重写）
        
        Returns:
            相关文档列表
        """
        return self.retriever.invoke(question)
    
    def ask(self, question: str) -> tuple[str, list]:
        """
        提问并获取回答
        
        Args:
            question: 用户问题
        
        Returns:
            (回答内容, 参考文档列表)
        """
        # 调用 RAG 链
        response = self.rag_chain.invoke({
            "chat_history": self.chat_history,
            "input": question
        })
        
        answer = response["answer"]
        context_docs = response.get("context", [])
        
        # 更新聊天历史
        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(AIMessage(content=answer))
        
        return answer, context_docs
    
    def ask_stream(self, question: str):
        """
        流式提问
        
        Args:
            question: 用户问题
        
        Yields:
            回答的文本片段
        """
        # 先检索文档
        docs = self.history_aware_retriever.invoke({
            "chat_history": self.chat_history,
            "input": question
        })
        
        # 构建上下文
        context = "\n\n---\n\n".join([
            f"[来源: {doc.metadata.get('source_file', '未知')}, 页码: {doc.metadata.get('page', 'N/A')}]\n{doc.page_content}"
            for doc in docs
        ])
        
        # 流式生成回答
        qa_prompt = get_qa_prompt()
        full_response = ""
        
        for chunk in (qa_prompt | self.llm | StrOutputParser()).stream({
            "chat_history": self.chat_history,
            "context": context,
            "input": question
        }):
            full_response += chunk
            yield chunk
        
        # 更新聊天历史
        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(AIMessage(content=full_response))
        
        # 返回文档供外部使用
        self._last_docs = docs
    
    def clear_history(self):
        """
        清空聊天历史
        """
        self.chat_history = []
        print("🗑️ 聊天历史已清空")
    
    def get_history_summary(self) -> str:
        """
        获取聊天历史摘要
        """
        if not self.chat_history:
            return "暂无聊天记录"
        
        summary = []
        for i, msg in enumerate(self.chat_history):
            role = "🙋 用户" if isinstance(msg, HumanMessage) else "🤖 AI"
            content = msg.content[:50] + "..." if len(msg.content) > 50 else msg.content
            summary.append(f"  {role}: {content}")
        
        return "\n".join(summary)


def print_sources(docs):
    """
    打印参考来源
    """
    print("\n📚 参考来源:")
    print("-" * 50)
    
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source_file", doc.metadata.get("source", "未知"))
        page = doc.metadata.get("page", "N/A")
        content_preview = doc.page_content[:80].replace("\n", " ")
        if len(doc.page_content) > 80:
            content_preview += "..."
        
        print(f"\n[{i}] 📄 {source} (页码: {page})")
        print(f"    {content_preview}")
    
    print()


def interactive_chat():
    """
    交互式多轮对话
    """
    print("=" * 60)
    print("🚀 RAG 多轮对话系统 - 初始化")
    print("=" * 60)
    
    # 1. 加载 API Key
    try:
        api_key = load_api_key()
        print("✅ API Key 加载成功")
    except ValueError as e:
        print(e)
        sys.exit(1)
    
    # 2. 初始化 RAG 系统
    try:
        print("📦 正在初始化 RAG 系统...")
        rag = RAGWithHistory(api_key, k=3, show_rewritten_query=True)
        print("✅ RAG 系统初始化成功")
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        sys.exit(1)
    
    # 3. 启动对话循环
    print("\n" + "=" * 60)
    print("💬 多轮对话已启动")
    print("=" * 60)
    print("命令:")
    print("  - 输入问题进行对话")
    print("  - 'quit' / 'exit' / 'q' - 退出")
    print("  - 'clear' - 清空对话历史")
    print("  - 'history' - 查看对话历史")
    print("  - 'sources' - 查看上次参考来源详情")
    print("-" * 60)
    
    last_docs = []
    
    while True:
        try:
            print("\n")
            question = input("🙋 你: ").strip()
            
            # 处理命令
            if question.lower() in ["quit", "exit", "q"]:
                print("\n👋 感谢使用，再见!")
                break
            
            if question.lower() == "clear":
                rag.clear_history()
                last_docs = []
                continue
            
            if question.lower() == "history":
                print("\n📜 对话历史:")
                print("-" * 40)
                print(rag.get_history_summary())
                print("-" * 40)
                continue
            
            if question.lower() == "sources":
                if last_docs:
                    print_sources(last_docs)
                else:
                    print("⚠️ 还没有进行过检索")
                continue
            
            if not question:
                print("⚠️ 请输入有效的问题")
                continue
            
            # 显示重写后的问题（如果启用）
            if rag.show_rewritten_query and rag.chat_history:
                print("\n🔄 正在理解您的问题...")
                rewritten = rag.rewrite_question(question)
                if rewritten != question:
                    print(f"   重写为: {rewritten}")
            
            # 检索相关内容
            print("\n🔍 正在检索相关内容...")
            
            # 流式输出回答
            print("\n🤖 AI:")
            print("-" * 40)
            
            for chunk in rag.ask_stream(question):
                print(chunk, end="", flush=True)
            
            print("\n" + "-" * 40)
            
            # 保存最后的文档
            last_docs = getattr(rag, '_last_docs', [])
            
            # 显示参考来源摘要
            if last_docs:
                print("\n📚 参考了以下文档:")
                for i, doc in enumerate(last_docs, 1):
                    source = doc.metadata.get("source_file", doc.metadata.get("source", "未知"))
                    page = doc.metadata.get("page", "N/A")
                    print(f"   [{i}] {source} (页码: {page})")
            
            print("\n💡 提示: 可以继续追问，或输入 'sources' 查看详细参考内容")
            
        except KeyboardInterrupt:
            print("\n\n👋 检测到中断，退出程序")
            break
        except Exception as e:
            print(f"\n❌ 发生错误: {e}")
            print("💡 请稍后重试，或检查网络连接和 API 配额")
            continue


if __name__ == "__main__":
    interactive_chat()
