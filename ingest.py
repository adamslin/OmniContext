"""
RAG 离线处理脚本 - PDF 文档向量化
将 data/ 文件夹中的 PDF 文档转化为向量，存储到 db/ 向量数据库中
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma


# 项目路径配置
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
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


def get_pdf_files() -> list[Path]:
    """
    获取 data/ 文件夹下所有 PDF 文件（递归搜索）
    
    Returns:
        PDF 文件路径列表
    """
    if not DATA_DIR.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        
    pdf_files = list(DATA_DIR.rglob("*.pdf"))
    
    return pdf_files


def load_documents(pdf_files: list[Path]) -> list:
    """
    加载所有 PDF 文档
    
    Args:
        pdf_files: PDF 文件路径列表
    
    Returns:
        加载的文档列表
    """
    all_documents = []
    
    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"📄 [{i}/{len(pdf_files)}] 正在加载: {pdf_path.name}")
        
        try:
            loader = PyPDFLoader(str(pdf_path))
            documents = loader.load()
            
            # 为每个文档添加来源信息
            for doc in documents:
                doc.metadata["source_file"] = pdf_path.name
            
            all_documents.extend(documents)
            print(f"   ✅ 成功加载 {len(documents)} 页")
            
        except Exception as e:
            print(f"   ⚠️ 加载失败: {e}")
            continue
    
    return all_documents


def split_documents(documents: list) -> list:
    """
    将文档切分为较小的文本块
    
    使用 RecursiveCharacterTextSplitter 进行智能切分：
    - chunk_size=1000: 每个文本块的最大字符数。较大的值保留更多上下文，
                       但会增加向量化成本和检索时的噪音。
    - chunk_overlap=200: 相邻文本块之间的重叠字符数。重叠可以防止在切分边界
                         处丢失重要的上下文信息，确保跨块的语义连贯性。
    
    Args:
        documents: 原始文档列表
    
    Returns:
        切分后的文本块列表
    """
    print("\n✂️ 正在切分文档...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,      # 每个文本块最大 1000 字符
        chunk_overlap=200,    # 相邻块重叠 200 字符，保持上下文连贯
        length_function=len,
        separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    
    print(f"   📊 原始文档: {len(documents)} 页")
    print(f"   📊 切分结果: {len(chunks)} 个文本块")
    
    return chunks


def create_vector_store(chunks: list, api_key: str) -> Chroma:
    """
    创建向量数据库并存储文档向量
    
    Args:
        chunks: 切分后的文本块列表
        api_key: Google API Key
    
    Returns:
        Chroma 向量数据库实例
    """
    print("\n🔢 正在向量化文档...")
    print(f"   📍 使用模型: models/text-embedding-004")
    
    # 初始化 Google 向量化模型
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=api_key,
    )
    
    # 确保数据库目录存在
    DB_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"   💾 存储位置: {DB_DIR}")
    print(f"   ⏳ 正在处理 {len(chunks)} 个文本块，请稍候...")
    
    # 创建并持久化向量数据库
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(DB_DIR),
        collection_name="rag_documents"
    )
    
    return vector_store


def main():
    """
    主函数：执行完整的 PDF 向量化流程
    """
    print("=" * 60)
    print("🚀 RAG 离线处理 - PDF 文档向量化")
    print("=" * 60)
    
    # 1. 加载 API Key
    try:
        api_key = load_api_key()
        print("✅ API Key 加载成功")
    except ValueError as e:
        print(e)
        sys.exit(1)
    
    # 2. 获取 PDF 文件列表
    pdf_files = get_pdf_files()
    
    if not pdf_files:
        print("\n⚠️ 未找到 PDF 文件!")
        print(f"   请将 PDF 文件放入: {DATA_DIR.absolute()}")
        print("\n💡 提示:")
        print("   1. 将你的 PDF 文件复制到 data/ 文件夹")
        print("   2. 支持子文件夹，脚本会递归搜索所有 PDF")
        print("   3. 重新运行此脚本")
        sys.exit(0)
    
    print(f"\n📚 找到 {len(pdf_files)} 个 PDF 文件:")
    for pdf in pdf_files:
        print(f"   - {pdf.name}")
    
    # 3. 加载文档
    print("\n" + "-" * 40)
    documents = load_documents(pdf_files)
    
    if not documents:
        print("\n❌ 所有 PDF 文件解析失败，请检查文件是否损坏")
        sys.exit(1)
    
    # 4. 切分文档
    print("-" * 40)
    chunks = split_documents(documents)
    
    if not chunks:
        print("\n❌ 文档切分失败，没有生成任何文本块")
        sys.exit(1)
    
    # 5. 向量化并存储
    print("-" * 40)
    try:
        vector_store = create_vector_store(chunks, api_key)
    except Exception as e:
        print(f"\n❌ 向量化失败: {e}")
        print("   请检查 API Key 和网络连接")
        sys.exit(1)
    
    # 6. 完成
    print("\n" + "=" * 60)
    print("🎉 向量化处理完成!")
    print("=" * 60)
    print(f"\n📊 处理统计:")
    print(f"   - PDF 文件数: {len(pdf_files)}")
    print(f"   - 原始页数: {len(documents)}")
    print(f"   - 文本块数: {len(chunks)}")
    print(f"   - 数据库位置: {DB_DIR.absolute()}")
    print("\n✅ 现在可以开始使用 RAG 检索功能了!")


if __name__ == "__main__":
    main()
