"""
Gemini RAG 项目验证脚本
用于测试 Google Gemini API 连接和流式输出功能
"""

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


def load_api_key() -> str:
    """
    从 .env 文件读取 Google API Key
    
    Returns:
        str: Google API Key
    
    Raises:
        ValueError: 如果 API Key 未设置或为空
    """
    load_dotenv()
    
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key or api_key == "your_api_key_here":
        raise ValueError(
            "请在 .env 文件中设置有效的 GOOGLE_API_KEY\n"
            "获取地址: https://aistudio.google.com/app/apikey"
        )
    
    return api_key


def get_gemini_model(model_name: str = "gemini-3-flash-preview") -> ChatGoogleGenerativeAI:
    """
    初始化 Gemini 模型
    
    Args:
        model_name: 模型名称，默认为 gemini-1.5-flash
    
    Returns:
        ChatGoogleGenerativeAI: LangChain Gemini 聊天模型实例
    """
    api_key = load_api_key()
    
    model = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0.7,
        streaming=True,
    )
    
    return model


def test_streaming():
    """
    测试 Gemini 模型的流式输出功能
    """
    print("=" * 50)
    print("🚀 Gemini RAG 项目 - 连接测试")
    print("=" * 50)
    
    try:
        # 初始化模型
        print("\n📡 正在初始化 Gemini 模型...")
        model = get_gemini_model()
        print("✅ 模型初始化成功!")
        
        # 测试消息
        test_message = "Hello, I am using Gemini for my RAG project. Are you ready?"
        
        print(f"\n📤 发送消息: {test_message}")
        print("\n📥 Gemini 回复 (流式输出):")
        print("-" * 40)
        
        # 流式输出
        for chunk in model.stream(test_message):
            print(chunk.content, end="", flush=True)
        
        print("\n" + "-" * 40)
        print("\n✅ 流式测试完成!")
        print("🎉 你的 Gemini RAG 项目环境已准备就绪!")
        
    except ValueError as e:
        print(f"\n❌ 配置错误: {e}")
    except Exception as e:
        print(f"\n❌ 连接失败: {e}")
        print("请检查:")
        print("  1. API Key 是否正确")
        print("  2. 网络连接是否正常")
        print("  3. API Key 是否有 Gemini API 的访问权限")


if __name__ == "__main__":
    test_streaming()
